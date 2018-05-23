import numpy as np
import random
from anytree import Node, LevelGroupOrderIter, RenderTree
from utilities import evaluators as bev
from copy import deepcopy
from utilities.constants import *
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='tree_search.log')



class MCTS:
    def __init__(self, board_env, explorer, board_evaluator):
        """
        An implementation of Monte-Carlo Tree Search for game play. An exploration rate 0.7 is set, as per the advice
        of Martin Mueller (https://scholar.google.com/scholar_url?url=https://link.springer.com/chapter/10.1007/978-3-642-12993-3_6&hl=en&sa=T&oi=gsb&ct=res&cd=0&ei=ME_oWqmzLovCmgH0j73gAQ&scisig=AAGBfm0KuSPQZ3FzXJi9WyFY7n-xi0nrmA)
        :param board_env: The game's board
        :param exploration: C Parameter for UCT. 0.7 s
        """
        self.board = board_env
        self.initial_node = {'visit_count': 0, 'win_count': 0, 'total_reward': 0} # Starting params for every node
        self.root = Node('0', board = self.board, action = None, **self.initial_node)
        self.max_player = board_env.current_player
        self.current_depth = 0
        self.max_depth = Parameters.uct_max_depth
        self.explorer = explorer
        self.maximisation = True
        self.evaluation_nn = board_evaluator
        # self.opponent = opponent.Adversary(verbose=False, search_depth=2, max_think=1)
        # self.opponent.initialise_engine(self.board.board)
        self.board_evaluator = bev.Simple()

    def board_evaluate(self, node):
        evaluation = self.board_evaluator.evaluate(node.board.board.fen())
        return evaluation

    # def ucb_heuristic(self, candidate_list, explore_param, parent_node):
    #     baseline = self.board_evaluate(parent_node)
    #     total_visits = np.sum([node.visit_count for node in candidate_list])
    #     means = np.array(
    #         [node.total_reward / node.visit_count + np.random.uniform(low=0.01, high=0.05) for node in candidate_list])
    #     cis = np.array([np.sqrt(2 * np.log(node.visit_count) / total_visits) for node in candidate_list])
    #     # TODO: Should it be baseline - evaluation?
    #     evaluations = np.array([baseline - self.board_evaluate(node) for node in candidate_list]) # TODO: Should the standardisation be across all boards or as currently with the possible max value?
    #     uct_values = means + evaluations + explore_param * cis
    #     max_value = np.argmax(uct_values)
    #     return candidate_list[max_value]

    # def ucb1(self, candidate_list, explore_param):
    #     total_visits = np.sum([node.visit_count for node in candidate_list])
    #     means = np.array([node.total_reward / node.visit_count + np.random.uniform(low=0.01, high=0.05) for node in candidate_list])
    #     cis = np.array([np.sqrt(2*np.log(node.visit_count)/total_visits) for node in candidate_list])

    def mini_maxi_update(self):
        """
        Potential legacy function if we wish to select the lowest scoring action for adversary.
        :return:
        """
        if self.maximisation:
            self.maximisation=False
        else:
            self.maximisation=True

    def selection(self):
        # Root the tree and initialise maximisation
        root_node = self.root
        self.maximisation = True
        self.current_depth = 0
        # Test all of the nodes potential children
        while root_node.children and self.current_depth < self.max_depth:
            children_nodes = list(root_node.children)

            # Update the boards current legal moves
            root_node.board.update_l_moves()

            # Test if every legal move has been already explored. If not, explore the unexplored
            potential_actions = root_node.board.l_moves
            explored = [child_node.action for child_node in children_nodes]
            if set(potential_actions) > set(explored):
                logging.info('Unexplored Child Nodes Still Exist')
                logging.info('Expanding {}'.format(root_node.name))
                return root_node

            # If every move has been taken, proceed down the tree
            else:
                logging.info('{} Expanded'.format(root_node.name))
                # Update root node based upon optimal MAB heuristics evaluation
                root_node = self.explorer.evaluate(children_nodes)
                logging.info('Expansion Yielded {}'.format(root_node.name))
                # Update to reflect whether maximisation or minimisation should take place next time around
                self.mini_maxi_update()
            self.current_depth += 1
        return root_node

    def expansion(self, parent):
        # Check for any unexplored child nodes
        parent.board.update_l_moves()
        explored = [node.action for node in list(parent.children)]
        to_explore = [action for action in parent.board.l_moves if action not in explored]

        # If any do exist, explore expand them
        if len(to_explore) > 0:
            move = random.choice(to_explore)

            # Add a node corresponding to this move to the game tree
            sim_game = deepcopy(parent.board)
            sim_game.board.push(move)
            child_node = Node(name='{}_{}'.format(parent.name, len(parent.children)), parent=parent, action = move,
                              board = sim_game, **self.initial_node)
            return child_node
        # In the absence of unexplored nodes, return the parent
        else:
            return parent

    def evaluate(self, node, n_simulation):
        # TODO: Add score bonus to differentiate between a good and bad win. - Martin Mueller
        win_count = 0
        reward = 0
        i = 0
        while i < n_simulation:
            current_board = deepcopy(node.board)
            current_board.player_update()
            active = current_board.current_player
            while current_board.active:
                if Parameters.nn_policy:
                    current_board.play_nn()
                else:
                    current_board.play_random()
                current_board.terminal()
                current_board.update_methods()

            # Determine Winner and store result in node
            result = current_board.board.result()
            if result == '1-0':
                win_count += 1
                reward += 1
                logging.info('From Node: {}, Result: {}'.format(node.name, 'Win'))
            elif result == '1-0':
                reward -= 1
                logging.info('From Node: {}, Result: {}'.format(node.name, 'Loss'))
            else:
                logging.info('From Node: {}, Result: {}'.format(node.name, 'Draw'))
            if Parameters.nn_evaluation:
                # TODO: How does the nn print predictions? Is it a 1x3 array?
                win_prob = self.evaluation_nn.predict(current_board.board.fen())
                reward += win_prob
            i += 1
        return win_count, reward

    @staticmethod
    def backprop(node, visits, win_count, reward):
        node.visit_count += visits
        node.win_count += win_count
        node.total_reward += reward

        # Now loop through all parents and their respective parents
        for parent in node.ancestors:
            parent.visit_count += visits
            parent.win_count += win_count
            parent.total_reward += reward

    def search(self, nits, tree_simulations, verbose = False):
        i = 0
        root_node = self.root
        logging.info('Tree Search Beginning From: {}'.format(root_node.board.board.fen()))
        for i in tqdm(range(nits)):
            root_node = self.selection()
            expand_node = self.expansion(root_node)
            win_count, total_reward = self.evaluate(expand_node, tree_simulations)
            self.backprop(expand_node, tree_simulations, win_count, total_reward)
        if verbose:
            self.print_tree()

    def get_best_action(self):
        all_nodes = list(LevelGroupOrderIter(self.root))
        if all_nodes:
            direct_children = all_nodes[1]
            best_node = self.explorer.best(direct_children)
            # TODO: Append board value to node, rather than calculating within uct
            # best_node = self.ucb_heuristic(children_nodes, self.exploration, root_node)

            return best_node.action

    def print_tree(self):
        for pre, fill, node in RenderTree(self.root):
            print('{}{}, mean_reward: {}, Visits: {}, Wins: {}'.format(pre, node.action,
                                                             np.round(node.total_reward/node.visit_count, 5),
                                                             node.visit_count, node.win_count))
