import numpy as np
import random
from scipy.stats import beta
from anytree import Node, LevelGroupOrderIter, RenderTree
from copy import deepcopy
import time
from game import board


# https://github.com/RomainSa/mcts

def average_wins(plays, wins, ties):
    """
    Simple average
    :param plays: number of times the arm has been played
    :param wins: number of successes
    :param ties: number of ties
    :return: score (min:0, max:1)
    """
    if plays == 0:
        return 99.   # to be sure that arm is played at least once
    else:
        return wins / plays


def ucb1(plays, wins, ties, total_plays, c_=1.0):
    """
    Upper Confidence Bound score
    :param plays: number of times the arm has been played
    :param ties: number of ties
    :param wins: number of successes
    :param total_plays: number of plays of all arms
    :param c_: constant (the more the larger the bound)
    :return: score (min:0, max:1)
    """
    if plays == 0:
        return 99.   # to be sure that arm is played at least once
    else:
        score = wins / plays + c_ * np.sqrt(np.log(total_plays) / plays)
        score = min(score, 1.0)
        score += np.random.rand() * 1e-6  # small random perturbation to avoid ties
        return score


def thompson(plays, wins, ties):
    """
    Thompson sampling
    :param plays: number of times the arm has been played
    :param wins: number of successes
    :param ties: number of ties
    :return: score (min:0, max:1)
    """
    if plays == 0:
        return 99.   # to be sure that arm is played at least once
    else:
        return beta.rvs(a=wins+1, b=plays-wins+1, size=1)[0]


class MCTS:
    def __init__(self, board):
        self.board = board
        self.nodes_initial = {'visits':0, 'wins':0, 'stalemates':0, 'reward':0}
        self.root = Node('0', board=self.board, **self.nodes_initial)

    def get_action(self):
        # Root the tree
        node = self.root
        while node.children:
            # Determine expansion
            legal_actions = [move for move in node.board.legal_moves]
            if len(legal_actions) > len(node.children):
                return node
            else:
                # Descend
                nodes = node.children
                scores = []
                for node in nodes:
                    n_plays = node.parent.n_plays
                    node.score = ucb1(node.visits, node.wins, node.stalemates, n_plays, 2)
                    scores.append(node.score)
                node = nodes[np.argmax(scores)]
        return node

    def expand(self, parent):
        # Remove any previously explored nodes
        # Indexing here could cause an error
        print('Already Exlpored: {}'.format([node.board.move_stack for node in parent.children]))
        explored = [node.board.move_stack for node in parent.children] # Will be empty on first run
        unexplored = [move for move in parent.board.legal_moves if move not in explored]
        if len(unexplored) > 0:
            # May be wrong
            random_move = random.choice(unexplored)
            sim_board = deepcopy(parent.board)
            sim_board.push(random_move)
            child_name = '{}_{}'.format(parent.name, len(parent.children))
            child = Node(name=child_name,
                         parent=parent,
                         board=sim_board,
                         **self.nodes_initial)
            node = child
        else:
            node = parent
        return node

    @staticmethod
    def result_parser(result):
        result = result.strip()
        if result == '1-0':
            return 1
        elif result == '1/2-1/2':
            return 0
        elif result == '0-1':
            return -1
        else:
            print('Unknown Result Found')

    def simulate(self, node, nits):
        wins = 0
        stalemates = 0
        losses = 0
        i = 0
        while i < nits:
            current_board = deepcopy(node.board)
            while not current_board.is_game_over():
                l_moves = [move for move in current_board.legal_moves]
                random_move = random.choice(l_moves)
                current_board.push(random_move)
            if self.result_parser(current_board.result()) == 1:
                wins += 1
                print('Win')
            elif self.result_parser(current_board.result()) == 0:
                stalemates += 1
                print('Draw')
            elif self.result_parser(current_board.result()) == -1:
                losses += 1
                print('Loss')
            i += 1
        return wins, stalemates

    # TODO: Backprop losses as well
    def backprop(self, node, visits, wins, stalemates):
        node.visits += visits
        node.wins += wins
        node.stalemates += stalemates

        for ancestor in node.ancestors:
            ancestor.visits += visits
            ancestor.wins += wins
            ancestor.stalemates += stalemates

    def print_tree(self, return_string=False, level=-1):
        def sort_by_move(nodes):
            return sorted(nodes, key = lambda x: x.board.moves[-2])

        result = ['\n']
        output = '%s%s | Last Move: %s | %s wins, %s losses, %s stalemates in %s games | Score: %.3f'

        selection = []
        if level > 0:
            selection = [n for sub in list(LevelGroupOrderIter(self.root))[:level+1] for n in sub]

        for indent, _, node in RenderTree(self.root, childiter=sort_by_move):
            if level == -1 or node in selection:
                result.append((output % (indent,
                                         node.name,
                                         node.board.moves[-2],
                                         node.wins,
                                         node.stalemates,
                                         node.visits,
                                         node.score)))
        result = '\n'.join(result)
        if return_string:
            return result
        print(result)

    def parse(self, depth, max_runtime, n_sims=1, print_tree=False):
        i = 0
        start_time = time.time()
        end_time = time.time() + max_runtime

        node = self.root
        while i < depth and time.time() < end_time:
            node = self.get_action()
            expansion = self.expand(parent=node)
            wins, stalemates = self.simulate(node=expansion, nits=n_sims)
            self.backprop(node=expansion, visits=n_sims, wins=wins, stalemates=stalemates)
            if print_tree:
                self.print_tree()
            i += 1

    def best_play(self):
        node_list = list(LevelGroupOrderIter(self.root))
        if node_list:
            first_layer = node_list[1]
            # Select most visited node
            scores = [n.visits for n in first_layer]
            optimum_action = first_layer[np.argmax(scores)]
            print(optimum_action.board)
            return optimum_action.board.moves[-2]


if __name__=='__main__':
    board_env = board.Env()
    uct = MCTS(board_env.board)
    print('Tree Initialised')
    uct.parse(depth=15, max_runtime=3, n_sims=1)
    print('Best next move: {}'.format(uct.best_play()))