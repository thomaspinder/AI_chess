from learning.online import uct
import random
from utilities.constants import *

class Agent:
    def __init__(self, colour):
        """
        :param colour: The agents piece colour. 0 = white, 1 = black.
        :param available_moves: A list of the possible legal moves for the player
        """
        self.colour = colour
        self.available_moves = None

    def get_moves(self, board):
        """
        Given the current board's state, update the agent's list of available, legal moves.
        :param board: Board object
        :return: Refree list of legal moves
        """
        self.available_moves = [move for move in board.legal_moves]

    def move_random(self, board):
        """
        Instruct the Agent to randomly select a move from a list of current legal moves
        :param board: The current board
        :return: A single move
        """
        self.get_moves(board.board)
        return random.choice(self.available_moves)

    def move_uct(self, board, exploration_param):
        """
        Calculate and select an action using UCT
        :param board: A board, from the Board object
        :return:
        """
        tree = uct.MCTS(board, exploration_param)
        tree.search(uct_sims, tree_sims, True)
        reccomendation = tree.get_best_action()
        return reccomendation