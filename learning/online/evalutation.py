from game import board, agent, opponent
import pandas as pd
import chess.svg
import chess.pgn
from IPython.display import SVG
from learning.online import uct
from utilities.constants import *


# TODO: Fix stuck endgame with just a horse and king.etc
class Evaluator:
    def __init__(self, number_of_games, starting_pos=None):
        self.n = number_of_games
        self.results = {}
        self.results_df = None
        self.start_fen = starting_pos
        print(self.start_fen)

    def test(self, summary=False, verbose=False):
        i = 0
        while i < self.n:
            if self.start_fen is None:
                chessboard = board.Env()
            else:
                chessboard = board.Env(starting_pos=self.start_fen)
                print(chessboard.board)
            player = agent.Agent('white')
            opp = opponent.Adversary(verbose=a_verbose, search_depth=a_depth, max_think=a_think, nodes = stockfish_nodes)
            opp.initialise_engine(chessboard.board)

            while chessboard.active and chessboard.move_count < max_moves:
                if chessboard.move_count % 2 == 0:
                    # move = player.move_random(chessboard)
                    move = player.move_uct(chessboard, 2)
                    print('MOVE: {}'.format(move))
                else:
                    #move = opp.calculate(chessboard)
                    move = chessboard.single_random_move()
                chessboard.make_move(str(move))
                if verbose:
                    chessboard.print_board()
                    print('-'*60)
            pgn_write = open('results/game_pgns/game_{}.pgn'.format(i), 'at')
            game = chess.pgn.Game.from_board(chessboard.board)
            game.headers['Event'] = 'Game_{}'.format(i)
            pgn_write.write(str(game)+'\n\n')
            pgn_write.close()

            self.results['game{}'.format(i)] = [chessboard.parse_player(), chessboard.move_count]
            i += 1
            print('{} games completed.'.format(i))

    def _format_results(self):
        self.results_df = pd.DataFrame(self.results, columns=self.results).transpose().reset_index()
        self.results_df.columns = ['game', 'winner', 'move_count']

    def print_results(self):
        if self.results_df is None:
            self._format_results()
        print('-'*60)
        print(self.results_df)

    def write_results(self, filename):
        """
        Write the results to a flat csv file
        :param filename: the file to be created and then contain results
        :return: csv file
        """
        if self.results_df is None:
            self._format_results()
        self.results_df.to_csv('results/{}'.format(filename), index=False)

    def return_results(self):
        if self.results_df is None:
            self._format_results()
        return self.results_df
