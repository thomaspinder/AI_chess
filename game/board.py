import chess
import random
import logging
logging.basicConfig(filename="board.log", level=logging.INFO)

class Env:
    def __init__(self, starting_pos=None):
        if starting_pos is None:
            self.board = chess.Board()
        else:
            self.board = chess.Board(fen=starting_pos)
        self.l_moves = [move for move in self.board.legal_moves]
        self.active = True
        self.moves = []
        self.current_player = 'white'
        self.move_count = 0
        self.no_legals = False

    def update_l_moves(self):
        self.l_moves = [move for move in self.board.legal_moves]

    def checkmate(self):
        """
        Is the current game in a checkmated position
        :return: Bool. False indicates checkmate
        """
        """
        :return: 
        """
        if self.board.is_checkmate():
            self.active=False
        return self.board.is_checkmate()

    def player_update(self):
        """
        Which player is currently playing
        :return: Str
        """
        if not self.board.turn:
            self.current_player='b'
        else:
            self.current_player='w'

    def parse_player(self):
        """
        Print the current player out in a readable format
        :return: Str
        """
        if self.current_player=='w':
            return 'White'
        else:
            return 'Black'

    def stalemate(self):
        """
        Is the current game stalemated
        :return: Bool. False indicates stalemate has occurred.
        """
        if self.board.is_stalemate():
            self.active=False
        return self.board.is_stalemate()

    def insufficient(self):
        """
        Is there enough material on the board to continue playing
        :return: Bool. False indicates insufficient material to continue play.
        """
        if self.board.is_insufficient_material():
            self.active=False
        return self.board.is_insufficient_material()

    def terminal(self):
        """
        Amalgamate the possible game terminations and determine if the game should continue
        :return: Bool. True indicates game has ended
        """
        if self.board.is_game_over() or self.no_legals==True:
            self.active = False
        self.no_legals==False
        return self.active

    def print_board(self):
        """
        Print the current board's state
        :return: Console string
        """
        print(self.board)

    def update_methods(self):
        self.move_count = len(self.moves)
        self.l_moves = self.legal_actions()

    def legal_actions(self):
        return [move for move in self.board.legal_moves]

    def make_move(self, move, verbose=False):
        """
        Moves a piece
        :param move_seal:
        :return:
        """
        self.player_update()
        if verbose:
            print('{} to move'.format(self.current_player))
        move_hold = chess.Move.from_uci(move)
        if move_hold in self.board.legal_moves:
            self.board.push(move_hold)
            self.moves.append(move_hold)
        else:
            print('Invalid move selection')
        self.terminal()
        self.update_methods()

    def single_random_move(self):
        self.update_l_moves()
        return random.choice(self.l_moves)

    def play_random(self, verbose = False):
        if verbose:
            print(self.board)
        curr = False
        while curr is False:
            self.update_l_moves()
            if len(self.l_moves) > 0:
                self.board.push(random.choice(self.l_moves))
                self.update_methods()
                if verbose:
                    print(self.board)
                curr = self.board.is_game_over()
                if verbose:
                    print(self.board.result())
                    print('-'*60)
            else:
                self.active = False
                logging.info('No Legal Moves: {}'.format(self.board.fen()))
                self.no_legals = True
                curr=True