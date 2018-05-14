import re


class Simple:
    def __init__(self):
        self.piece_list = list('prnbqkPRNBQK')
        self.values = [100, 500, 300, 300, 900, 0, -100, -500, -300, -300, -900, 0]
        self.mappings = dict(zip(self.piece_list, self.values))

    def evaluate(self, fen_string, standardise = True):
        board = fen_string.split(' ')[0]
        board_value = 0
        for k, v in self.mappings.items():
            board_value += len(re.findall(k, board))*self.mappings[k]
        if standardise:
            board_value /= 3900  # The value of the board for a complete white piece set and only a black king
        return board_value
