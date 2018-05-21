import chess.pgn
import pandas as pd
import re
import glob


def fen_expander(board_string, mapping_dict):
    board_string = board_string.split(' ', 1)[0]
    numbers = list(set(re.findall('\d', board_string)))
    for number in numbers:
        board_string = board_string.replace(str(number), 'e'*int(number))
    for k,v in mapping_dict.items():
        board_string = board_string.replace(k, v)
    return board_string


notation = list('prnbqkPRNBQKe')
mappings = {}
i = 0
for item in notation:
    mappings[item] = str(i*'0') + str('1') + str(int(len(notation)-int(i)-1) * '0')
    i += 1

pgn = open('learning/offline/data/2013_01.pgn')

number_of_games = 10

i = 0
j = 0
training_data = []
while i < number_of_games:
    game = chess.pgn.read_game(pgn)
    board = game.board()
    to_play = 0 # 0 = White
    if int(game.headers['BlackElo']) > 2200 and int(game.headers['WhiteElo']) > 2200:
        for move in game.main_line():
            training_data.append((fen_expander(board.fen(), mappings), to_play, str(move)))
            if to_play == 0:
                to_play = 1
            board.push(move)
        j += 1
    i += 1
    if (i*10/number_of_games)%1 == 0:
        print('{}% Processed'.format(100*i/number_of_games))

print('{} games parsed'.format(j))
results_df = pd.DataFrame(training_data)
results_df.columns = ['board', 'moving_colour', 'move']
print('{} Training Items Retrieved'.format(results_df.shape[0]))
results_df.to_csv('train.csv', index=False)