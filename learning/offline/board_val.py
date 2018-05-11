import chess.pgn
import re


pgn = open('/home/tpin3694/Documents/python/AI_chess/learning/offline/data/2013_01.pgn')

number_of_games = 10

i = 0
j = 0
training_data = []

to_move = {'w':0, 'b':1}
notation = list('prnbqkPRNBQKe')
castle_status = list('QKqk')


def castle_bool(character, string):
    if character in string:
        return 1
    else:
        return 0


def evaluate_fen(fen_string, to_move_dict, notation_list, castling_list, board):
    components = fen_string.split(' ')
    # Initialise list and store the player to move
    representation = [to_move_dict[components[1]]]
    # Add in castling status - 4 elements
    representation.extend(castle_bool(piece, components[2]) for piece in castling_list)
    # Add in piece status
    representation.extend([len(re.findall(char, fen_string)) for char in notation_list])
    # Get pieces and their position
    for k, v in board.piece_map().items():
        representation.append(str(v))
        representation.append(k) # TODO: Convert position -k, to a1, b4.etc?
    print(representation)



while i < number_of_games:
    game = chess.pgn.read_game(pgn)
    board = game.board()
    to_play = 0 # 0 = White
    if int(game.headers['BlackElo']) > 1500 and int(game.headers['WhiteElo']) > 1500:
        for move in game.main_line():
            evaluate_fen(board.fen(), to_move, notation, castle_status, board)
            board.push(move)
        j += 1
    i += 1
    if (i*10/number_of_games)%1 == 0:
        print('{}% Processed'.format(100*i/number_of_games))