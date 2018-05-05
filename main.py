from learning.online import evalutation as ev
import chess
from utilities.constants import *

b = chess.Board()

testing = ev.Evaluator(games_to_play)
testing.test()
testing.write_results('test.csv')
testing.print_results()