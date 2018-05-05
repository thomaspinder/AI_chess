from learning.online import evalutation as ev
from utilities.constants import *

testing = ev.Evaluator(games_to_play)
testing.test()
testing.write_results('test.csv')
testing.print_results()