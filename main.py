from learning.online import evalutation as ev
from game import opponent as op
from utilities.constants import *
import pandas as pd
from learning.offline import load_model as lm

end_games = True
results_df = pd.DataFrame(columns=['moves_to_mate', 'starting_variation', 'game', 'winner', 'move_count'])
opp = op.Adversary(verbose=Parameters.a_verbose, search_depth=Parameters.a_depth,
                                 max_think=Parameters.a_think, nodes=Parameters.stockfish_nodes)
nn_evaluator = lm.NeuralNet('learning/offline/chess_ann.h5')
if Parameters.nn_evaluation:
    nn_bool = 'with_nn'
else:
    nn_bool = 'wout_nn'
print('Objects Loaded.')
if end_games:
    for k, v in Parameters.end_games.items():
        print(k, v)
        i = 0
        for position in v:
            print('Game Count: {}'.format(Parameters.games_to_play))
            testing = ev.Evaluator(Parameters.games_to_play, Parameters.exploration_alg, nn_evaluator, position)
            testing.test()
            # testing.write_results('{}_{}_test.csv'.format(k, i))
            testing.print_results()
            results_temp = testing.return_results()
            results_temp['moves_to_mate'] = k
            results_temp['starting_variation'] = i
            i += 1
        results_df = results_df.append(results_temp)
    results_df['sims'] = Parameters.uct_sims
    results_df['exploration_alg'] = Parameters.exploration_alg.name
    results_df.to_csv('results/{}_{}_results.csv'.format(Parameters.exploration_alg.name, nn_bool), index=False)
else:
    i=0
    testing = ev.Evaluator(Parameters.games_to_play, Parameters.exploration_alg, nn_evaluator)
    testing.test()
    # testing.write_results('{}_{}_test.csv'.format(k, i))
    testing.print_results()
    results_temp = testing.return_results()
    results_df = results_df.append(results_temp)
    results_df['sims'] = Parameters.uct_sims
    results_df['exploration_alg'] = Parameters.exploration_alg.name
    results_df.to_csv('results/openings.csv'.format(Parameters.exploration_alg.name, nn_bool), index=False)