from learning.online import evalutation as ev
from utilities.constants import *
import pandas as pd

results_df = pd.DataFrame(columns=['moves_to_mate', 'starting_variation', 'game','winner','move_count'])

for k,v in end_games.items():
    print(k, v)
    i = 0
    for position in v:
        testing = ev.Evaluator(games_to_play, position)
        testing.test()
        #testing.write_results('{}_{}_test.csv'.format(k, i))
        testing.print_results()
        results_temp = testing.return_results()
        results_temp['moves_to_mate'] = k
        results_temp['starting_variation'] = i
        i += 1
    results_df = results_df.append(results_temp)
results_df['sims']=uct_sims
results_df['exploration_rate']= exploration_rate
results_df.to_csv('results/uct_results.csv', index=False)