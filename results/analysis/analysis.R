library(dplyr)
library(broom)
library(stringr)
library(reshape2)
library(ggplot2)

rm(list=ls())
directory = '/home/tpin3694/Documents/python/AI_chess/results/full/clean/'
setwd(directory)
file_list = list.files()

# Read in data
results <- data.frame()
for (file in file_list){
  to_append <- ifelse(grepl('no_nn', file), 'MCTS', 'MCTS-NN')
  result_set <- read.csv(paste(directory, '/', file, sep = ''))
  result_set$nn_status <- to_append
  results <- rbind(results, result_set)
}

results$mate_count <- factor(results$mate_count, levels = c('mate_in_1', 'mate_in_2', 'mate_in_3', 'mate_in_4', 'mate_in_5'))

times <- results %>% 
  filter(value > 0)

nn_times <- times %>%
  group_by(nn_status) %>% 
  summarise(mean = mean(value), var = var(value), n=n())

nn_times_w <- times %>% 
  filter(nn_status=='Present') %>% 
  select(value)

nn_times_wout <- times %>% 
  filter(nn_status!='Present') %>% 
  select(value)

# t.test(nn_times_w, nn_times_wout, var.equal = FALSE)

results <- results %>% 
  filter(value > 0) %>% 
  mutate(winner_bool = ifelse(winner=='White', 'Win', 'Draw')) %>% 
  select(-value) %>% 
  unique()

print(head(results))

# Proportion of puzzles solved
number_of_wins <- results %>% 
  filter(winner == 'White') %>% 
  nrow()*100/nrow(results)

number_of_wins <- results %>% 
  group_by(winner) %>% 
  summarise(total = n()) %>% 
  mutate(prop = total/sum(total))


print('Percentage of overall wins')
print(number_of_wins)


wins_by_puzzle <- results %>% 
  group_by(mate_count, winner_bool) %>% 
  summarise(total = n()) %>% 
  mutate(prop = total/sum(total))

print('Wins By Puzzle:')
print(wins_by_puzzle)

wis_by_nn <- results %>% 
  group_by(nn_status, winner_bool) %>% 
  summarise(total = n()) %>% 
  mutate(prop = total/sum(total),
         se = sqrt((prop*(1-prop))/total))

print('Wins by network status:')
print(wis_by_nn)

wis_by_nn %>% 
  ggplot(aes(x = nn_status, y = prop, fill=winner_bool)) +
  geom_bar(stat='identity', position ='dodge') +
  geom_errorbar(aes(ymin=prop-se, ymax=prop+se), position=position_dodge(.9), width=0.4)+
  labs(x='Algorithm', y = 'Result Proportion', title = 'Result Proportions by Neural Network Presence', fill='Game Result')+
  theme_minimal()

wins_by_nn_puz <- results %>% 
  mutate(mate_count=as.factor(mate_count)) %>% 
  group_by(nn_status, winner_bool, mate_count) %>% 
  summarise(total = n()) %>%
  group_by(mate_count) %>% 
  mutate(prop = total/sum(total),
         se = sqrt((prop*(1-prop))/total))

print('Wins by network status and puzzle:')
print(wins_by_nn_puz)

wins_by_nn_puz %>% 
  group_by(winner_bool, mate_count) %>% 
  summarise(tot = sum(prop),
            total=sum(total),
            se = sqrt((tot*(1-tot))/total)) %>% 
  ggplot(aes(x = mate_count, y = tot, fill=winner_bool)) +
  geom_bar(stat='identity', position='dodge') +
  geom_errorbar(aes(ymin=tot-se, ymax=tot+se),position=position_dodge(.9), width=0.4)+
  labs(x='Checkmate Puzzle', y = 'Result Proportion', title='Result Proportions by Mate Puzzle', fill='Game Status')+
  theme_minimal()

wins_by_nn_puz %>% 
  group_by(winner_bool, mate_count) %>% 
  summarise(tot = sum(total))

prop.test(table(results$nn_status, results$winner_bool))

mab_split <- results %>% 
  group_by(winner_bool, exp_alg) %>% 
  summarise(total = n()) %>% 
  group_by(exp_alg) %>% 
  mutate(prop = total/sum(total),
         se = sqrt((prop*(1-prop))/sqrt(total)))

mab_split %>% 
  ggplot(aes(x = exp_alg, y = prop, fill=winner_bool)) +
  geom_bar(stat='identity', position='dodge') +
  geom_errorbar(aes(ymin=prop-se, ymax=prop+se),position=position_dodge(.9), width=0.4)+
  labs(x='MAB Heuristic', y = 'Result Proportion', title='Result Proportions by MAB Heuristic', fill='Game Status')+
  theme_minimal()


table(results$exp_alg, results$winner_bool)
