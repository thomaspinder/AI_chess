library(dplyr)
library(broom)
library(stringr)
library(reshape2)
library(ggplot2)

adhoc_data <- data.frame(input = readLines('/home/tpin3694/Documents/python/AI_chess/results/UCB_wout_nn_400results.csv')) 

adhoc_split <- data.frame(str_split_fixed(adhoc_data$input, ',', 8)) %>% 
  `colnames<-`(c('game', 'move_count','move_times', 'mate_count', 'starting_var', 'winner', 'simulation', 'exp_alg')) %>% 
  filter(row_number()!=1) %>% 
  select(-move_times, move_times) %>% 
  mutate(move_times = gsub('\\[|\\]', '', move_times)) 

cleaned <- data.frame(str_split_fixed(adhoc_split$move_times, '\\|', 50))

final <- cbind(subset(adhoc_split, select = -move_times), cleaned)

times <- melt(final, id.vars = c('game', 'move_count', 'mate_count', 'starting_var', 'winner', 'simulation', 'exp_alg')) %>% 
  mutate(value = as.numeric(value),
         value = ifelse(is.na(value), 0, value),
         expected = 2*(as.numeric(gsub('mate_in_', '', mate_count)))-1,
         difference = as.numeric(as.character(move_count)) - expected) 

write.csv(times, '/home/tpin3694/Documents/python/AI_chess/results/times/ucb_no_nn_400.csv', row.names = FALSE)
