library(dplyr)
library(broom)
library(stringr)
library(reshape2)
library(ggplot2)

rm(list=ls())
directory = '/home/tpin3694/Documents/python/AI_chess/results/times//'
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

results <- results %>% 
  filter(value >0)

nn_results <- results %>% 
  filter(nn_status=='MCTS-NN')
no_nn_results <- results %>% 
  filter(nn_status=='MCTS')

cpu_lm <- lm(value ~ 0 + simulation + as.factor(nn_status), data = results)
nn_lm <- lm(value ~ 0 + simulation, data = nn_results)
no_nn_lm <- lm(value ~ 0 + simulation, data = no_nn_results)
summary(cpu_lm)
summary(nn_lm)
summary(no_nn_lm)

nn_lm_poly <- lm(value ~ 0 + poly(simulation, degree = 2), data = nn_results)
no_nn_lm_poly <- lm(value ~ 0 + poly(simulation, degree=2), data = no_nn_results)
cpu_lm_poly <- lm(value ~ 0 + simulation + as.factor(nn_status), data = results)
summary(nn_lm_poly)
summary(no_nn_lm_poly)

results %>% 
  ggplot(aes(x=jitter(simulation), y=value, colour=nn_status)) +
  geom_point(alpha=0.5) +
  stat_smooth() + 
  labs(x='Jittered Simulation Value', y = 'Simulation Times (secs)', title='Simulation Count against Simulation Time Stratified by Algorithm', colour='Algorithm') +
  theme_minimal()

results %>% 
  ggplot(aes(nn_status, value, fill=nn_status)) +
  geom_violin(alpha = 0.8, adjust=1.5) +
  labs(x='Algorithm', y = 'Simulation Time (secs)', title='Distribution of Times split by Algortithm', fill='Algorithm') +
  theme_minimal()

