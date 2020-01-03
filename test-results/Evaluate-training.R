library('forecast')
library('ggplot2')
library('gridExtra')
theme_set(theme_bw())

mav <- function(x,n){as.numeric(filter(x,rep(1/n,n), sides=1))}


### Load base DQN runs
setwd('D:/Dokumente/Uni/Deep-Learning/project-pong/runs/base-dqn-runs/results')
training_list = list()
for (i in 1:10) {
  training_list[[i]] = read.csv(paste('training_results_', i, ".csv", sep = ""))
  training_list[[i]]$Run = sprintf("Run %02d", i)
  training_list[[i]]$Reward.smooth = mav(training_list[[i]]$Reward, 5)
  if(i == 1){
    training = training_list[[i]]
  } else {
    training = rbind(training, training_list[[i]])
  }
}

p.train <- 
  ggplot(data = training, aes(x=Completed_at, y = Reward.smooth, color = Run)) +
  geom_line(size = 1) +
  labs(title = "Basic DQN",
       x = "Training iteration",
       y = "Smoothed reward")



### Load DDQN runs
setwd('D:/Dokumente/Uni/Deep-Learning/project-pong/runs/ddqn-runs/results')
training_list = list()
for (i in 1:10) {
  training_list[[i]] = read.csv(paste('training_results_', i, ".csv", sep = ""))
  training_list[[i]]$Run = sprintf("Run %02d", i)
  training_list[[i]]$Reward.smooth = mav(training_list[[i]]$Reward, 5)
  if(i == 1){
    training_ddqn = training_list[[i]]
  } else {
    training_ddqn = rbind(training_ddqn, training_list[[i]])
  }
}

p.train.ddqn <- 
  ggplot(data = training_ddqn, aes(x=Completed_at, y = Reward.smooth, color = Run)) +
  geom_line(size = 1) +
  labs(title = "Double DQN",
       x = "Training iteration",
       y = "Smoothed reward")



### Load DDQN runs with priority replay
setwd('D:/Dokumente/Uni/Deep-Learning/project-pong/runs/prio-ddqn-runs/results')
training_list = list()
for (i in 1:10) {
  training_list[[i]] = read.csv(paste('training_results_', i, ".csv", sep = ""))
  training_list[[i]]$Run = sprintf("Run %02d", i)
  training_list[[i]]$Reward.smooth = mav(training_list[[i]]$Reward, 5)
  if(i == 1){
    training_prio = training_list[[i]]
  } else {
    training_prio = rbind(training_prio, training_list[[i]])
  }
}

p.train.prio <- 
  ggplot(data = training_prio, aes(x=Completed_at, y = Reward.smooth, color = Run)) +
  geom_line(size = 1) +
  labs(title = "Double DQN with prioritized experience replay",
       x = "Training iteration",
       y = "Smoothed reward")



grid.arrange(p.train, p.train.ddqn, p.train.prio)
