from collections import deque
import time
import random
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.grad as grad
import torch.optim as optim
import torch.nn.functional as func
import cv2

from network import wrap_dqn, ReplayBuffer, DQN, DQN_Conv

use_gpu = True

if torch.cuda.is_available() and use_gpu:
    available_device = torch.device('cuda')
    print("Using cuda")
else:
    available_device = torch.device('cpu')
    print("Using cpu")




##################### BEGIN TRAINING ###########################

run_in_colab = False
load_networks = False

if(run_in_colab):
    from google.colab import drive
    drive.mount('/content/gdrive')
    policy_net_path = "/content/gdrive/My Drive/Colab Notebooks/pong-policy.pt"
    target_net_path = "/content/gdrive/My Drive/Colab Notebooks/pong-target.pt"
else:
    policy_net_path = "/home/philipp/Dokumente/AAAUniversitaet/Deep-Learning/Reinforcement-learning/pong-policy.pt"
    target_net_path = "/home/philipp/Dokumente/AAAUniversitaet/Deep-Learning/Reinforcement-learning/pong-target.pt"

# Setup environment
env = gym.make("PongNoFrameskip-v4")
env = wrap_dqn(env)

# Set seeds
seed = 42
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
env.seed(seed)
env.action_space.np_random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Set hyperparameters
num_epochs = 5000000
batch_size = 32
learning_rate = 0.0001
gamma = 0.99 #0.95
replay_buffer_capacity = 100000 #100000
replay_init_size = 10000
epsilon = 0.9 #1.0
epsilon_final = 0.02 #0.05
epsilon_final_reached = 90000 #100000 # ebert 50000 #100000
epsilon_decay = (epsilon - epsilon_final)/epsilon_final_reached
target_update_frequency = 1000 #ebert 5000 #1000
# tau = 0.01
validation_frequency = 100#10000
save_frequency = 100000
do_validation = False



# Define policy and target networks
input_channels = env.observation_space.shape[0]
input_size = env.observation_space.shape[1]
n_output = env.action_space.n


policy_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(available_device)
target_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(available_device)
target_net.load_state_dict(policy_net.state_dict())


if(load_networks):
    print("Loading saved networks from file")
    policy_net.load_state_dict(torch.load(policy_net_path))
    target_net.load_state_dict(torch.load(target_net_path))

    # We have a (somewhat) working net already -> Use network to prefill buffer
    print("Prefilling replay buffer")
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    s = env.reset()
    for i in range(replay_init_size):
        with torch.no_grad():
            s_tensor = torch.as_tensor(s, device = available_device).float()
            a = policy_net.forward(s_tensor).argmax().item()
        s1, r, done, _ = env.step(a)
        replay_buffer.add([s,a,s1,r,done])
        s = s1
        if(done):
            s = env.reset()
            done = False
else:
    # Prefill the replay buffer randomly
    print("Prefilling replay buffer")
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    s = env.reset()
    for i in range(replay_init_size):
        a = env.action_space.sample()
        s1, r, done, _ = env.step(a)
        replay_buffer.add([s,a,s1,r,done])
        s = s1
        if(done):
            s = env.reset()
            done = False

# Start training
print("Starting training")
losses, rewards, episode_duration, episode_time = [], [], [], []
episode_loss, episode_reward, episode_it, episode_starttime = 0, 0, 0, time.time()
completed_at_last_validation = 0
s = env.reset()
starttime = time.time()
try:
    for i in range(num_epochs):    
        # Do one gradient step
        batch = replay_buffer.sample(batch_size)
        ss = torch.as_tensor(batch[0], device = available_device).float()
        aa = torch.as_tensor(batch[1], device = available_device)
        ss1 = torch.as_tensor(batch[2], device = available_device).float()
        rr = torch.as_tensor(batch[3], device = available_device)
        ddone = torch.as_tensor(batch[4], device = available_device)
        
        policy_net.optimizer.zero_grad()
        Q = policy_net.forward(ss)
        q_policy = Q[range(len(aa)), aa]
        
        with torch.no_grad():
            q_target = rr + gamma * target_net.forward(ss1).max(dim=1)[0] * (~ ddone)
            #aa1 = target_net.forward(ss1).argmax(dim=1)
            #q_target = rr + gamma * target_net.forward(ss1)[range(len(aa1)), aa1] * (~ ddone)
            
        loss = policy_net.loss(q_policy, q_target)
        loss.backward()
        policy_net.optimizer.step()
        
        # Update target network parameters from policy network parameters
        if((i+1)%target_update_frequency == 0):
            target_net.load_state_dict(policy_net.state_dict())
        #target_net.update_params(policy_net.state_dict(), tau)

        # Decrease epsilon
        if(epsilon > epsilon_final):
            epsilon -= epsilon_decay
        
        # Add new sample to buffer
        if (np.random.uniform() < epsilon):
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                s_tensor = np.array(s, float).reshape((1, input_channels, input_size, input_size))
                s_tensor = torch.as_tensor(s_tensor, device = available_device).float()
                a = policy_net.forward(s_tensor).argmax().item()

        s1, r, done, _ = env.step(a)
        replay_buffer.add([s, a, s1, r, done])
        s = s1
            
        episode_it += 1
        episode_loss += loss.item()
        episode_reward += r

        if(done):
            done = False
            s = env.reset()

            episode_duration.append(episode_it)
            losses.append(episode_loss/episode_it)
            rewards.append(episode_reward)
            episode_time.append(time.time()-episode_starttime)

            print("%i: Episode %d completed with reward %d in %5.2f iterations. Mean reward (last 100): %5.2f, speed: %5.2f" % 
                    (i+1, len(rewards), episode_reward, episode_it, np.mean(rewards[-100:]), episode_it/episode_time[-1]))
                    
            episode_loss, episode_reward, episode_it, episode_starttime = 0, 0, 0, time.time()

        if ((i+1)%validation_frequency == 0):
            #print("Iteration ", i+1, " finished after ", time.time()-starttime)

            if do_validation:
                validation_rewards, validation_duration = [], []
                episode_reward, episode_it = 0, 0
                s = env.reset()
                k = 0
                while k < 10:
                    with torch.no_grad():
                        s_tensor = np.array(s, float).reshape((1, input_channels, input_size, input_size))
                        s_tensor = torch.as_tensor(s_tensor, device = available_device).float()
                        a = policy_net.forward(s_tensor).argmax().item()
                    s1, r, done, _ = env.step(a)
                    episode_reward += r
                    episode_it += 1
                    s = s1
                    if(done):
                        validation_duration.append(episode_it)
                        validation_rewards.append(episode_reward)
                        episode_reward, episode_it = 0, 0
                        done = False
                        k += 1
                        s = env.reset()
                
                print("%i: Episodes completed: %d \t Mean training reward: %5.2f \t Mean validation reward: %5.2f \t Mean normalized loss: %5.2f \t Mean training duration: %5.2f \t Mean validation duration: %5.2f" % 
                    (i+1, len(rewards[completed_at_last_validation:]), np.mean(rewards[completed_at_last_validation:]), np.mean(validation_rewards), np.mean(losses[completed_at_last_validation:]), np.mean(episode_duration[completed_at_last_validation:]), np.mean(validation_duration)))
            #else:
            #    print("%i: Episodes completed: %d \t Mean training reward: %5.2f \t Mean normalized loss: %5.2f \t Mean training duration: %5.2f" % 
            #        (i+1, len(rewards[completed_at_last_validation:]), np.mean(rewards[completed_at_last_validation:]), np.mean(losses[completed_at_last_validation:]), np.mean(episode_duration[completed_at_last_validation:])))
            #
            #completed_at_last_validation = len(rewards)
        
        # Save the networks intermittently
        if((i+1)%save_frequency == 0):
            torch.save(policy_net.state_dict(), policy_net_path)
            torch.save(target_net.state_dict(), target_net_path)            
except KeyboardInterrupt:
    print('Training interrupted early.')

torch.save(policy_net.state_dict(), policy_net_path)
torch.save(target_net.state_dict(), target_net_path)   

endtime = time.time()
print("Finished training. Completed %d episodes in %s." % (len(rewards), str(endtime - starttime)))


# Run tests:
try:
    print("\nRunning tests")
    test_rewards, test_duration = [], []
    episode_reward, episode_it = 0, 0
    s = env.reset()
    k, nr_tests = 0, 10
    while k < nr_tests:
        with torch.no_grad():
            s_tensor = np.array(s, float).reshape((1, input_channels, input_size, input_size))
            s_tensor = torch.as_tensor(s_tensor, device = available_device).float()
            a = policy_net.forward(s_tensor).argmax().item()
        s1, r, done, _ = env.step(a)
        episode_reward += r
        episode_it += 1
        s = s1
        if(done):
            test_duration.append(episode_it)
            test_rewards.append(episode_reward)
            episode_reward, episode_it = 0, 0
            done = False
            k += 1
            s = env.reset()
except KeyboardInterrupt:
    print('Testing interrupted early.')    
                
print("Mean test reward: %5.2f \t Mean test duration: %5.2f" % (np.mean(test_rewards), np.mean(test_duration)))

env.close()