import random
import numpy as np
import gym
import torch
import time
from datetime import datetime
import csv
import argparse
import os

from network import ReplayBuffer, DQN, DQN_Conv, AVAILABLE_DEVICE
from wrappers import wrap_dqn



########## Process input arguments and prepare save file ##########
parser = argparse.ArgumentParser(description='Train a neural network for Pong.')
parser.add_argument('-s', '--seed', help='Set the random seed for training', type=int)
parser.add_argument('--load_networks', help = 'Load the networks ./policy-net.pt and ./target-net.pt', action='store_true')
args = parser.parse_args()

SAVE_FOLDER = "runs/pong_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"
try:
    os.makedirs(SAVE_FOLDER)
except FileExistsError:
    # directory already exists
    pass



########## Setup environment ##########
env = wrap_dqn(gym.make("PongNoFrameskip-v4"))
val_env = wrap_dqn(gym.make("PongNoFrameskip-v4"))

# Set seeds
if(args.seed != None):
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.np_random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



########## Set hyperparameters ##########
num_iterations = 500000
batch_size = 32
learning_rate = 0.0001
gamma = 0.99
replay_buffer_capacity = 100000
replay_init_size = 10000
epsilon = 1.0
epsilon_final = 0.05
epsilon_final_reached = 100000
epsilon_decay = (epsilon - epsilon_final)/epsilon_final_reached
target_update_frequency = 1000

validation_frequency = 50000
validation_count = 10
save_frequency = 100000
do_validation = True



########## Define policy and target networks ##########
input_channels = env.observation_space.shape[0]
input_size = env.observation_space.shape[1]
n_output = env.action_space.n

policy_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(AVAILABLE_DEVICE)
target_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(AVAILABLE_DEVICE)
target_net.load_state_dict(policy_net.state_dict())



########## Prefill replay buffer ##########
if(args.load_networks):
    print("Loading saved networks from file")
    policy_net.load_state_dict(torch.load('policy-net.pt'))
    target_net.load_state_dict(torch.load('target-net.pt'))

    # We have a (somewhat) working net already -> Use network to prefill buffer
    print("Prefilling replay buffer")
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    s = env.reset()
    for i in range(replay_init_size):
        with torch.no_grad():
            s_tensor = torch.as_tensor(s, device = AVAILABLE_DEVICE).float()
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



########## Begin training ##########
print("Starting training")
completed_at, rewards, losses, durations_it, durations_sec = [], [], [], [], []
val_completed_at, val_rewards, val_durations_it, val_durations_sec = [], [], [], []
test_completed_at, test_rewards, test_durations_it, test_durations_sec = [], [], [], []
episode_reward, episode_loss, episode_it, episode_starttime = 0, 0, 0, time.time()
starttime = time.time()
final_iteration = 0

s = env.reset()
try:
    for i in range(num_iterations):
        final_iteration = i
        # Do one gradient step
        batch = replay_buffer.sample(batch_size)
        ss = torch.as_tensor(batch[0], device = AVAILABLE_DEVICE).float()
        aa = torch.as_tensor(batch[1], device = AVAILABLE_DEVICE)
        ss1 = torch.as_tensor(batch[2], device = AVAILABLE_DEVICE).float()
        rr = torch.as_tensor(batch[3], device = AVAILABLE_DEVICE)
        ddone = torch.as_tensor(batch[4], device = AVAILABLE_DEVICE)
        
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

        # Decrease epsilon
        if(epsilon > epsilon_final):
            epsilon -= epsilon_decay
        
        # Add new sample to buffer
        if (np.random.uniform() < epsilon):
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                s_tensor = np.array(s, float).reshape((1, input_channels, input_size, input_size))
                s_tensor = torch.as_tensor(s_tensor, device = AVAILABLE_DEVICE).float()
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
            
            completed_at.append(i+1)
            rewards.append(episode_reward)
            losses.append(episode_loss)
            durations_it.append(episode_it)
            durations_sec.append(time.time()-episode_starttime)

            print("%d: Episode %d completed with reward %d in %5.2f iterations. Mean reward (last 100): %5.2f, speed: %5.2f" % 
                    (i+1, len(rewards), episode_reward, episode_it, np.mean(rewards[-100:]), episode_it/durations_sec[-1]))
                    
            episode_reward, episode_loss, episode_it, episode_starttime = 0, 0, 0, time.time()


        ### Do validation
        if (do_validation and (i+1)%validation_frequency == 0):
            val_episode_reward, val_episode_it, val_episode_starttime = 0, 0, time.time()
            s_val = val_env.reset()
            k = 0
            while k < validation_count:
                with torch.no_grad():
                    s_tensor = np.array(s_val, float).reshape((1, input_channels, input_size, input_size))
                    s_tensor = torch.as_tensor(s_tensor, device = AVAILABLE_DEVICE).float()
                    a = policy_net.forward(s_tensor).argmax().item()
                s1, r, done, _ = env.step(a)
                val_episode_reward += r
                val_episode_it += 1
                s_val = s1
                if(done):
                    val_completed_at.append(i+1)
                    val_rewards.append(val_episode_reward)
                    val_durations_it.append(val_episode_it)
                    val_durations_sec.append(time.time()-val_episode_starttime)
                    val_episode_reward, val_episode_it, val_episode_starttime = 0, 0, time.time()
                    k += 1
                    s = env.reset()
            print("%d: Validation over %d episodes completed with mean reward %d in %5.2f iterations on average." % 
                    (i+1, k, np.mean(val_rewards[-k:]), np.mean(val_durations_it[-k:])))

                

        # Save the results periodically
        if((i+1)%save_frequency == 0):
            torch.save(policy_net.state_dict(), SAVE_FOLDER + str(i+1) + '_policy-net.pt')
            torch.save(target_net.state_dict(), SAVE_FOLDER + str(i+1) + '_target-net.pt')

            training_results = zip(*[completed_at, rewards, losses, durations_it, durations_sec])
            with open(SAVE_FOLDER + 'training_results.csv', 'w') as csvfile:
                wr = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
                wr.writerow(['Completed_at', 'Reward', 'Loss', 'Duration_It', 'Duration_Sec'])
                wr.writerows(training_results)
            
            validation_results = zip(*[val_completed_at, val_rewards, val_durations_it, val_durations_sec])
            with open(SAVE_FOLDER + 'validation_results.csv', 'w') as csvfile:
                wr = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
                wr.writerow(['Completed_at', 'Reward', 'Duration_It', 'Duration_Sec'])
                wr.writerows(validation_results)
            

except KeyboardInterrupt:
    print('Training interrupted early.')


########## Save final results ##########
torch.save(policy_net.state_dict(), SAVE_FOLDER + 'final_policy-net.pt')
torch.save(target_net.state_dict(), SAVE_FOLDER + 'final_target-net.pt')

training_results = zip(*[completed_at, rewards, losses, durations_it, durations_sec])
with open(SAVE_FOLDER + 'training_results.csv', 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    wr.writerow(['Completed_at', 'Reward', 'Loss', 'Duration_It', 'Duration_Sec'])
    wr.writerows(training_results)

validation_results = zip(*[val_completed_at, val_rewards, val_durations_it, val_durations_sec])
with open(SAVE_FOLDER + 'validation_results.csv', 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    wr.writerow(['Completed_at', 'Reward', 'Duration_It', 'Duration_Sec'])
    wr.writerows(validation_results)

endtime = time.time()
print("Finished training. Completed %d episodes in %s." % (len(rewards), str(endtime - starttime)))



########## Run tests ##########
try:
    print("\nRunning tests")
    val_episode_reward, val_episode_it, val_episode_starttime = 0, 0, time.time()
    s_val = val_env.reset()
    k = 0
    while k < 5*validation_count:
        with torch.no_grad():
            s_tensor = np.array(s_val, float).reshape((1, input_channels, input_size, input_size))
            s_tensor = torch.as_tensor(s_tensor, device = AVAILABLE_DEVICE).float()
            a = policy_net.forward(s_tensor).argmax().item()
        s1, r, done, _ = env.step(a)
        val_episode_reward += r
        val_episode_it += 1
        s_val = s1
        if(done):
            val_completed_at.append(final_iteration)
            val_rewards.append(val_episode_reward)
            val_durations_it.append(val_episode_it)
            val_durations_sec.append(time.time()-val_episode_starttime)
            val_episode_reward, val_episode_it, val_episode_starttime = 0, 0, time.time()
            k += 1
            s = env.reset()

except KeyboardInterrupt:
    print('Testing interrupted early.')    
                
print("%d: Testing over %d episodes completed with mean reward %d in %5.2f iterations on average." % 
        (final_iteration, k, np.mean(val_rewards[-(k+1):]), np.mean(val_durations_it[-k:])))

validation_results = zip(*[val_completed_at, val_rewards, val_durations_it, val_durations_sec])
with open(SAVE_FOLDER + 'validation_results.csv', 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    wr.writerow(['Completed_at', 'Reward', 'Duration_It', 'Duration_Sec'])
    wr.writerows(validation_results)

env.close()