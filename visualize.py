import torch
import gym
import time
import numpy as np
from network import wrap_dqn, ReplayBuffer, DQN, DQN_Conv

use_gpu = True

if torch.cuda.is_available() and use_gpu:
    available_device = torch.device('cuda')
    print("Using cuda")
else:
    available_device = torch.device('cpu')
    print("Using cpu")

policy_net_path = "./runs/pong-policy.pt"
target_net_path = "./runs/pong-target.pt"

print("Create environment.")
env = gym.make("PongNoFrameskip-v4")
env = wrap_dqn(env)

learning_rate = 0.0001
input_channels = env.observation_space.shape[0]
input_size = env.observation_space.shape[1]
n_output = env.action_space.n

print("Loading saved networks from file")
policy_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(available_device)
target_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(available_device)
target_net.load_state_dict(policy_net.state_dict())

policy_net.load_state_dict(torch.load(policy_net_path, map_location=available_device))
target_net.load_state_dict(torch.load(target_net_path, map_location=available_device))
print("Done.")

try:
    print("\nRunning tests")
    test_rewards, test_duration = [], []
    episode_reward, episode_it = 0, 0
    s = env.reset()
    k, nr_tests = 0, 10
    while k < nr_tests:
        env.render()
        time.sleep(0.01)
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

print("Rewards: ", test_rewards)                
print("Duration: ", test_duration)
print("Mean test reward: %5.2f \t Mean test duration: %5.2f" % (np.mean(test_rewards), np.mean(test_duration)))

env.close()