import torch
import gym
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from network import ReplayBuffer, DQN, DQN_Conv, AVAILABLE_DEVICE
from wrappers import wrap_dqn_standard

render_env = True
show_input = False

policy_net_path = "./runs/visualize_policy-net.pt"
target_net_path = "./runs/visualize_target-net.pt"

print("Create environment.")
env = wrap_dqn_standard(gym.make("PongDeterministic-v4"))

learning_rate = 0.0001
input_channels = env.observation_space.shape[0]
input_size = env.observation_space.shape[1]
n_output = env.action_space.n

print("Loading saved networks from file")
policy_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(AVAILABLE_DEVICE)
target_net = DQN_Conv(input_channels, input_size, n_output, learning_rate).to(AVAILABLE_DEVICE)
target_net.load_state_dict(policy_net.state_dict())

policy_net.load_state_dict(torch.load(policy_net_path, map_location=AVAILABLE_DEVICE))
target_net.load_state_dict(torch.load(target_net_path, map_location=AVAILABLE_DEVICE))
print("Done.")

def draw_preprocessed_input(s):
    image = np.zeros((168, 168))
    image[0:84, 0:84] = s[3]
    image[84:168, 0:84] = s[2]
    image[0:84, 84:168] = s[1]
    image[84:168, 84:168] = s[0]
    image = np.transpose(image)
    plt.imshow(image, cmap='gray')
    plt.draw()
    plt.pause(0.001)
    #input("Press [enter] to continue.")

if(show_input):
    plt.figure(figsize=(2, 2))
    plt.axis('off')
    plt.ion()
    plt.show()

try:
    print("\nRunning tests")
    test_rewards, test_duration = [], []
    episode_reward, episode_it = 0, 0
    s = env.reset()
    k, nr_tests = 0, 100
    it = 0
    while k < nr_tests:
        if(render_env):
            env.render()
            time.sleep(0.01)
        with torch.no_grad():
            s_tensor = s.reshape((1, input_channels, input_size, input_size))
            s_tensor = torch.as_tensor(s_tensor, device = AVAILABLE_DEVICE)
            quality = policy_net.forward(s_tensor)
            a = quality.argmax().item()
        s1, r, done, _ = env.step(a)
        if(show_input and it % 50 == 0):
            print(quality)
            draw_preprocessed_input(np.array(s))
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
    it = it + 1
except KeyboardInterrupt:
    print('Testing interrupted early.')    

print("Rewards: ", test_rewards)                
print("Duration: ", test_duration)
print("Mean test reward: %5.2f \t Mean test duration: %5.2f" % (np.mean(test_rewards), np.mean(test_duration)))

env.close()