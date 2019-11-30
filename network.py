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

use_gpu = True

if torch.cuda.is_available() and use_gpu:
    available_device = torch.device('cuda')
    print("Using cuda")
else:
    available_device = torch.device('cpu')
    print("Using cpu")

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)
        
    def sample(self, k):
        samples = random.sample(self.buffer, k)
        samples = [list(i) for i in zip(*samples)]
        # Transform lazy frames
        samples[0] = [np.array(s, float, copy=False) for s in samples[0]]
        samples[2] = [np.array(s, float, copy=False) for s in samples[2]]
        for i in range(5):
            samples[i] = np.array(samples[i], copy=False)
        return samples
    
    def add(self, new_sample):
        self.buffer.append(new_sample)
        
    def count(self):
        return len(self.buffer)
    
class DQN(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output, learning_rate):
        super(DQN, self).__init__()
        if(n_hidden_3 > 0):        
            self.layers = nn.Sequential(
                nn.Linear(n_input, n_hidden_1).to(available_device),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_hidden_2).to(available_device),
                nn.ReLU(),
                nn.Linear(n_hidden_2, n_hidden_3).to(available_device),
                nn.ReLU(),
                nn.Linear(n_hidden_3, n_output).to(available_device),
            )
        elif(n_hidden_2 > 0):        
            self.layers = nn.Sequential(
                nn.Linear(n_input, n_hidden_1).to(available_device),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_hidden_2).to(available_device),
                nn.ReLU(),
                nn.Linear(n_hidden_2, n_output).to(available_device),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(n_input, n_hidden_1).to(available_device),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_output).to(available_device),
            )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fct = nn.SmoothL1Loss()
    
    def forward(self, x):
        return self.layers(x)
    
    def loss(self, q_outputs, q_targets):
        #return 0.5 * torch.sum(torch.pow(q_outputs - q_targets, 2))
        return self.loss_fct(q_outputs.float(), q_targets.float())
        
    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

class DQN_Conv(nn.Module):
    def __init__(self, input_channels, input_size, n_output, learning_rate):
        super(DQN_Conv, self).__init__()

        kernels = [8,4,3]
        strides = [4,2,1]
        channels = [32, 64, 64]

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=kernels[0], stride=strides[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=kernels[1], stride=strides[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=kernels[2], stride=strides[2]),
            nn.ReLU()
        )
        conv_0_size = (input_size-kernels[0])/strides[0] + 1
        conv_1_size = (conv_0_size-kernels[1])/strides[1] + 1
        conv_2_size = (conv_1_size-kernels[2])/strides[2] + 1
        assert(conv_2_size.is_integer())
        conv_out_count = channels[2] * int(conv_2_size)**2

        self.lin = nn.Sequential(
            nn.Linear(conv_out_count, 512),
            nn.ReLU(),
            nn.Linear(512, n_output)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.loss_fct = nn.SmoothL1Loss()
        self.loss_fct = nn.MSELoss()
    
    def loss(self, q_outputs, q_targets):
        return self.loss_fct(q_outputs.float(), q_targets.float())
    

    def forward(self, x):
        x = x.float() / 256
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x

# Wrappers mostly copy-pasted from 
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*k, shp[1], shp[2]), dtype=np.float32)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def wrap_dqn(env, stack_frames=4, reward_clipping=True):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    #env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    #if 'FIRE' in env.unwrapped.get_action_meanings():
    #    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, stack_frames)
    #if reward_clipping:
    #    env = ClippedRewardsWrapper(env)
    return env
