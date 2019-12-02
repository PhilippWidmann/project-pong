from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    AVAILABLE_DEVICE = torch.device('cuda')
    print("Using cuda")
else:
    AVAILABLE_DEVICE = torch.device('cpu')
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
                nn.Linear(n_input, n_hidden_1).to(AVAILABLE_DEVICE),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_hidden_2).to(AVAILABLE_DEVICE),
                nn.ReLU(),
                nn.Linear(n_hidden_2, n_hidden_3).to(AVAILABLE_DEVICE),
                nn.ReLU(),
                nn.Linear(n_hidden_3, n_output).to(AVAILABLE_DEVICE),
            )
        elif(n_hidden_2 > 0):        
            self.layers = nn.Sequential(
                nn.Linear(n_input, n_hidden_1).to(AVAILABLE_DEVICE),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_hidden_2).to(AVAILABLE_DEVICE),
                nn.ReLU(),
                nn.Linear(n_hidden_2, n_output).to(AVAILABLE_DEVICE),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(n_input, n_hidden_1).to(AVAILABLE_DEVICE),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_output).to(AVAILABLE_DEVICE),
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