import random 
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def __len__(self):
        return len(self.memory)
    
    def push(self, observation, action, next_observation, reward, terminated):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (observation, action, next_observation, reward, terminated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))
    

class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()

        # Read from config
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.eps_start = config['eps_start']
        self.eps_end = config['eps_end']
        self.decay_steps = config['decay_steps']
        self.n_actions = config['n_actions']
        
        # Define layers
        self.conv1 = nn.Conv2d(4, 32, kernal_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
    def act(self, observation, step, exploit=False):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * max(0, (self.decay_steps - step) / self.decay_steps)
        if exploit or random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(observation)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)