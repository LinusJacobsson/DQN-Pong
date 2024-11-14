import random 
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from collections import deque, namedtuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Replay Memory class
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    

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



def optimize(policy_dqn, target_dqn, memory, optimizer, device):
    if len(memory) < policy_dqn.batch_size:
        return
    
    observation, action, next_observation, reward, terminated = memory.sample(batch_size=policy_dqn.batch_size)

    # Stack previous observations
    observation_batch = torch.stack(observation).to(device)
    next_observation_batch = torch.stack(next_observation).to(device)
    action_batch = torch.tensor(action, device=device, dtype=torch.int8).unsqueeze(1)
    reward_batch = torch.tensor(reward, device=device, dtpye=torch.float32)
    terminated_batch = torch.tensor(terminated, device=device, dtype=torch.int8)

    q_values = policy_dqn(observation_batch).gather(1, action_batch).squeeze()
    with torch.no_grad():
        max_next_q_values = target_dqn(next_observation_batch).max(1)[0]
        q_value_targets = reward_batch + (1 - terminated_batch) * (policy_dqn.gamma * max_next_q_values)

    loss = F.mse_loss(q_values, q_value_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()