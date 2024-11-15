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
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
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
    
     # Refactored with linear decay and index return
    def act(self, state, steps_done, exploit=False):
        epsilon = max(self.eps_end, self.eps_start - steps_done / self.decay_steps)
        if exploit or random.random() > epsilon:
            with torch.no_grad():
                q_index = self.forward(state).max(1)[1].view(1, 1)
                return q_index
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
    

def optimize(policy_dqn, target_dqn, memory, optimizer, device):
    batch_size = policy_dqn.batch_size

    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    # Compute Q(s_t, a)
    state_action_values = policy_dqn.forward(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for next states
    with torch.no_grad():
        next_state_values = target_dqn.forward(next_state_batch).max(1)[0].unsqueeze(1)
        next_state_values[done_batch] = 0.0 # Should be zero for terminal states

    # Compute the exptected Q values 
    expected_state_action_values = reward_batch + policy_dqn.gamma * next_state_values
    
    # Compute loss and optimize
    loss = nn.SmoothL1Loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    # Gradient clip to stabilize learning
    torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1)
    optimizer.step()