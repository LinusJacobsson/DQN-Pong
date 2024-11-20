import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Union

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, config: dict) -> None:
        super(DQN, self).__init__()

        # Read from config
        self.batch_size: int = config['batch_size']
        self.gamma: float = config['gamma']
        self.eps_start: float = config['eps_start']
        self.eps_end: float = config['eps_end']
        self.decay_steps: int = config['decay_steps']
        self.n_actions: int = config['n_actions']

        # Define layers
        self.conv1: nn.Conv2d = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2: nn.Conv2d = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3: nn.Conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.flatten: nn.Flatten = nn.Flatten()
        self.fc1: nn.Linear = nn.Linear(3136, 512)
        self.fc2: nn.Linear = nn.Linear(512, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network."""
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, state: torch.Tensor, steps_done: int, exploit: bool = False) -> torch.Tensor:
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current state.
            steps_done: Number of steps taken in the environment.
            exploit: If True, use greedy policy; otherwise, apply epsilon-greedy.
        
        Returns:
            Selected action index as a tensor.
        """
        epsilon: float = max(self.eps_end, self.eps_start - steps_done / self.decay_steps)
        if exploit or random.random() > epsilon:
            with torch.no_grad():
                q_index: torch.Tensor = self.forward(state).max(1)[1].view(1, 1)
                return q_index
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
