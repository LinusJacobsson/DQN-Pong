import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Union

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Transition named tuple for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args: Union[np.ndarray, torch.Tensor]) -> None:
        """Store a transition in replay memory."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return the current size of the replay memory."""
        return len(self.memory)


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


def optimize(
    policy_dqn: DQN,
    target_dqn: DQN,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Optimize the policy DQN using a batch of transitions from replay memory.
    
    Args:
        policy_dqn: The policy DQN to optimize.
        target_dqn: The target DQN used for computing target Q-values.
        memory: Replay memory containing past transitions.
        optimizer: Optimizer for updating the policy network.
        device: The device (CPU or GPU) to use for computations.
    """
    batch_size: int = policy_dqn.batch_size

    if len(memory) < batch_size:
        return

    # Sample a batch of transitions
    transitions: List[Transition] = memory.sample(batch_size)
    batch: Transition = Transition(*zip(*transitions))

    # Prepare tensors
    state_batch: torch.Tensor = torch.cat(batch.state)
    action_batch: torch.Tensor = torch.cat(batch.action)
    reward_batch: torch.Tensor = torch.cat(batch.reward)
    next_state_batch: torch.Tensor = torch.cat(batch.next_state)
    done_batch: torch.Tensor = torch.cat(batch.done)

    # Compute Q(s_t, a)
    state_action_values: torch.Tensor = policy_dqn.forward(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for next states
    with torch.no_grad():
        next_state_values: torch.Tensor = target_dqn.forward(next_state_batch).max(1)[0].unsqueeze(1)
        next_state_values[done_batch] = 0.0  # Zero for terminal states

    # Compute expected Q-values
    expected_state_action_values: torch.Tensor = reward_batch + policy_dqn.gamma * next_state_values

    # Compute loss
    criterion: nn.SmoothL1Loss = nn.SmoothL1Loss()
    loss: torch.Tensor = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1)
    optimizer.step()
