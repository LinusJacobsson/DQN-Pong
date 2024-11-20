import torch
import logging
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from typing import Any, Union
from collections import deque, namedtuple
import random
# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess(state: Union[np.ndarray, torch.Tensor], env: str) -> torch.Tensor:
    """
    Preprocess the input state for the given environment.

    Args:
        state: The raw state from the environment.
        env: The name of the environment.

    Returns:
        Processed state as a PyTorch tensor.
    """
    if env in ['ALE/Pong-v5']:
        state = np.asarray(state)  # Ensure the state is a NumPy array
        return torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
    else:
        raise NotImplementedError("You are using an unsupported environment!")


def setup_logger(log_dir: str) -> logging.Logger:
    """
    Set up a logger to log training or evaluation events.

    Args:
        log_dir: Directory where the log file will be created.

    Returns:
        Configured logger instance.
    """
    logger: logging.Logger = logging.getLogger('DQN')
    logger.setLevel(logging.INFO)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path: str = os.path.join(log_dir, 'training.log')

    # File handler
    fh: logging.FileHandler = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    # Stream handler
    ch: logging.StreamHandler = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter for logs
    formatter: logging.Formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logging setup complete...")
    return logger


def evaluate_policy(
    dqn: Any,
    env_eval: gym.Env,
    n_eval_episodes: int,
    render: bool = False
) -> float:
    """
    Evaluate the performance of the policy network.

    Args:
        dqn: The trained DQN model.
        env_eval: The evaluation environment.
        n_eval_episodes: Number of episodes to evaluate.
        render: Whether to render the environment during evaluation.

    Returns:
        The mean reward over the evaluated episodes.
    """
    ACTION_SPACE: list[int] = [2, 3]
    total_rewards: list[float] = []

    for _ in range(n_eval_episodes):
        state, _ = env_eval.reset()
        state = torch.from_numpy(np.array(state)).unsqueeze(0).to(device, dtype=torch.float32)
        total_reward: float = 0.0
        done: bool = False

        while not done:
            with torch.no_grad():
                action_index: torch.Tensor = dqn.act(state, steps_done=0, exploit=True)
            action: int = ACTION_SPACE[action_index.item()]
            next_state, reward, terminated, truncated, _ = env_eval.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state = torch.from_numpy(np.array(next_state)).unsqueeze(0).to(device, dtype=torch.float32)
            state = next_state

            if render:
                frame = env_eval.render()
                if frame is not None:
                    plt.imshow(frame)
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.001)
                    plt.clf()

        total_rewards.append(total_reward)

    env_eval.close()
    return np.mean(total_rewards)


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
