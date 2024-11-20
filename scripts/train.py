import argparse
import gymnasium as gym
import torch
import os
import logging
import numpy as np
import config.config
from typing import Dict, List
from models.network import DQN
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from utils.utils import evaluate_policy, preprocess, setup_logger, ReplayMemory, optimize

# Set device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser setup
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5',
                    help="Specify the environment to use (default: 'ALE/Pong-v5').")
parser.add_argument('--evaluate_freq', type=int, default=50, nargs='?',
                    help="Frequency of evaluations during training (default: 50 episodes).")
parser.add_argument('--evaluation_episodes', type=int, default=10, nargs='?',
                    help="Number of episodes for evaluation (default: 10).")
parser.add_argument('--model_path', default='models/',
                    help="Path to save the trained model weights (default: 'models/').")
parser.add_argument('--log_dir', default='logs/',
                    help="Directory to store the training logs (default: 'logs/').")

# Environment-specific configurations
ENV_CONFIGS: Dict[str, Dict] = {
    'ALE/Pong-v5': config.Pong
}


def main() -> None:
    """Main training loop for DQN on Atari Pong."""
    args: argparse.Namespace = parser.parse_args()

    # Set up logger
    logger: logging.Logger = setup_logger(args.log_dir)

    # Action space mapping (e.g., UP, DOWN)
    ACTION_SPACE: List[int] = [2, 3]

    # Initialize environment
    env: gym.Env = gym.make(args.env, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30, scale_obs=False)
    env = FrameStack(env, num_stack=4)  # Stack 4 frames for input consistency

    # Retrieve configuration for the selected environment
    env_config: Dict = ENV_CONFIGS[args.env]

    # Initialize networks
    policy_dqn: DQN = DQN(config=env_config).to(device)
    target_dqn: DQN = DQN(config=env_config).to(device)
    target_dqn.load_state_dict(policy_dqn.state_dict())

    # Initialize memory and optimizer
    memory: ReplayMemory = ReplayMemory(env_config['memory_size'])
    optimizer: torch.optim.Adam = torch.optim.Adam(policy_dqn.parameters(), lr=env_config['lr'])

    # Training variables
    best_mean_reward: float = -float('Inf')
    steps_done: int = 0

    # Training loop
    for episode in range(env_config['n_episodes']):
        total_reward: float = 0.0
        state, _ = env.reset()
        state: torch.Tensor = preprocess(state, env=args.env)
        done: bool = False

        while not done:
            # Select action using the policy network
            action_index: int = policy_dqn.act(state, steps_done)
            action: int = ACTION_SPACE[action_index]

            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Prepare data for memory
            reward_tensor: torch.Tensor = torch.tensor([[reward]], device=device)
            total_reward += reward_tensor.item()
            next_state: torch.Tensor = preprocess(next_state, env=args.env)
            done_tensor: torch.Tensor = torch.tensor([[done]], dtype=torch.bool, device=device)

            # Store transition in replay memory
            memory.push(state, action_index, next_state, reward_tensor, done_tensor)
            state = next_state

            # Optimize the model at specified intervals
            if steps_done % env_config['train_frequency'] == 0:
                optimize(policy_dqn, target_dqn, memory, optimizer, device=device)

            # Update target network at specified intervals
            if steps_done % env_config['target_update_frequency'] == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            steps_done += 1

        logger.info(f"Episode {episode + 1}/{env_config['n_episodes']} completed with total reward: {total_reward:.2f}")

        # Evaluate the agent at specified intervals
        if episode % args.evaluate_freq == 0:
            avg_reward: float = evaluate_policy(policy_dqn, env, args.evaluation_episodes)
            logger.info(f"Evaluation after episode {episode}: Average reward: {avg_reward:.2f}")

            if avg_reward >= best_mean_reward:
                best_mean_reward = avg_reward

                # Save model weights
                save_path: str = os.path.join(args.model_path, f'Episode_{episode}_weights')
                torch.save(policy_dqn.state_dict(), save_path)
                logger.info(f"New best model saved at: {save_path}")


if __name__ == "__main__":
    main()
