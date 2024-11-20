"""
evaluate.py

This script evaluates a trained DQN (Deep Q-Network) model on an Atari environment
(e.g., Pong). It supports rendering, saving videos, and computing the average reward
over a specified number of episodes.

Usage:
    python evaluate.py --env ALE/Pong-v5 --path <path_to_model_weights> --render --save_video
"""

import argparse
import gymnasium as gym
import torch
import os
from typing import Optional
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import matplotlib.pyplot as plt
from network import DQN
from utils import evaluate_policy
import config

# Set up device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default action space for Pong (Up, Down)
ACTION_SPACE: list[int] = [2, 3]

def main() -> None:
    # Parse command-line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evaluate a trained DQN model on an Atari environment."
    )
    parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5',
                        help="The Atari environment to evaluate (default: 'ALE/Pong-v5').")
    parser.add_argument('--path', type=str, required=True,
                        help="Path to the stored DQN model weights.")
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                        help="Number of evaluation episodes (default: 10).")
    parser.add_argument('--render', action='store_true',
                        help="Render the environment during evaluation.")
    parser.add_argument('--save_video', action='store_true',
                        help="Save the evaluation episodes as video.")
    args: argparse.Namespace = parser.parse_args()

    # Set render mode
    render_mode: Optional[str] = 'rgb_array' if args.save_video or args.render else None

    # Initialize the environment
    video_folder: str = os.path.join('.', 'video')
    env: gym.Env = gym.make(args.env, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = FrameStack(env, num_stack=4)

    if args.save_video:
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        else:
            print(f"Warning: Video folder {video_folder} already exists. Existing videos may be overwritten.")
        env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda episode_idx: True)

    # Load the DQN model
    try:
        dqn: DQN = DQN(config=config.Pong).to(device)
    except AttributeError:
        raise ImportError("The configuration for Pong is missing. Ensure 'config.py' is correctly set up.")

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Model weights file not found at {args.path}. Please provide a valid path.")

    # Load model weights
    weights: dict = torch.load(args.path, map_location=device)
    dqn.load_state_dict(weights)
    dqn.eval()  # Set to evaluation mode

    # Evaluate policy
    print(f"Evaluating policy on {args.env} for {args.n_eval_episodes} episodes...")
    mean_reward: float = evaluate_policy(dqn, env, args.n_eval_episodes, render=args.render)
    print(f"The policy achieved a mean return of {mean_reward:.2f} over {args.n_eval_episodes} episodes.")

    # Clean up
    env.close()

if __name__ == '__main__':
    main()
