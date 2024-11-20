import argparse
import gymnasium as gym
import torch
import os
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import config
import matplotlib.pyplot as plt
from network import DQN
from utils import evaluate_policy
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_SPACE = [2, 3]

# Needed for several enviroments
CONFIG = config.Pong
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
    parser.add_argument('--path', type=str, help='Path to stored DQN model.')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of evaluation episodes.', nargs='?')
    parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
    parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
    parser.set_defaults(render=False)
    parser.set_defaults(save_video=False)

    args = parser.parse_args()

    if args.save_video or args.render:
        render_mode = 'rgb_array'
    else:
        render_mode = None

    # Initialize environment
    env = gym.make(args.env, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = FrameStack(env, num_stack=4)

    if args.save_video:
        env = gym.wrappers.RecordVideo(env, './video/', episode_trigger=lambda episode_ird: True)
    dqn = DQN(config=CONFIG).to(device)

    # Sanitize name for environment
    safe_name = args.env.replace('/', '_')

    # Load model weights
    weights = torch.load(args.path, map_location=device)
    dqn.load_state_dict(weights)
    dqn.eval() # Not really needed for current structure

    # Evaluate policy
    mean_reward = evaluate_policy(dqn, env, args, render=args.render)
    print(f'The policy got a mean return of {mean_reward} over {args.n_eval_episodes} episodes.')
    env.close()