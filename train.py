import argparse
import gymnasium as gym
import torch
import os
import logging
import numpy as np
import config
from network import DQN, ReplayMemory, optimize
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from utils import evaluate_policy, setup_logger, preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=50, nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=10, nargs='?')
parser.add_argument('--model_path', default='models/')
# Needed for several enviroments
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()
    ACTION_SPACE = [2, 3] # Maps to Uo, Down
    # Initialize enviroment and config, to change frame skipping, modify frame_skip, not frameskip!
    env = gym.make(args.env, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30, scale_obs=True)
    env = FrameStack(env, num_stack=4) # Makes sure that we always stack 4 frames 
    env_config = ENV_CONFIGS[args.env]
    # Initialize network
    policy_dqn = DQN(config=env_config).to(device)
    target_dqn = DQN(config=env_config).to(device)
    target_dqn.load_state_dict(policy_dqn.state_dict())
    # Initialize memory
    memory = ReplayMemory(env_config['memory_size'])
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=env_config['lr'])

    best_mean_reward = -float('Inf')
    steps_done = 0

    for episode in range(env_config['n_episodes']):
        total_reward = 0
        state, _ = env.reset()
        state = preprocess(state, env=args.env)
        done = False
        while not done:
            action_index = policy_dqn.act(state, steps_done)
            action = ACTION_SPACE[action_index]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = torch.tensor([[reward]], device=device)
            total_reward += reward.item()
            next_state = preprocess(next_state, env=args.env)
            done_tensor = torch.tensor([[done]], dtype=torch.bool, device=device)
            memory.push(state, action_index, next_state, reward, done_tensor)
            state = next_state
            
            # Update every train_frequency steps
            if steps_done % env_config['train_frequency'] == 0:
                optimize(policy_dqn, target_dqn, memory, optimizer, device=device)

            # Update target network every target_update steps
            if steps_done % env_config['target_update_frequency'] == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())
            
            steps_done += 1
        print(f'Episode {episode}/10 done')
        # Evaluate the tagent every evaluate_freq episodes
        if episode % args.evaluate_freq == 0:
            avg_reward = evaluate_policy(policy_dqn, env, args)


            if avg_reward >= best_mean_reward:
                best_mean_reward = avg_reward
                # Save model weights
                torch.save(policy_dqn.state_dict(), os.path.join(args.model_path, f'Episode_{episode}_weights'))
                