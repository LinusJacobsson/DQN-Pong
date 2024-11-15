import torch
import logging
import os
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(state, env):
    if env in ['ALE/Pong-v5']:
        return torch.tensor(state, device=device, dtype=torch.float32)
    else:
        raise NotImplementedError("You are using an unsupported enviroment!")
    

def setup_logger(log_dir):
    logger = logging.getLogger('DQN')
    logger.setLevel(logging.INFO)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, 'training.log')

    # Handler for log file
    fh = logging.FileHandler(log_file_path)
    fh.logging.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.logging.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logging setup complete...")
    return logger

def evaluate_policy(dqn, env_eval, args):

    total_rewards = []
    for _ in range(args.n_eval_episodes):
        state, _ = env_eval.reset()
        state = torch.from_numpy(np.array(state)).unsqueeze(0).to(device)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action_index = dqn(state).max(1)[1].view(1, 1)
            action = ACTION_SPACE[action_index.item()] # ACTION_SPACE is defined in evaluate.py
            next_state, reward, terminated, truncated, _ = env_eval.step(action)
            done = terminated or truncated
            total_reward+= reward
            next_state = torch.from_numpy(np.array(next_state)).unsqueeze(0).to(device)
            state = next_state
        total_rewards.append(total_reward)
    env_eval.close()
    return np.mean(total_rewards)
