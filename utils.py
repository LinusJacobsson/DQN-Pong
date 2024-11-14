import torch
import logging
import os
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