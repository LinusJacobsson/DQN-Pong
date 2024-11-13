import random 
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def __len__(self):
        return len(self.memory)
    
    def push(self, observation, action, next_observation, reward, terminated):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (observation, action, next_observation, reward, terminated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))
    

