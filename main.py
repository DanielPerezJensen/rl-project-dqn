import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from replaymemory import ReplayMemory
from models import *
from train import train

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Select which env to use
def select_env(env_name):
    if env_name == 'cartpole':
        return gym.make('CartPole-v0').unwrapped
    if env_name == 'pendulum':
        return gym.make('Pendulum-v0')

def select_network(input_size, hidden_layer, output_size):
    if env_name == 'cartpole':
        return DQN_cartpole(input_size, hidden_layer, output_size)
    if env_name == 'pendulum':
        return DQN_pendulum(p_actions)

if __name__ == "__main__":
    env_name = 'pendulum'
    max_size = 10000
    learning_rate = 0.001
    episodes = 50
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    batch_size = 64
    hidden_layer = 256
    # We use a discrete action space for pendulum
    p_actions = 5

    env = select_env(env_name)
    output_size = p_actions if env_name is 'pendulum' else env.action_space.n
    input_size = env.observation_space.shape[0]
    model = select_network(input_size, hidden_layer, output_size)
    memory = ReplayMemory(max_size)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    train(env, env_name, model, memory, optimizer, episodes,
          eps_start, eps_end, eps_decay, batch_size, learning_rate, p_actions)

