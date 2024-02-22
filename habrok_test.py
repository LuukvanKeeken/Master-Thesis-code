from datetime import date
import os
import random
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import deque
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from ncps_time_constant_extraction.ncps.wirings import AutoNCP

device = "cpu"

env = gym.make('CartPole-v0')
evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')
training_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/training_seeds.npy')

num_neurons = 32
seed = 5
mode = "neuromodulated"
wiring = None
neuromod_network_dims = [3, 256, 128, num_neurons]


agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)

hidden_state = None

priv = torch.tensor([0.5, 0.1, 10.0], dtype=torch.float32).unsqueeze(0).to(device)

state = env.reset()
state = torch.from_numpy(state)
state = state.unsqueeze(0).to(device)

policy_output, value, hidden_state = agent_net((state.float(), priv), hidden_state)

print(policy_output)
print(value)
print(hidden_state)