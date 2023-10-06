import os
import gym
import site
import torch
import random

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

site.addsitedir('cartpole_stuff/src/')

from datetime import date
from model import QNetwork, DSNN
from dqn_agent import Agent, ReplayBuffer
from matplotlib.gridspec import GridSpec

from backpropamine_DQN import FFNetwork

# Environment specific parameters
env_name = 'CartPole-v0'
n_runs = 5
n_evaluations = 100
max_steps = 200
num_episodes = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create Results Directory
dirs = os.listdir('.')
if not any('result' in d for d in dirs):
    result_id = 1
else:
    results = [d for d in dirs if 'result' in d]
    result_id = len(results) + 1

# Get today's date and add it to the results directory
d = date.today()
result_dir = 'dqn_result_' + str(result_id) + '_{}'.format(
    str(d.year) + str(d.month) + str(d.day))
os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))

# Hyperparameters
batch_size = 128
discount_factor = 0.999
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.999
update_every = 4
target_update_frequency = 100
learning_rate = 0.001 # lr is 0.0001 for simple maze as default
l2_coef = 0 # 0 is default in simple maze task
replay_memory_size = 4*10**4
tau = 1e-3


# SNN Hyperparameters
simulation_time = 3
alpha = 0.5
beta = 0.5
threshold = 0.2
weight_scale = 1
architecture = [4, 64, 64, 2]

seeds = np.load('cartpole_stuff/seeds/training_seeds.npy')


smoothed_scores_dqn_all = []
dqn_completion_after = []

for i_run in range(n_runs):
    print("Run # {}".format(i_run))
    seed = int(seeds[i_run])
    
    torch.manual_seed(seed)
    random.seed(seed)

    # policy_net = QNetwork(architecture, seed).to(device)
    # target_net = QNetwork(architecture, seed).to(device)
    policy_net = FFNetwork(4, 64, 64, 2, seed).to(device)
    target_net = FFNetwork(4, 64, 64, 2, seed).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    # optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate, weight_decay = l2_coef) 
    agent = Agent(env_name, policy_net, target_net, architecture, batch_size,
                  replay_memory_size, discount_factor, eps_start, eps_end, eps_decay,
                  update_every, target_update_frequency, optimizer, learning_rate,
                  num_episodes, max_steps, i_run, result_dir, seed, tau)
    
    smoothed_scores, scores, best_average, best_average_after = agent.train_agent()

    np.save(result_dir + '/scores_{}'.format(i_run), scores)
    np.save(result_dir + '/smoothed_scores_DQN_{}'.format(i_run), smoothed_scores)

    # save smoothed scores in list to plot later
    dqn_completion_after.append(best_average_after)
    smoothed_scores_dqn_all.append(smoothed_scores)
    print("")

