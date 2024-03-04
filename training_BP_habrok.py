import argparse
import torch
import numpy as np
import random
import os
from datetime import date
from Master_Thesis_Code.backpropamine_A2C import BP_RNetwork, Standard_RNetwork, Standard_FFNetwork
from Master_Thesis_Code.BP_A2C.BP_A2C_agent import A2C_Agent


parser = argparse.ArgumentParser(description='Train an A2C agent on the CartPole environment')
parser.add_argument('--num_neurons', type=int, default=32, help='Number of neurons in the hidden layer')
parser.add_argument('--network_type', type=str, default='BP_RNN', help='Type of network to use')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the agent')
parser.add_argument('--num_models', type=int, default=10, help='Number of models to train')

args = parser.parse_args()
learning_rate = args.learning_rate
num_neurons = args.num_neurons
network_type = args.network_type
num_models = args.num_models


device = "cpu"

env_name = 'CartPole-v0'
entropy_coef = 0.01 #THIS IS NOT ACTUALLY USED
value_pred_coef = 0.1
gammaR = 0.99
max_grad_norm = 4.0
max_steps = 200
batch_size = 1
num_training_episodes = 20000
evaluate_every = 10
selection_method = "range_evaluation_all_params"
num_evaluation_episodes = 10
evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')
training_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/training_seeds.npy')
max_reward = 200
training_method = "original"
range_min = 0.7
range_max = 1.3


# Create Results Directory
dirs = os.listdir('Master_Thesis_Code/BP_A2C/training_results/')
if not any('a2c_result' in d for d in dirs):
    result_id = 1
else:
    results = [d for d in dirs if 'a2c_result' in d]
    result_id = len(results) + 1

# Get today's date and add it to the results directory
d = date.today()
result_dir = f'Master_Thesis_Code/BP_A2C/training_results/{network_type}_a2c_result_' + str(result_id) + "_{}_entropycoef_{}_valuepredcoef_{}_batchsize_{}_maxsteps_{}_\
maxgradnorm_{}_gammaR_{}_learningrate_{}_numtrainepisodes_{}_selectionmethod_{}_trainingmethod_{}".format(
    str(d.year) + str(d.month) + str(d.day), entropy_coef, value_pred_coef, batch_size, max_steps, max_grad_norm, gammaR,
    learning_rate, num_training_episodes, selection_method, training_method)
if training_method == "range":
    result_dir += "_rangemin_{}_rangemax_{}".format(range_min, range_max)

os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))





best_average_after_all = []
best_average_all = []
for i_run in range(num_models):
    print("Run # {}".format(i_run))
    seed = int(training_seeds[i_run])
    
    torch.manual_seed(seed)
    random.seed(seed)

    if network_type == 'BP_RNN':
        agent_net = BP_RNetwork(4, num_neurons, 2, seed)
    elif network_type == 'Standard_RNN':
        agent_net = Standard_RNetwork(4, num_neurons, 2, seed)
    elif network_type == 'Standard_MLP':
        agent_net = Standard_FFNetwork(4, num_neurons, 2, seed)
    else:
        raise NotImplementedError("Network type not recognized")

    
    # optimizer = torch.optim.Adam(agent_net.parameters(), lr=1.0*learning_rate, eps=1e-4, weight_decay=l2_coef)
    optimizer = torch.optim.Adam(agent_net.parameters(), lr = learning_rate)
    agent = A2C_Agent(env_name, seed, agent_net, entropy_coef, value_pred_coef, gammaR,
                      max_grad_norm, max_steps, batch_size, num_training_episodes, optimizer, 
                      i_run, result_dir, selection_method, num_evaluation_episodes, evaluation_seeds, max_reward, evaluate_every, network_type)

    if training_method == "original":
        smoothed_scores, scores, best_average, best_average_after = agent.train_agent()
    elif training_method == "range":
        smoothed_scores, scores, best_average, best_average_after = agent.train_agent_on_range(range_min, range_max)

    best_average_after_all.append(best_average_after)
    best_average_all.append(best_average)



with open(f"{result_dir}/best_average_after.txt", 'w') as f:
    for i, best_episode in enumerate(best_average_after_all):
        f.write(f"{i}: {best_episode}\n")

    f.write(f"Average training episodes: {np.mean(best_average_after_all)}, std dev: {np.std(best_average_after_all)}")
    f.write(f"Mean average performance: {np.mean(best_average_all)}, std dev: {np.std(best_average_all)}")
