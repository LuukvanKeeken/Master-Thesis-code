import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from Master_Thesis_Code.backpropamine_A2C import BP_RNetwork, Standard_RNetwork, Standard_FFNetwork
from Master_Thesis_Code.BP_A2C.BP_A2C_agent import evaluate_BP_agent_pole_length, evaluate_BP_agent_force_mag, evaluate_BP_agent_pole_mass


device = "cpu"

env_name = 'CartPole-v0'
max_reward = 200
max_steps = 200
seed = 5

n_evaluations = 100

evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')


network_type = 'Standard_MLP'
num_neurons = 32
results_dir = f"Standard_MLP_a2c_result_64_202434_entropycoef_0.01_valuepredcoef_0.1_batchsize_1_maxsteps_200_maxgradnorm_4.0_gammaR_0.99_learningrate_5e-05_numtrainepisodes_20000_selectionmethod_range_evaluation_all_params_trainingmethod_original"
os.mkdir(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}")


weights_0 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_0.pt", map_location=torch.device(device))
weights_1 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_1.pt", map_location=torch.device(device))
weights_2 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_2.pt", map_location=torch.device(device))
weights_3 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_3.pt", map_location=torch.device(device))
weights_4 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_4.pt", map_location=torch.device(device))
weights_5 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_5.pt", map_location=torch.device(device))
weights_6 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_6.pt", map_location=torch.device(device))
weights_7 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_7.pt", map_location=torch.device(device))
weights_8 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_8.pt", map_location=torch.device(device))
weights_9 = torch.load(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/checkpoint_{network_type}_A2C_9.pt", map_location=torch.device(device))
weights = [weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9]


eraser = '\b \b'

original_eval_rewards = []

for i, w in enumerate(weights):
    print('Run {:02d} ...'.format(i), end='')

    if network_type == 'BP_RNN':
        agent_net = BP_RNetwork(4, num_neurons, 2, seed).to(device)
    elif network_type == 'Standard_RNN':
        agent_net = Standard_RNetwork(4, num_neurons, 2, seed).to(device)
    elif network_type == 'Standard_MLP':
        agent_net = Standard_FFNetwork(4, num_neurons, num_neurons, 2, seed).to(device)
    agent_net.loadWeights(w)

    rewards = evaluate_BP_agent_pole_length(agent_net, env_name, n_evaluations, evaluation_seeds, 1.0)
    original_eval_rewards.append(rewards)
    
    print(eraser*3 + '-> Avg reward: {:7.2f}'.format(np.mean(rewards)))

print(f"Mean avg reward: {np.mean(original_eval_rewards)}")




# POLE LENGTH EVALUATION ---------------------------
percentages = np.linspace(0.1, 2.0, 20)
percentages = np.concatenate((percentages, np.linspace(2.5, 20, 36)))
all_modified_env_eval_rewards = []
for percentage in percentages:
    print(percentage)
    modified_env_eval_rewards = []

    for i, w in enumerate(weights):
        print('Run {:02d} ...'.format(i), end='')
        if network_type == 'BP_RNN':
            agent_net = BP_RNetwork(4, num_neurons, 2, seed).to(device)
        elif network_type == 'Standard_RNN':
            agent_net = Standard_RNetwork(4, num_neurons, 2, seed).to(device)
        elif network_type == 'Standard_MLP':
            agent_net = Standard_FFNetwork(4, num_neurons, num_neurons, 2, seed).to(device)
        agent_net.loadWeights(w)

        rewards = evaluate_BP_agent_pole_length(agent_net, env_name, n_evaluations, evaluation_seeds, percentage)
        modified_env_eval_rewards.append(rewards)
        
        print(eraser*3 + '-> Avg reward: {:7.2f}'.format(np.mean(rewards)))

    all_modified_env_eval_rewards.append(modified_env_eval_rewards)


mean_avgs = []
std_dev_avgs = []
median_avgs = []
for results in all_modified_env_eval_rewards:
    means_per_model = np.mean(results, axis = 1)
    mean_avgs.append(np.mean(means_per_model))
    std_dev_avgs.append(np.std(means_per_model))
    median_avgs.append(np.median(means_per_model))


os.mkdir(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_length")
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_length/means.npy", mean_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_length/stddevs.npy", std_dev_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_length/medians.npy", median_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_length/percentages.npy", percentages)




# POLE MASS EVALUATION ---------------------------
percentages = np.linspace(5.0, 20.0, 16)
all_modified_env_eval_rewards = []
for percentage in percentages:
    print(percentage)
    modified_env_eval_rewards = []

    for i, w in enumerate(weights):
        print('Run {:02d} ...'.format(i), end='')
        if network_type == 'BP_RNN':
            agent_net = BP_RNetwork(4, num_neurons, 2, seed).to(device)
        elif network_type == 'Standard_RNN':
            agent_net = Standard_RNetwork(4, num_neurons, 2, seed).to(device)
        elif network_type == 'Standard_MLP':
            agent_net = Standard_FFNetwork(4, num_neurons, num_neurons, 2, seed).to(device)
        agent_net.loadWeights(w)

        rewards = evaluate_BP_agent_pole_mass(agent_net, env_name, n_evaluations, evaluation_seeds, percentage)
        modified_env_eval_rewards.append(rewards)
        
        print(eraser*3 + '-> Avg reward: {:7.2f}'.format(np.mean(rewards)))

    all_modified_env_eval_rewards.append(modified_env_eval_rewards)


mean_avgs = []
std_dev_avgs = []
median_avgs = []
for results in all_modified_env_eval_rewards:
    means_per_model = np.mean(results, axis = 1)
    mean_avgs.append(np.mean(means_per_model))
    std_dev_avgs.append(np.std(means_per_model))
    median_avgs.append(np.median(means_per_model))


os.mkdir(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_mass")
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_mass/means.npy", mean_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_mass/stddevs.npy", std_dev_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_mass/medians.npy", median_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/pole_mass/percentages.npy", percentages)




# FORCE MAG EVALUATION ---------------------------
percentages = np.concatenate((np.linspace(0.2, 2.0, 19), np.linspace(2.5, 6.0, 8)))
all_modified_env_eval_rewards = []
for percentage in percentages:
    print(percentage)
    modified_env_eval_rewards = []

    for i, w in enumerate(weights):
        print('Run {:02d} ...'.format(i), end='')
        if network_type == 'BP_RNN':
            agent_net = BP_RNetwork(4, num_neurons, 2, seed).to(device)
        elif network_type == 'Standard_RNN':
            agent_net = Standard_RNetwork(4, num_neurons, 2, seed).to(device)
        elif network_type == 'Standard_MLP':
            agent_net = Standard_FFNetwork(4, num_neurons, num_neurons, 2, seed).to(device)
        agent_net.loadWeights(w)

        rewards = evaluate_BP_agent_force_mag(agent_net, env_name, n_evaluations, evaluation_seeds, percentage)
        modified_env_eval_rewards.append(rewards)
        
        print(eraser*3 + '-> Avg reward: {:7.2f}'.format(np.mean(rewards)))

    all_modified_env_eval_rewards.append(modified_env_eval_rewards)


mean_avgs = []
std_dev_avgs = []
median_avgs = []
for results in all_modified_env_eval_rewards:
    means_per_model = np.mean(results, axis = 1)
    mean_avgs.append(np.mean(means_per_model))
    std_dev_avgs.append(np.std(means_per_model))
    median_avgs.append(np.median(means_per_model))


os.mkdir(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/force_mag")
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/force_mag/means.npy", mean_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/force_mag/stddevs.npy", std_dev_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/force_mag/medians.npy", median_avgs)
np.save(f"Master_Thesis_Code/BP_A2C/evaluation_results/{results_dir}/force_mag/percentages.npy", percentages)