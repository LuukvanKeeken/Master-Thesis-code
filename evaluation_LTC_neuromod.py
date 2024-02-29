import os
import numpy as np
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from ncps_time_constant_extraction.ncps.wirings import AutoNCP
import torch
import gym


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_privileged_info(env):
    pole_length = env.unwrapped.length
    # gravity = env.unwrapped.gravity
    # masscart = env.unwrapped.masscart
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    privileged_info = [pole_length, masspole, force_mag]
    return torch.tensor(privileged_info, dtype=torch.float32)


def evaluate_LTC_agent_pole_length(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
        
    for i_episode in range(num_episodes):
        hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            policy_output, value, hidden_state = agent_net((state.float(), privileged_info), hidden_state)
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards

def evaluate_LTC_agent_pole_mass(agent_net, env_name, num_episodes, evaluation_seeds, pole_mass_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.masspole *= pole_mass_modifier
        
    for i_episode in range(num_episodes):
        hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            policy_output, value, hidden_state = agent_net((state.float(), privileged_info), hidden_state)
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards

def evaluate_LTC_agent_force_mag(agent_net, env_name, num_episodes, evaluation_seeds, force_mag_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.force_mag *= force_mag_modifier
        
    for i_episode in range(num_episodes):
        hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            policy_output, value, hidden_state = agent_net((state.float(), privileged_info), hidden_state)
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards



env_name = "CartPole-v0"
max_reward = 200
max_steps = 200
n_evaluations = 100
neuron_type = "CfC"
num_neurons = 32
sparsity_level = 0.5
seed = 5
mode = "neuromodulated"
neuromod_network_dims = [3, 256, 128, num_neurons]
# wiring = AutoNCP(num_neurons, 3, sparsity_level=sparsity_level, seed=seed)
wiring = None

evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')

results_dir = f"CfC_a2c_result_174_2024227_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated"
os.mkdir(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}")


weights_0 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))
weights_1 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
weights_2 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
weights_3 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
weights_4 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
weights_5 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_5.pt', map_location=torch.device(device))
weights_6 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_6.pt', map_location=torch.device(device))
weights_7 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_7.pt', map_location=torch.device(device))
weights_8 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_8.pt', map_location=torch.device(device))
weights_9 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_9.pt', map_location=torch.device(device))
weights = [weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9]
# weights = [weights_0, weights_1, weights_2, weights_3, weights_4]


# ORIGINAL ENVIRONMENT EVALUATION ---------------------------
with open(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/original_env_evals.txt", "w") as f:
    eraser = '\b \b'
    original_eval_rewards = []
    for i, w in enumerate(weights):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            raise NotImplementedError
            agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        elif neuron_type == "CfC":
            agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
            w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        agent_net.load_state_dict(w)

        rewards = evaluate_LTC_agent_pole_length(agent_net, env_name, n_evaluations, evaluation_seeds, 1.0)
        original_eval_rewards.append(rewards)
        print(eraser*3 + '-> Avg reward: {:7.2f}'.format(np.mean(rewards)))
        f.write(f"Run {i}: avg reward: {np.mean(rewards)}\n")

    print(f"Mean avg reward: {np.mean(np.mean(original_eval_rewards, axis = 1))}, +/- {np.std(np.mean(original_eval_rewards, axis = 1))}")
    f.write(f"Mean avg reward: {np.mean(np.mean(original_eval_rewards, axis = 1))}, +/- {np.std(np.mean(original_eval_rewards, axis = 1))}\n")



# POLE LENGTH EVALUATION ---------------------------
percentages = np.linspace(0.1, 2.0, 20)
percentages = np.concatenate((percentages, np.linspace(2.5, 20, 36)))
# percentages = [0.1, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0]
all_modified_env_eval_rewards = []
for percentage in percentages:
    print(percentage)
    modified_env_eval_rewards = []

    for i, w in enumerate(weights):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            raise NotImplementedError
            agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        elif neuron_type == "CfC":
            agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
            w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        agent_net.load_state_dict(w)

        rewards = evaluate_LTC_agent_pole_length(agent_net, env_name, n_evaluations, evaluation_seeds, percentage)
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


os.mkdir(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_length")
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_length/means.npy", mean_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_length/stddevs.npy", std_dev_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_length/medians.npy", median_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_length/percentages.npy", percentages)



# POLE MASS EVALUATION ---------------------------
percentages = np.linspace(5.0, 20.0, 16)
all_modified_env_eval_rewards = []
for percentage in percentages:
    print(percentage)
    modified_env_eval_rewards = []

    for i, w in enumerate(weights):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            raise NotImplementedError
            agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        elif neuron_type == "CfC":
            agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
            w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        agent_net.load_state_dict(w)

        rewards = evaluate_LTC_agent_pole_mass(agent_net, env_name, n_evaluations, evaluation_seeds, percentage)
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


os.mkdir(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_mass")
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_mass/means.npy", mean_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_mass/stddevs.npy", std_dev_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_mass/medians.npy", median_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/pole_mass/percentages.npy", percentages)




# FORCE MAG EVALUATION ---------------------------
percentages = np.concatenate((np.linspace(0.2, 2.0, 19), np.linspace(2.5, 6.0, 8)))
all_modified_env_eval_rewards = []
for percentage in percentages:
    print(percentage)
    modified_env_eval_rewards = []

    for i, w in enumerate(weights):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            raise NotImplementedError
            agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        elif neuron_type == "CfC":
            agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
            w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        agent_net.load_state_dict(w)

        rewards = evaluate_LTC_agent_force_mag(agent_net, env_name, n_evaluations, evaluation_seeds, percentage)
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


os.mkdir(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/force_mag")
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/force_mag/means.npy", mean_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/force_mag/stddevs.npy", std_dev_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/force_mag/medians.npy", median_avgs)
np.save(f"Master_Thesis_Code/LTC_A2C/evaluation_results/{results_dir}/force_mag/percentages.npy", percentages)




