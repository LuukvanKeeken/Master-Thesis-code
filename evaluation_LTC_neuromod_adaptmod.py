import argparse
from collections import OrderedDict
import os
import numpy as np
from Master_Thesis_Code.Adaptation_Module import StandardRNN
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from Master_Thesis_Code.Neuromodulated_Agent import NeuromodulatedAgent
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


def evaluate_LTC_agent_pole_length(policy_net, adaptation_module, env_name, num_episodes, evaluation_seeds, pole_length_modifier):
    
    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
        
    for i_episode in range(num_episodes):
        policy_hidden_state = None
        adaptation_module_hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        # Randomly sample an action and state to 
        # feed as the first input to the adaptation
        # module.
        prev_action = env.action_space.sample()
        prev_action = torch.tensor(prev_action).view(1, -1)
        prev_state = env.observation_space.sample()
        prev_state = torch.from_numpy(prev_state)
        prev_state = prev_state.unsqueeze(0).to(device)

        while not done:
            adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
            adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)
            
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            prev_state = state


            policy_output, _, policy_hidden_state = policy_net(state.float(), policy_hidden_state, adaptation_module_output)
            
            # Get distribution over the action space and select
            # the action with the highest probability.
            policy_dist = torch.softmax(policy_output, dim = 1)
            action = torch.argmax(policy_dist).item()
            prev_action = action
            prev_action = torch.tensor(prev_action).view(1, -1)
            

            # Take a step in the environment
            state, r, done, _ = env.step(action)
            total_reward += r

        eval_rewards.append(total_reward)

    return eval_rewards

def evaluate_LTC_agent_pole_mass(policy_net, adaptation_module, env_name, num_episodes, evaluation_seeds, pole_mass_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.masspole *= pole_mass_modifier
        
    for i_episode in range(num_episodes):
        policy_hidden_state = None
        adaptation_module_hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        # Randomly sample an action and state to 
        # feed as the first input to the adaptation
        # module.
        prev_action = env.action_space.sample()
        prev_action = torch.tensor(prev_action).view(1, -1)
        prev_state = env.observation_space.sample()
        prev_state = torch.from_numpy(prev_state)
        prev_state = prev_state.unsqueeze(0).to(device)

        while not done:
            adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
            adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)
            
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            prev_state = state


            policy_output, _, policy_hidden_state = policy_net(state.float(), policy_hidden_state, adaptation_module_output)
            
            # Get distribution over the action space and select
            # the action with the highest probability.
            policy_dist = torch.softmax(policy_output, dim = 1)
            action = torch.argmax(policy_dist).item()
            prev_action = action
            prev_action = torch.tensor(prev_action).view(1, -1)
            

            # Take a step in the environment
            state, r, done, _ = env.step(action)
            total_reward += r

        eval_rewards.append(total_reward)

    return eval_rewards

def evaluate_LTC_agent_force_mag(policy_net, adaptation_module, env_name, num_episodes, evaluation_seeds, force_mag_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.force_mag *= force_mag_modifier
        
    for i_episode in range(num_episodes):
        policy_hidden_state = None
        adaptation_module_hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        # Randomly sample an action and state to 
        # feed as the first input to the adaptation
        # module.
        prev_action = env.action_space.sample()
        prev_action = torch.tensor(prev_action).view(1, -1)
        prev_state = env.observation_space.sample()
        prev_state = torch.from_numpy(prev_state)
        prev_state = prev_state.unsqueeze(0).to(device)

        while not done:
            adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
            adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)
            
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            prev_state = state


            policy_output, _, policy_hidden_state = policy_net(state.float(), policy_hidden_state, adaptation_module_output)
            
            # Get distribution over the action space and select
            # the action with the highest probability.
            policy_dist = torch.softmax(policy_output, dim = 1)
            action = torch.argmax(policy_dist).item()
            prev_action = action
            prev_action = torch.tensor(prev_action).view(1, -1)
            

            # Take a step in the environment
            state, r, done, _ = env.step(action)
            total_reward += r

        eval_rewards.append(total_reward)

    return eval_rewards


parser = argparse.ArgumentParser(description='Evaluate adaptation module for neuromodulated CfC')
parser.add_argument('--adapt_mod_type', type=str, default='StandardRNN', help='Type of adaptation module')
parser.add_argument('--state_dims', type=int, default=4, help='Number of state dimensions')
parser.add_argument('--action_dims', type=int, default=1, help='Number of action dimensions')
parser.add_argument('--num_neurons_adaptmod', type=int, default=32, help='Number of neurons in the adaptation module')
parser.add_argument('--num_neurons_policy', type=int, default=32, help='Number of neurons in the policy network')
parser.add_argument('--num_actions', type=int, default=2, help='Number of actions')
parser.add_argument('--mode', type=str, default='only_neuromodulated', help='Mode of the CfC network')

args = parser.parse_args()

adapt_mod_type = args.adapt_mod_type
state_dims = args.state_dims
action_dims = args.action_dims
num_neurons_adaptmod = args.num_neurons_adaptmod
num_neurons_policy = args.num_neurons_policy
num_actions = args.num_actions
mode = args.mode

env_name = "CartPole-v0"
max_reward = 200
max_steps = 200
n_evaluations = 100
neuron_type = "CfC"
sparsity_level = 0.5
seed = 5



wiring = None

evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')

policy_dir = "CfC_1136_2024326_lr_0.0001_nn_32_encoutact_relu_mode_neuromodulated_neuromod_network_dims_3_256_128"
adapt_mod_dir = "adaptation_module_StandardRNN_result_725_2024327_CfC_result_296_202437_numneuronsadaptmod_32_lradaptmod_0.0005_wdadaptmod_0.01"

os.mkdir(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}")


policy_weights_0 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))
policy_weights_1 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
policy_weights_2 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
policy_weights_3 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
policy_weights_4 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
policy_weights_5 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_5.pt', map_location=torch.device(device))
policy_weights_6 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_6.pt', map_location=torch.device(device))
policy_weights_7 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_7.pt', map_location=torch.device(device))
policy_weights_8 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_8.pt', map_location=torch.device(device))
policy_weights_9 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{policy_dir}/checkpoint_{neuron_type}_A2C_9.pt', map_location=torch.device(device))
policy_weights = [policy_weights_0, policy_weights_1, policy_weights_2, policy_weights_3, policy_weights_4, policy_weights_5, policy_weights_6, policy_weights_7, policy_weights_8, policy_weights_9]

am_weights_0 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_0.pt', map_location=torch.device(device))
am_weights_1 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_1.pt', map_location=torch.device(device))
am_weights_2 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_2.pt', map_location=torch.device(device))
am_weights_3 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_3.pt', map_location=torch.device(device))
am_weights_4 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_4.pt', map_location=torch.device(device))
am_weights_5 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_5.pt', map_location=torch.device(device))
am_weights_6 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_6.pt', map_location=torch.device(device))
am_weights_7 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_7.pt', map_location=torch.device(device))
am_weights_8 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_8.pt', map_location=torch.device(device))
am_weights_9 = torch.load(f'Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{adapt_mod_dir}/best_adaptation_module_loss_CfC_A2C_9.pt', map_location=torch.device(device))
am_weights = [am_weights_0, am_weights_1, am_weights_2, am_weights_3, am_weights_4, am_weights_5, am_weights_6, am_weights_7, am_weights_8, am_weights_9]

with torch.no_grad():
    # ORIGINAL ENVIRONMENT EVALUATION ---------------------------
    with open(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/original_env_evals.txt", "w") as f:
        eraser = '\b \b'
        original_eval_rewards = []
        for i, (pw, amw) in enumerate(zip(policy_weights, am_weights)):
            print('Run {:02d} ...'.format(i), end='')
            if neuron_type == "LTC":
                raise NotImplementedError
                agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
            elif neuron_type == "CfC":
                if adapt_mod_type == 'StandardRNN':
                    adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
                else:
                    raise NotImplementedError

                policy_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring).to(device)
                w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
                w_policy['cfc_model.rnn_cell.tau_system'] = torch.reshape(w_policy['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
                policy_net.load_state_dict(w_policy)

                adaptation_module.load_state_dict(amw)


            rewards = evaluate_LTC_agent_pole_length(policy_net, adaptation_module, env_name, n_evaluations, evaluation_seeds, 1.0)
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

        for i, (pw, amw) in enumerate(zip(policy_weights, am_weights)):
            print('Run {:02d} ...'.format(i), end='')
            if neuron_type == "LTC":
                raise NotImplementedError
                agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
            elif neuron_type == "CfC":
                if adapt_mod_type == 'StandardRNN':
                    adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
                else:
                    raise NotImplementedError

                policy_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring).to(device)
                w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
                w_policy['cfc_model.rnn_cell.tau_system'] = torch.reshape(w_policy['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
                policy_net.load_state_dict(w_policy)

                adaptation_module.load_state_dict(amw)

            

            rewards = evaluate_LTC_agent_pole_length(policy_net, adaptation_module, env_name, n_evaluations, evaluation_seeds, percentage)
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


    os.mkdir(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_length")
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_length/means.npy", mean_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_length/stddevs.npy", std_dev_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_length/medians.npy", median_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_length/percentages.npy", percentages)



    # POLE MASS EVALUATION ---------------------------
    percentages = np.linspace(5.0, 20.0, 16)
    all_modified_env_eval_rewards = []
    for percentage in percentages:
        print(percentage)
        modified_env_eval_rewards = []

        for i, (pw, amw) in enumerate(zip(policy_weights, am_weights)):
            print('Run {:02d} ...'.format(i), end='')
            if neuron_type == "LTC":
                raise NotImplementedError
                agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
            elif neuron_type == "CfC":
                if adapt_mod_type == 'StandardRNN':
                    adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
                else:
                    raise NotImplementedError

                policy_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring).to(device)
                w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
                w_policy['cfc_model.rnn_cell.tau_system'] = torch.reshape(w_policy['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
                policy_net.load_state_dict(w_policy)

                adaptation_module.load_state_dict(amw)

            

            rewards = evaluate_LTC_agent_pole_mass(policy_net, adaptation_module, env_name, n_evaluations, evaluation_seeds, percentage)
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


    os.mkdir(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_mass")
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_mass/means.npy", mean_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_mass/stddevs.npy", std_dev_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_mass/medians.npy", median_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/pole_mass/percentages.npy", percentages)




    # FORCE MAG EVALUATION ---------------------------
    percentages = np.concatenate((np.linspace(0.2, 2.0, 19), np.linspace(2.5, 6.0, 8)))
    all_modified_env_eval_rewards = []
    for percentage in percentages:
        print(percentage)
        modified_env_eval_rewards = []

        for i, (pw, amw) in enumerate(zip(policy_weights, am_weights)):
            print('Run {:02d} ...'.format(i), end='')
            if neuron_type == "LTC":
                raise NotImplementedError
                agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
            elif neuron_type == "CfC":
                if adapt_mod_type == 'StandardRNN':
                    adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
                else:
                    raise NotImplementedError

                policy_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring).to(device)
                w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
                w_policy['cfc_model.rnn_cell.tau_system'] = torch.reshape(w_policy['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
                policy_net.load_state_dict(w_policy)

                adaptation_module.load_state_dict(amw)

            

            rewards = evaluate_LTC_agent_force_mag(policy_net, adaptation_module, env_name, n_evaluations, evaluation_seeds, percentage)
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


    os.mkdir(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/force_mag")
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/force_mag/means.npy", mean_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/force_mag/stddevs.npy", std_dev_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/force_mag/medians.npy", median_avgs)
    np.save(f"Master_Thesis_Code/LTC_A2C/adaptation_module/evaluation_results/{adapt_mod_dir}/force_mag/percentages.npy", percentages)




