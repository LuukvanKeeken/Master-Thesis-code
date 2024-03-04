from datetime import date
import gym
import numpy as np
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from ncps_time_constant_extraction.ncps.wirings import AutoNCP

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


def evaluate_agent_all_params(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier, pole_mass_modifier, force_mag_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
    env.unwrapped.masspole *= pole_mass_modifier
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


def get_privileged_info(env):
    pole_length = env.unwrapped.length
    # gravity = env.unwrapped.gravity
    # masscart = env.unwrapped.masscart
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    privileged_info = [pole_length, masspole, force_mag]
    return torch.tensor(privileged_info, dtype=torch.float32)



randomization_factors = [0.2, 0.35, 0.5]
learning_rates = [0.0005, 0.0001, 0.00005]

result_id = 175
flag = False

all_results = []
all_pole_length_adapt_results = []
all_pole_mass_adapt_results = []
all_force_mag_adapt_results = []
all_mixed_adaptation_eval_rewards = []
d = 2024227


for factor in randomization_factors:
    for learning_rate in learning_rates:

        print(f"learning rate: {learning_rate}, rand factor: {factor}")
        device = "cpu"
        # learning_rate = 0.001
        selection_method = "range_evaluation_all_params"
        gamma = 0.99
        training_method = "standard"
        num_neurons = 32
        neuron_type = "CfC"
        mode = "neuromodulated"
        neuromod_network_dims = [3, 256, 128, num_neurons]
        randomization_params = 3*[factor]
        tau_sys_extraction = True
        num_models = 5
        sparsity_level = 0.5
        seed = 5
        # wiring = AutoNCP(num_neurons, 3, sparsity_level=sparsity_level, seed=seed)
        wiring = None
        env_name = "CartPole-v0"
        n_evaluations = 100
        evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')

        
        result_dir = f'{neuron_type}_a2c_result_' + str(result_id) + f'_{str(d)}_learningrate_{learning_rate}_selectiomethod_{selection_method}_gamma_{gamma}_trainingmethod_{training_method}_numneurons_{num_neurons}_tausysextraction_{tau_sys_extraction}'
        if neuron_type == "CfC":
            result_dir += "_mode_" + mode
        if wiring:
            result_dir += "_wiring_" + "AutoNCP"
        if randomization_params:
            result_dir += "_randomization_params_" + str(randomization_params)

        

        weights_0 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{result_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))
        weights_1 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{result_dir}/checkpoint_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
        weights_2 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{result_dir}/checkpoint_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
        weights_3 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{result_dir}/checkpoint_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
        weights_4 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{result_dir}/checkpoint_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
        weights = [weights_0, weights_1, weights_2, weights_3, weights_4]


        eraser = '\b \b'
        # original_eval_rewards = []
        # for i, w in enumerate(weights):
        #     print('Run {:02d} ...'.format(i), end='')
        #     if neuron_type == "LTC":
        #         agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        #     elif neuron_type == "CfC":
        #         agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
        #         w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        #     agent_net.load_state_dict(w)

        #     rewards = evaluate_LTC_agent_pole_length(agent_net, env_name, n_evaluations, evaluation_seeds, 1.0)
        #     original_eval_rewards.append(rewards)
        #     print(eraser*3 + '-> Avg reward: {:7.2f}'.format(np.mean(rewards)))
            

        # all_results.append((result_id, neuron_type, num_neurons, sparsity_level, learning_rate, factor, np.mean(np.mean(original_eval_rewards, axis = 1)), np.std(np.mean(original_eval_rewards, axis = 1))))
        # print(f"Mean avg reward: {np.mean(np.mean(original_eval_rewards, axis = 1))}, +/- {np.std(np.mean(original_eval_rewards, axis = 1))}")
        

        # pole_length_adaptation_eval_rewards = []
        # pole_length_mods = [0.1, 0.5, 2.0, 6.0, 10.0, 15.0, 20.0]
        # for i, w in enumerate(weights):
        #     print('Run {:02d} ...'.format(i), end='')
        #     if neuron_type == "LTC":
        #         agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        #     elif neuron_type == "CfC":
        #         agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
        #         w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        #     agent_net.load_state_dict(w)

        #     rewards_sum = 0
        #     for pole_length_mod in pole_length_mods:
        #         rewards = evaluate_LTC_agent_pole_length(agent_net, env_name, n_evaluations, evaluation_seeds, pole_length_mod)
        #         rewards_sum += np.mean(rewards)
            
        #     pole_length_adaptation_eval_rewards.append(rewards_sum/len(pole_length_mods))
        #     print(eraser*3 + '-> Avg adaptation reward: {:7.2f}'.format(rewards_sum/len(pole_length_mods)))

        # all_pole_length_adapt_results.append((result_id, neuron_type, num_neurons, sparsity_level, learning_rate, factor, np.mean(pole_length_adaptation_eval_rewards), np.std(pole_length_adaptation_eval_rewards)))
        # print(f"Mean avg pole length adapt reward: {np.mean(pole_length_adaptation_eval_rewards)}, +/- {np.std(pole_length_adaptation_eval_rewards)}")


        
        # pole_mass_adaptation_eval_rewards = []
        # pole_mass_mods = [5.0, 9.0, 13.0, 14.0, 18.0]
        # for i, w in enumerate(weights):
        #     print('Run {:02d} ...'.format(i), end='')
        #     if neuron_type == "LTC":
        #         agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        #     elif neuron_type == "CfC":
        #         agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
        #         w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        #     agent_net.load_state_dict(w)

        #     rewards_sum = 0
        #     for pole_mass_mod in pole_mass_mods:
        #         rewards = evaluate_LTC_agent_pole_mass(agent_net, env_name, n_evaluations, evaluation_seeds, pole_mass_mod)
        #         rewards_sum += np.mean(rewards)
            
        #     pole_mass_adaptation_eval_rewards.append(rewards_sum/len(pole_mass_mods))
        #     print(eraser*3 + '-> Avg adaptation reward: {:7.2f}'.format(rewards_sum/len(pole_mass_mods)))

        # all_pole_mass_adapt_results.append((result_id, neuron_type, num_neurons, sparsity_level, learning_rate, factor, np.mean(pole_mass_adaptation_eval_rewards), np.std(pole_mass_adaptation_eval_rewards)))
        # print(f"Mean avg pole mass adapt reward: {np.mean(pole_mass_adaptation_eval_rewards)}, +/- {np.std(pole_mass_adaptation_eval_rewards)}")
            


        # force_mag_adaptation_eval_rewards = []
        # force_mag_mods = [0.2, 2.0, 4.0, 6.0]
        # for i, w in enumerate(weights):
        #     print('Run {:02d} ...'.format(i), end='')
        #     if neuron_type == "LTC":
        #         agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
        #     elif neuron_type == "CfC":
        #         agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
        #         w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

        #     agent_net.load_state_dict(w)

        #     rewards_sum = 0
        #     for force_mag_mod in force_mag_mods:
        #         rewards = evaluate_LTC_agent_force_mag(agent_net, env_name, n_evaluations, evaluation_seeds, force_mag_mod)
        #         rewards_sum += np.mean(rewards)
            
        #     force_mag_adaptation_eval_rewards.append(rewards_sum/len(force_mag_mods))
        #     print(eraser*3 + '-> Avg adaptation reward: {:7.2f}'.format(rewards_sum/len(force_mag_mods)))

        # all_force_mag_adapt_results.append((result_id, neuron_type, num_neurons, sparsity_level, learning_rate, factor, np.mean(force_mag_adaptation_eval_rewards), np.std(force_mag_adaptation_eval_rewards)))
        # print(f"Mean avg force mag adapt reward: {np.mean(force_mag_adaptation_eval_rewards)}, +/- {np.std(force_mag_adaptation_eval_rewards)}")


        mixed_adaptation_eval_rewards = []
        pole_length_mods = [0.55, 10.5]
        pole_mass_mods = [3.0]
        force_mag_mods = [0.6, 3.5]
        for i, w in enumerate(weights):
            print('Run {:02d} ...'.format(i), end='')
            if neuron_type == "LTC":
                agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
            elif neuron_type == "CfC":
                agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
                w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))

            agent_net.load_state_dict(w)

            rewards_sum = 0
            for i in range(n_evaluations):
                pole_length_mod = np.random.choice(pole_length_mods)
                pole_mass_mod = np.random.choice(pole_mass_mods)
                force_mag_mod = np.random.choice(force_mag_mods)
                rewards_sum += np.mean(evaluate_agent_all_params(agent_net, env_name, 1, evaluation_seeds[i:], pole_length_mod, pole_mass_mod, force_mag_mod))
            
            rewards_sum /= n_evaluations
            mixed_adaptation_eval_rewards.append(rewards_sum)
            print(eraser*3 + '-> Avg adaptation reward: {:7.2f}'.format(rewards_sum))

        all_mixed_adaptation_eval_rewards.append((result_id, neuron_type, num_neurons, sparsity_level, learning_rate, factor, np.mean(mixed_adaptation_eval_rewards), np.std(mixed_adaptation_eval_rewards)))
        print(f"Mean avg mixed adapt reward: {np.mean(mixed_adaptation_eval_rewards)}, +/- {np.std(mixed_adaptation_eval_rewards)}")

        result_id += 1
        
        

        




        

        

            


print(all_results)
print(all_pole_length_adapt_results)
# print(sorted(all_results, key = lambda x: x[6], reverse=True))
# print(sorted(all_pole_length_adapt_results, key = lambda x: x[6], reverse=True))
print(all_pole_mass_adapt_results)
print(all_force_mag_adapt_results)
