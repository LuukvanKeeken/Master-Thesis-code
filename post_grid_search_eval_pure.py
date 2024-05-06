from datetime import date
import random
import gym
import numpy as np
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from Master_Thesis_Code.Neuromodulated_Agent import NeuromodulatedAgent
from Master_Thesis_Code.backpropamine_A2C import BP_RNetwork, Standard_RNetwork, Standard_FFNetwork
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
            state = state.unsqueeze(0).to(device)

            policy_output, value, hidden_state = agent_net(state.float(), hidden_state)

            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards


def evaluate_agent_all_fdzgvzrdffgparams(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier, pole_mass_modifier, force_mag_modifier):

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
    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
    env.unwrapped.masspole *= pole_mass_modifier
    env.unwrapped.force_mag *= force_mag_modifier
        
    for i_episode in range(num_episodes):
        policy_hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            policy_output, value, policy_hidden_state = agent_net(state.float(), privileged_info, policy_hidden_state)
            
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


nums_neurons = [48]
types = ["Standard_MLP"]
learning_rates = [0.001, 0.00005, 0.000001]
entropy_coefs = [0.0, 0.0001, 0.01, 1.0]
value_coefs = [0.0001, 0.01, 0.5, 1.0]

result_id = 25000
flag = False

all_results = []
all_pole_length_adapt_results = []
all_pole_mass_adapt_results = []
all_force_mag_adapt_results = []
all_mixed_adaptation_eval_rewards = []
d = 202453


for neuron_type in types:
    for num_neurons in nums_neurons:
        for learning_rate in learning_rates:
            for entropy_coef in entropy_coefs:
                for value_coef in value_coefs:
                    if result_id not in [44000, 44001, 44002, 44003, 44004, 44005, 44006, 44008, 44009, 44010, 44011, 44012, 44016, 44017, 44020, 44021, 44024, 44025, 44028, 44029]:
                        result_id += 1
                        continue

                    print(f"num_neurons: {num_neurons}, learning rate: {learning_rate}, type: {neuron_type}")
                    
                    if neuron_type == "StandardRNN":
                        model_signifier = "Standard_RNN"
                        top_dir = "BP_A2C"
                    elif neuron_type == "Standard_MLP":
                        model_signifier = "Standard_MLP"
                        top_dir = "BP_A2C"
                    elif neuron_type == "BP":
                        model_signifier = "BP_RNN"
                        top_dir = "BP_A2C"
                    
                    
                    device = "cpu"
                    # learning_rate = 0.001
                    factor = 0.1
                    selection_method = "range_evaluation_all_params"
                    gamma = 0.99
                    training_method = "original"
                    # num_neurons = 32
                    mode = "pure"
                    
                    # neuromod_network_dims = [3, 256, 128, num_neurons]
                    if training_method == "quarter_range":
                        randomization_params = [(0.775, 5.75), (1.0, 2.0), (0.8, 2.25)]
                    else:
                        randomization_params = 3*[factor]

                    tau_sys_extraction = True

                    num_models = 10
                    sparsity_level = 0.5
                    seed = 5
                    # wiring = AutoNCP(num_neurons, 3, sparsity_level=sparsity_level, seed=seed)
                    wiring = None
                    env_name = "CartPole-v0"
                    n_evaluations = 100
                    evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')

                    result_dir = f'{model_signifier}_a2c_result_' + str(result_id) + f'_{str(d)}_entropycoef_{entropy_coef}_valuepredcoef_{value_coef}_learningrate_{learning_rate}_numtrainepisodes_40000_selectionmethod_true_range_eval_all_params_trainingmethod_quarter_range_numneurons_{num_neurons}'
                    
                    # result_dir  = f'{neuron_type}_' + str(result_id) + f'_{str(d)}_lr_{learning_rate}_nn_{num_neurons}_encoutact_relu_mode_neuromodulated_neuromod_network_dims_{"_".join(map(str, neuromod_network_dims[:-1]))}'
                    # result_dir = f'{neuron_type}_a2c_result_' + str(result_id) + f'_{str(d)}_learningrate_{learning_rate}_selectiomethod_{selection_method}_gamma_{gamma}_trainingmethod_{training_method}_numneurons_{num_neurons}_tausysextraction_{tau_sys_extraction}'
                    if neuron_type == "CfC":
                        result_dir += "_mode_" + mode
                    if wiring:
                        result_dir += "_wiring_" + "AutoNCP"
                    # if randomization_params:
                    #     result_dir += "_randomization_params_" + str(randomization_params)


                    weights_0 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
                    weights_1 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_1.pt', map_location=torch.device(device))
                    weights_2 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_2.pt', map_location=torch.device(device))
                    weights_3 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_3.pt', map_location=torch.device(device))
                    weights_4 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_4.pt', map_location=torch.device(device))
                    weights_5 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_5.pt', map_location=torch.device(device))
                    weights_6 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_6.pt', map_location=torch.device(device))
                    weights_7 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_7.pt', map_location=torch.device(device))
                    weights_8 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_8.pt', map_location=torch.device(device))
                    weights_9 = torch.load(f'Master_Thesis_Code/{top_dir}/5_MLP_range_48/{result_dir}/checkpoint_{model_signifier}_A2C_9.pt', map_location=torch.device(device))
                    weights = [weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9]


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
                    # pole_length_mods = [0.55, 10.5]
                    # pole_mass_mods = [3.0]
                    # force_mag_mods = [0.6, 3.5]
                    validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]

                    for i, w in enumerate(weights):
                        print('Run {:02d} ...'.format(i), end='')
                        if neuron_type == "LTC":
                            agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
                        elif neuron_type == "CfC":
                            agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring).to(device)
                            w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons,))
                            # layer_list = []
                            # for dim in range(len(neuromod_network_dims) - 1):
                            #     layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
                            #     if dim < len(neuromod_network_dims)-2:
                            #         layer_list.append(encoder_hidden_activation)
                            #     else:
                            #         layer_list.append(encoder_output_activation)
                            # encoder = torch.nn.Sequential(*layer_list)
                
                            # policy_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring).to(device)

                            # agent_net = NeuromodulatedAgent(policy_net, encoder, policy_has_hidden_state=True).to(device)
                            # w['policy_net.cfc_model.rnn_cell.tau_system'] = torch.reshape(w['policy_net.cfc_model.rnn_cell.tau_system'], (num_neurons,))
                        elif neuron_type == "BP":
                            agent_net = BP_RNetwork(4, num_neurons, 2, seed).to(device)
                        elif neuron_type == "StandardRNN":
                            agent_net = Standard_RNetwork(4, num_neurons, 2, seed).to(device)



                        agent_net.load_state_dict(w)

                        rewards_sum = 0
                        for i in range(n_evaluations):
                            np.random.seed((evaluation_seeds[i] + seed)%(2**32))
                            random.seed((evaluation_seeds[i] + seed)%(2**32))
                            pole_length_range = random.choice(validation_ranges[0])
                            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                            force_mag_range = random.choice(validation_ranges[2])
                            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                            rewards_sum += np.mean(evaluate_agent_all_params(agent_net, env_name, 1, evaluation_seeds[i:], pole_length_mod, pole_mass_mod, force_mag_mod))
                        
                        rewards_sum /= n_evaluations
                        mixed_adaptation_eval_rewards.append(rewards_sum)
                        print(eraser*3 + '-> Avg adaptation reward: {:7.2f}'.format(rewards_sum))

                    all_mixed_adaptation_eval_rewards.append((result_id, neuron_type, num_neurons, learning_rate, entropy_coef, value_coef, np.mean(mixed_adaptation_eval_rewards), np.std(mixed_adaptation_eval_rewards)))
                    print(f"Mean avg mixed adapt reward: {np.mean(mixed_adaptation_eval_rewards)}, +/- {np.std(mixed_adaptation_eval_rewards)}")

                    result_id += 1
        
        

        




        

        

            


print(all_results)
print(all_pole_length_adapt_results)
# print(sorted(all_results, key = lambda x: x[6], reverse=True))
# print(sorted(all_pole_length_adapt_results, key = lambda x: x[6], reverse=True))
print(all_pole_mass_adapt_results)
print(all_force_mag_adapt_results)
print(all_mixed_adaptation_eval_rewards)