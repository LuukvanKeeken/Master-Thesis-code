from datetime import date
import random
import gym
import numpy as np
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from Master_Thesis_Code.Neuromodulated_Agent import NeuromodulatedAgent
from Master_Thesis_Code.backpropamine_A2C import BP_RNetwork
from ncps_time_constant_extraction.ncps.wirings import AutoNCP



def evaluate_BW(agent_net, env_name, num_episodes, evaluation_seeds, env_parameter_settings = None):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)

        for param, value in env_parameter_settings.items():
            setattr(env.unwrapped, param, value)
            
        for i_episode in range(num_episodes):
            policy_hidden_states = None
            
            env.seed(int(evaluation_seeds[i_episode]))
            
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device) #This as well?
                privileged_info = get_privileged_info(env_parameter_settings).unsqueeze(0).to(device)
                policy_outputs, values, policy_hidden_states = agent_net(state.float(), privileged_info, policy_hidden_states)

                
                means, std_devs = policy_outputs
                
                # Get greedy action
                action = means
                

                state, r, done, _ = env.step(action[0].cpu().numpy())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards



def get_privileged_info(env_parameter_settings):
    params_values = [val for val in env_parameter_settings.values()]
    params_tensor = torch.tensor(params_values, dtype=torch.float32)

    return params_tensor





gym.envs.registration.register(
    id='AdjustableBipedalWalker-v3',
    entry_point='Master_Thesis_Code.AdjustableBipedalWalker:AdjustableBipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)



training_ranges = [(0.925, 1.125), (0.9875, 1.025), (0.925, 1.125), (0.9875, 1.025), (0.875, 1.125), (0.925, 1.025), (0.95, 1.075), (0.875, 1.125), (0.875, 1.125), (0.975, 1.05), (0.925, 1.01875)]
validation_ranges = [[(0.85, 0.925), (1.125, 1.25)], [(0.975, 0.9875), (1.025, 1.05)], [(0.85, 0.925), (1.125, 1.25)], [(0.975, 0.9875), (1.025, 1.05)], [(0.75, 0.875), (1.125, 1.25)], [(0.85, 0.925), (1.025, 1.05)], [(0.9, 0.95), (1.075, 1.15)], [(0.75, 0.875), (1.125, 1.25)], [(0.75, 0.875), (1.125, 1.25)], [(0.95, 0.975), (1.05, 1.1)], [(0.85, 0.925), (1.01875, 1.0375)]]
testing_ranges = [[(0.7, 0.85), (1.25, 1.5)], [(0.95, 0.975), (1.05, 1.1)], [(0.7, 0.85), (1.25, 1.5)], [(0.95, 0.975), (1.05, 1.1)], [(0.5, 0.75), (1.25, 1.5)], [(0.7, 0.85), (1.05, 1.1)], [(0.8, 0.9), (1.15, 1.3)], [(0.5, 0.75), (1.25, 1.5)], [(0.5, 0.75), (1.25, 1.5)], [(0.9, 0.95), (1.1, 1.2)], [(0.7, 0.85), (1.0375, 1.075)]]
# training_ranges_bpstdavg = [(0.86875, 1.121875), (0.989375, 1.021875), (0.86875, 1.121875), (0.989375, 1.021875), (0.840625, 1.55625), (0.9125, 1.06875), (0.928125, 1.153125), (0.8625, 1.23125), (0.8625, 1.23125), (0.9765625, 1.071875), (0.9265625, 1.015625)]
# training_ranges_true = [(0.95, 1.025), (0.99375, 1.0125), (0.95, 1.025), (0.99375, 1.0125), (0.875, 2.0), (0.95, 1.0125), (0.975, 1.25), (0.95, 1.025), (0.95, 1.025), (0.99375, 1.05), (0.925, 1.00625)]
# training_ranges_old = [(0.79, 1.22),(0.99, 1.03),(0.79, 1.22),(0.99, 1.03),(0.81, 1.11),(0.88, 1.13),(0.88, 1.06),(0.78, 1.44),(0.78, 1.44),(0.96, 1.09),(0.93, 1.03)]
# validation_ranges_bpstdavg = [[(0.7375, 0.86875), (1.121875, 1.24375)], [(0.97875, 0.989375), (1.021875, 1.04375)], [(0.7375, 0.86875), (1.121875, 1.24375)], [(0.97875, 0.989375), (1.021875, 1.04375)], [(0.68125, 0.840625), (1.55625, 2.1125)], [(0.825, 0.9125), (1.06875, 1.1375)], [(0.85625, 0.928125), (1.153125, 1.30625)], [(0.725, 0.8625), (1.23125, 1.4625)], [(0.725, 0.8625), (1.23125, 1.4625)], [(0.953125, 0.9765625), (1.071875, 1.14375)], [(0.853125, 0.9265625), (1.015625, 1.03125)]]
# validation_ranges_true = [[(0.9, 0.95), (1.025, 1.05)], [(0.9875, 0.99375), (1.0125, 1.025)], [(0.9, 0.95), (1.025, 1.05)], [(0.9875, 0.99375), (1.0125, 1.025)], [(0.75, 0.875), (2.0, 3.0)], [(0.9, 0.95), (1.0125, 1.025)], [(0.95, 0.975), (1.25, 1.5)], [(0.9, 0.95), (1.025, 1.05)], [(0.9, 0.95), (1.025, 1.05)], [(0.9875, 0.99375), (1.05, 1.1)], [(0.85, 0.925), (1.00625, 1.0125)]]
# validation_ranges_old = [[(0.58, 0.79), (1.22, 1.44)], [(0.97, 0.99), (1.03, 1.06)], [(0.58, 0.79), (1.22, 1.44)], [(0.97, 0.99), (1.03, 1.06)], [(0.61, 0.81), (1.11, 1.23)], [(0.75, 0.88), (1.13, 1.25)], [(0.76, 0.88), (1.06, 1.11)], [(0.55, 0.78), (1.44, 1.88)], [(0.55, 0.78), (1.44, 1.88)], [(0.92, 0.96), (1.09, 1.19)], [(0.86, 0.93), (1.03, 1.05)]]
# testing_ranges_bpstdavg = [[(0.475, 0.7375), (1.24375, 1.4875)], [(0.9575, 0.97875), (1.04375, 1.0875)], [(0.475, 0.7375), (1.24375, 1.4875)], [(0.9575, 0.97875), (1.04375, 1.0875)], [(0.3625, 0.68125), (2.1125, 3.225)], [(0.65, 0.825), (1.1375, 1.275)], [(0.7125, 0.85625), (1.30625, 1.6125)], [(0.45, 0.725), (1.4625, 1.925)], [(0.45, 0.725), (1.4625, 1.925)], [(0.90625, 0.953125), (1.14375, 1.2875)], [(0.70625, 0.853125), (1.03125, 1.0625)]]
# testing_ranges_true = [[(0.8, 0.9), (1.05, 1.1)], [(0.975, 0.9875), (1.025, 1.05)], [(0.8, 0.9), (1.05, 1.1)], [(0.975, 0.9875), (1.025, 1.05)], [(0.5, 0.75), (3.0, 5.0)], [(0.8, 0.9), (1.025, 1.05)], [(0.9, 0.95), (1.5, 2.0)], [(0.8, 0.9), (1.05, 1.1)], [(0.8, 0.9), (1.05, 1.1)], [(0.975, 0.9875), (1.1, 1.2)], [(0.7, 0.85), (1.0125, 1.025)]]
# testing_ranges_old = [[(0.15, 0.58), (1.44, 1.88)], [(0.94, 0.97), (1.06, 1.13)], [(0.15, 0.58), (1.44, 1.88)], [(0.94, 0.97), (1.06, 1.13)], [(0.23, 0.61), (1.23, 1.45)], [(0.5, 0.75), (1.25, 1.5)], [(0.53, 0.76), (1.11, 1.23)], [(0.1, 0.55), (1.88, 2.75)], [(0.1, 0.55), (1.88, 2.75)], [(0.84, 0.92), (1.19, 1.38)], [(0.71, 0.86), (1.05, 1.1)]]

default_values = [8.0, 34.0, 8.0, 34.0, 2.5, 4.0, 6.0, 1.0, 1.0, 5.0, 160.0]
env_params = ['left_leg_w_unscaled', 'left_leg_h_unscaled', 'right_leg_w_unscaled', 'right_leg_h_unscaled', 'terrain_friction', 'speed_hip', 'speed_knee', 'left_leg_density', 'right_leg_density', 'hull_density', 'lidar_range_unscaled']


device = "cpu"
neuron_type = "CfC"
if neuron_type == "BP":
    top_dir = "BP_A2C"
    model_signifier = "BP"
else:
    top_dir = "LTC_A2C"
    model_signifier = "CfC"
mode = "only_neuromodulated"
num_neurons_policy = 96
neuromod_network_dims = [11, 256, 128, num_neurons_policy]
batch_dir = "bipedal_walker"
encoder_func = "tanh"
num_models = 9
seed = 5
env_name = "AdjustableBipedalWalker-v3"
n_evaluations = 1000

wiring = None

if encoder_func == "tanh":
    encoder_hidden_activation = torch.nn.Tanh()
    encoder_output_activation = torch.nn.Tanh()
elif encoder_func == "relu":
    encoder_hidden_activation = torch.nn.ReLU()
    encoder_output_activation = torch.nn.ReLU()



evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')
arrays = [evaluation_seeds]

# Generate the new arrays and add them to the list
for i in range(1, 10):
    new_array = evaluation_seeds + i
    arrays.append(new_array)

# Concatenate all arrays together
evaluation_seeds = np.concatenate(arrays)

# result_dir = "CfC_a2c_result_3200008_2024628_learningrate_0.0001_numneurons_96_encoutact_relu_mode_neuromodulated_neuromod_network_dims_11_256_128_96"



# policy_weights_0 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
# policy_weights_1 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
# policy_weights_2 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
# policy_weights_3 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
# policy_weights_4 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
# policy_weights_5 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_5.pt', map_location=torch.device(device))
# policy_weights_6 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_6.pt', map_location=torch.device(device))
# policy_weights_7 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_7.pt', map_location=torch.device(device))
# policy_weights_8 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_8.pt', map_location=torch.device(device))
# policy_weights_9 = torch.load(f'Master_Thesis_Code/{top_dir}/14_NMBP_reluandtanh_32and64/{result_dir}/checkpoint_{neuron_type}_A2C_9.pt', map_location=torch.device(device))
# policy_weights = [policy_weights_0, policy_weights_1, policy_weights_2, policy_weights_3, policy_weights_4, policy_weights_5, policy_weights_6, policy_weights_7, policy_weights_8, policy_weights_9]
# policy_weights = [policy_weights_0]
policy_weights_1 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300000_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_2 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300001_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_3 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300002_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_4 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300003_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_5 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300004_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_6 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300005_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_7 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300006_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_8 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300007_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights_9 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/CfC_a2c_result_4300008_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_mode_only_neuromodulated_neuromod_network_dims_11_256_128_96/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
policy_weights = [policy_weights_1, policy_weights_2, policy_weights_3, policy_weights_4, policy_weights_5, policy_weights_6, policy_weights_7, policy_weights_8, policy_weights_9]
eraser = '\b \b'


training_rewards = []
validation_rewards = []
testing_rewards = []
with torch.no_grad():
    for i, w in enumerate(policy_weights):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            agent_net = LTC_Network(24, num_neurons_policy, 4, seed, wiring = wiring).to(device)
        elif neuron_type == "CfC":
            layer_list = []
            for dim in range(len(neuromod_network_dims) - 1):
                layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
                if dim < len(neuromod_network_dims)-2:
                    layer_list.append(encoder_hidden_activation)
                else:
                    layer_list.append(encoder_output_activation)
            encoder = torch.nn.Sequential(*layer_list)

            policy_net = CfC_Network(24, num_neurons_policy, 4, seed, mode = mode, wiring = wiring, continuous_actions=True).to(device)

            agent_net = NeuromodulatedAgent(policy_net, encoder, policy_has_hidden_state=True).to(device)
            w['policy_net.cfc_model.rnn_cell.tau_system'] = torch.reshape(w['policy_net.cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
        elif neuron_type == "BP":
            layer_list = []
            for dim in range(len(neuromod_network_dims) - 1):
                layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
                if dim < len(neuromod_network_dims)-2:
                    layer_list.append(encoder_hidden_activation)
                else:
                    layer_list.append(encoder_output_activation)
            encoder = torch.nn.Sequential(*layer_list)

            policy_net = BP_RNetwork(24, num_neurons_policy, 4, seed, external_neuromodulation = True, continuous_actions=True).to(device)

            agent_net = NeuromodulatedAgent(policy_net, encoder, policy_has_hidden_state=True).to(device)

        agent_net.load_state_dict(w)

        
        # Training rewards ---------------------------
        rewards_sum = 0
        all_rewards = []
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            random_env_param_settings = {}
            for param, fract_range, default_val in zip(env_params, training_ranges, default_values):
                random_env_param_settings[param] = np.random.uniform(fract_range[0], fract_range[1]) * default_val

            r = np.mean(evaluate_BW(agent_net, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
            rewards_sum += r
            all_rewards.append(r)

        rewards_sum /= n_evaluations
        training_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg training reward: {:7.2f}'.format(rewards_sum))
        print(np.std(all_rewards))
        all_rewards_train = all_rewards

        # Validation rewards ---------------------------
        rewards_sum = 0
        all_rewards = []
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            random_env_param_settings = {}
            for param, ranges, default_val in zip(env_params, validation_ranges, default_values):
                val_range = random.choice(ranges)
                random_env_param_settings[param] = np.random.uniform(val_range[0], val_range[1]) * default_val

            r = np.mean(evaluate_BW(agent_net, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
            rewards_sum += r
            all_rewards.append(r)

        rewards_sum /= n_evaluations
        validation_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg validation reward: {:7.2f}'.format(rewards_sum))
        print(np.std(all_rewards))
        all_rewards_val = all_rewards

        # Testing rewards ---------------------------
        rewards_sum = 0
        all_rewards = []
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            random_env_param_settings = {}
            for param, ranges, default_val in zip(env_params, testing_ranges, default_values):
                test_range = random.choice(ranges)
                random_env_param_settings[param] = np.random.uniform(test_range[0], test_range[1]) * default_val

            r = np.mean(evaluate_BW(agent_net, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
            rewards_sum += r
            all_rewards.append(r)

        rewards_sum /= n_evaluations
        testing_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg testing reward: {:7.2f}'.format(rewards_sum))
        print(np.std(all_rewards))
        all_rewards_test = all_rewards







    with open(f"Master_Thesis_Code/{top_dir}/{batch_dir}/remaining_ONMCfC/train_val_test.txt", "w") as f:
        f.write(f"All training rewards: {training_rewards}\n")
        print(f"All training rewards: {training_rewards}")
    # print(f"Alll training rewards: {all_rewards_train}")
        f.write(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}\n")
        print(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}")
        f.write(f"All validation rewards: {validation_rewards}\n")
        print(f"All validation rewards: {validation_rewards}")
    # print(f"All validation rewards: {all_rewards_val}")
        f.write(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}\n")
        print(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}")
        f.write(f"All testing rewards: {testing_rewards}\n")
        print(f"All testing rewards: {testing_rewards}")
    # print(f"All testing rewards: {all_rewards_test}")
        f.write(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}\n")
        print(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}")



        
        

        


