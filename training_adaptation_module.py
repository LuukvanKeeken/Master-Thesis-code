from collections import OrderedDict, deque
from copy import deepcopy
from datetime import date
import gym
import numpy as np
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from Master_Thesis_Code.Adaptation_Module import StandardRNN
import os
import argparse
torch.autograd.set_detect_anomaly(True)

def get_privileged_info(env):
    pole_length = env.unwrapped.length
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    privileged_info = [pole_length, masspole, force_mag]
    return torch.tensor(privileged_info, dtype=torch.float32)

def randomize_env_params(env, randomization_params):
    pole_length = env.unwrapped.length
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    params = [pole_length, masspole, force_mag]

    
    for i in range(len(params)):
        if isinstance(randomization_params[i], float):
            low = params[i] - params[i] * randomization_params[i]
            high = params[i] + params[i] * randomization_params[i]
        elif isinstance(randomization_params[i], tuple):
            low = params[i]*randomization_params[i][0]
            high = params[i]*randomization_params[i][1]
            
        params[i] = np.random.uniform(low, high)

    env.unwrapped.length = params[0]
    env.unwrapped.masspole = params[1]
    env.unwrapped.force_mag = params[2]

    return env

def check_same_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def validate_adaptation_module(agent_net, encoder, adaptation_module, evaluation_seeds, env_name, num_validation_eps, max_steps):
    with torch.no_grad():
        pole_length_mods = [0.55, 10.5]
        pole_mass_mods = [3.0]
        force_mag_mods = [0.6, 3.5]
        
        
        validation_rewards = []
        validation_losses = []
        

        for episode in range(num_validation_eps):
            env = gym.make(env_name)
            policy_hidden_state = None
            adaptation_module_hidden_state = None
            adaptation_module_outputs = []
            encoder_outputs = []

            np.random.seed(evaluation_seeds[episode])
            pole_length_mod = np.random.choice(pole_length_mods)
            pole_mass_mod = np.random.choice(pole_mass_mods)
            force_mag_mod = np.random.choice(force_mag_mods)
            env.seed(int(evaluation_seeds[episode]))

            env.unwrapped.length = pole_length_mod
            env.unwrapped.masspole = pole_mass_mod
            env.unwrapped.force_mag = force_mag_mod

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

            for step in range(max_steps):

                adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
                adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)

                privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
                encoder_output = encoder(privileged_info)

                # Transform the state to the correct format and save it
                # to be used as previous state in the next time step.
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device)
                prev_state = state

                # In the first step, we don't save the output of the adaptation module
                # and the encoder, since they are based on a randomly sampled state and action.
                if not step == 0:
                    adaptation_module_outputs.append(adaptation_module_output)
                    encoder_outputs.append(encoder_output)
                
                # Feed the state and adaptation module input into the agent network
                policy_output, value, policy_hidden_state = agent_net(state.float(), policy_hidden_state, adaptation_module_output)

                # Get distribution over the action space and select
                # the action with the highest probability.
                policy_dist = torch.softmax(policy_output, dim = 1)
                action = torch.argmax(policy_dist).item()
                prev_action = action
                prev_action = torch.tensor(prev_action).view(1, -1)

                # Take a step in the environment
                state, r, done, _ = env.step(action)
                total_reward += r

                if done or step == max_steps - 1:
                    validation_rewards.append(total_reward)
                    assert len(adaptation_module_outputs) == len(encoder_outputs)
                    loss = torch.nn.MSELoss()
                    loss_val = loss(torch.stack(adaptation_module_outputs), torch.stack(encoder_outputs))
                    validation_losses.append(loss_val.item())
                    break

        return np.mean(validation_losses), np.mean(validation_rewards)







def train_adaptation_module(env, num_training_episodes, max_steps, agent_net, num_outputs, evaluation_seeds, i_run, neuron_type, encoder, adaptation_module, optimizer, selection_method = "100 episode average", gamma = 0.99, max_reward = 200, env_name = "CartPole-v0", num_validation_eps = 10, validate_every = 10, randomization_params = None, randomize_every = 5):


    training_losses = []
    training_total_rewards = []
    validation_losses = []
    validation_total_rewards = []

    best_validation_reward = -np.inf
    best_validation_reward_after = -1
    best_validation_loss = np.inf
    best_validation_loss_after = -1

    for episode in range(1, num_training_episodes + 1):
        if randomization_params and episode % randomize_every == 0:
            env = gym.make(env_name)
            env = randomize_env_params(env, randomization_params)

        policy_hidden_state = None
        adaptation_module_hidden_state = None
        total_reward = 0
        done = False

        selected_actions = []
        states = []
        adaptation_module_outputs = []
        encoder_outputs = []

        # Randomly sample an action and state to 
        # feed as the first input to the adaptation
        # module.
        prev_action = env.action_space.sample()
        prev_action = torch.tensor(prev_action).view(1, -1)
        prev_state = env.observation_space.sample()
        prev_state = torch.from_numpy(prev_state)
        prev_state = prev_state.unsqueeze(0).to(device)

        state = env.reset()
        for step in range(max_steps):
            # Concatenate the previous state and action to create one
            # vector. This vector is then fed into a copy of the adaptation module.
            adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
            adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)

            # Get the output of the encoder given privileged info.
            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            encoder_output = encoder(privileged_info)
            
            

            # Transform the state to the correct format and save it
            # to be used as previous state in the next time step.
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device)
            states.append(state)
            prev_state = state

            # In the first step, we don't save the output of the adaptation module
            # and the encoder, since they are based on a randomly sampled state and action.
            if not step == 0:
                adaptation_module_outputs.append(adaptation_module_output)
                encoder_outputs.append(encoder_output)
            
            # Feed the state and adaptation module input into the agent network
            policy_output, value, policy_hidden_state = agent_net(state.float(), policy_hidden_state, adaptation_module_output)


            # Get distribution over the action space and select
            # the action with the highest probability.
            policy_dist = torch.softmax(policy_output, dim = 1)
            action = torch.argmax(policy_dist).item()
            selected_actions.append(action)
            prev_action = action
            prev_action = torch.tensor(prev_action).view(1, -1)

            # Take a step in the environment
            state, r, done, _ = env.step(action)
            total_reward += r


            if done or step == max_steps - 1:
                training_total_rewards.append(total_reward)
                assert len(adaptation_module_outputs) == len(encoder_outputs)
                loss = torch.nn.MSELoss()
                loss_val = loss(torch.stack(adaptation_module_outputs), torch.stack(encoder_outputs))

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                training_losses.append(loss_val.item())
                print(f"Episode {episode}/{num_training_episodes}, loss: {loss_val.item()}, total reward: {total_reward}")

                if episode % validate_every == 0:
                    mean_valid_loss, mean_valid_reward = validate_adaptation_module(agent_net, encoder, adaptation_module, evaluation_seeds, env_name, num_validation_eps, max_steps)
                    validation_losses.append(mean_valid_loss)
                    validation_total_rewards.append(mean_valid_reward)
                    print(f"Validation loss: {mean_valid_loss}, validation reward: {mean_valid_reward}")

                    if mean_valid_reward >= best_validation_reward:
                        best_validation_reward = mean_valid_reward
                        best_validation_reward_after = episode
                        torch.save(adaptation_module.state_dict(), f"{results_dir}/best_adaptation_module_reward_{neuron_type}_A2C_{i_run}.pt")
                    if mean_valid_loss <= best_validation_loss:
                        best_validation_loss = mean_valid_loss
                        best_validation_loss_after = episode
                        torch.save(adaptation_module.state_dict(), f"{results_dir}/best_adaptation_module_loss_{neuron_type}_A2C_{i_run}.pt")
                break

    print(f"Best validation reward: {best_validation_reward} after {best_validation_reward_after} episodes")
    print(f"Best validation loss: {best_validation_loss} after {best_validation_loss_after} episodes")
    
    return training_losses, training_total_rewards, validation_losses, validation_total_rewards, best_validation_reward, best_validation_reward_after, best_validation_loss, best_validation_loss_after



parser = argparse.ArgumentParser(description='Train adaptation module for neuromodulated CfC')
parser.add_argument('--neuron_type', type=str, default='CfC', help='Type of neuron to train')
parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
parser.add_argument('--state_dims', type=int, default=4, help='Number of state dimensions')
parser.add_argument('--action_dims', type=int, default=1, help='Number of action dimensions')
parser.add_argument('--num_neurons_policy', type=int, default=32, help='Number of neurons in the policy network')
parser.add_argument('--num_neurons_adaptation', type=int, default=64, help='Number of neurons in the adaptation module')
parser.add_argument('--num_actions', type=int, default=2, help='Number of actions')
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--mode', type=str, default='neuromodulated', help='Mode of the CfC network')
parser.add_argument('--wiring', type=str, default='None', help='Wiring of the CfC network')
parser.add_argument('--neuromod_network_dims', type=int, nargs='+', default = [3, 192, 96], help='Dimensions of the neuromodulation network, without output layer')
parser.add_argument('--num_training_eps', type=int, default=1000, help="Number of episodes to train the adaptation module")
parser.add_argument('--env_name', type=str, default="CartPole-v0", help="Gym RL environment name")
parser.add_argument('--lr_adapt_mod', type=float, default=0.0005, help="Learning rate of the adaptation module")
parser.add_argument('--wd_adapt_mod', type=float, default=0.0, help="Weight decay of the adaptation module")
parser.add_argument('--training_range', type=str, default='quarter_range', help='Range from which training data is sampled')
parser.add_argument('--randomize_every', type=int, default=1, help='Number of episodes between randomization of environment parameters')
parser.add_argument('--validate_every', type=int, default=10, help='Number of training episodes between validations')
parser.add_argument('--num_validation_eps', type=int, default=10, help='Number of episodes to validate the adaptation module')
parser.add_argument('--adapt_mod_type', type=str, default='StandardRNN', help='Type of adaptation module to use')


args = parser.parse_args()
neuron_type = args.neuron_type
device = args.device
state_dims = args.state_dims
action_dims = args.action_dims
num_neurons_policy = args.num_neurons_policy
num_neurons_adaptation = args.num_neurons_adaptation
num_actions = args.num_actions
seed = args.seed
mode = args.mode
if args.wiring == 'None':
    wiring = None
neuromod_network_dims = args.neuromod_network_dims
neuromod_network_dims.append(num_neurons_policy)
num_training_eps = args.num_training_eps
env_name = args.env_name
lr_adapt_mod = args.lr_adapt_mod
wd_adapt_mod = args.wd_adapt_mod
training_range = args.training_range
randomize_every = args.randomize_every
validate_every = args.validate_every
num_validation_eps = args.num_validation_eps
adapt_mod_type = args.adapt_mod_type
if training_range == 'quarter_range':
    randomization_params = [(0.775, 5.75), (1.0, 2.0), (0.8, 2.25)]
else:
    raise NotImplementedError

if env_name == 'CartPole-v0':
    env = gym.make('CartPole-v0')
else:
    raise NotImplementedError
evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')

phase_1_dir = "CfC_a2c_result_296_202437_learningrate_0.0005_selectiomethod_range_evaluation_all_params_trainingmethod_original_numneurons_32_tausysextraction_True_mode_neuromodulated_neuromod_network_dims_3_192_96_32"


dirs = os.listdir('Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/')
if not any('adaptation_module' in d for d in dirs):
    result_id = 1
else:
    results = [d for d in dirs if 'adaptation_module' in d]
    result_id = len(results) + 1
d = date.today()


results_dir = f"Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/adaptation_module_{adapt_mod_type}_result_{result_id}_{str(d.year) + str(d.month) + str(d.day)}_CfC_result_296_202437_numneuronsadaptmod_{num_neurons_adaptation}_lradaptmod_{lr_adapt_mod}_wdadaptmod_{wd_adapt_mod}"
os.mkdir(results_dir)


weights_0 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))
weights_1 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
weights_2 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
weights_3 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
weights_4 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
weights_5 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_5.pt', map_location=torch.device(device))
weights_6 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_6.pt', map_location=torch.device(device))
weights_7 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_7.pt', map_location=torch.device(device))
weights_8 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_8.pt', map_location=torch.device(device))
weights_9 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{phase_1_dir}/checkpoint_{neuron_type}_A2C_9.pt', map_location=torch.device(device))
weights = [weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9]


all_training_losses = []
all_training_total_rewards = []
all_validation_losses = []
all_validation_total_rewards = []
best_validation_rewards = []
best_validation_losses = []
for i, w in enumerate(weights):
    print(f"Training adaptation module for model {i+1}")

    if neuron_type == "CfC":
        agent_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring)
        
        
        layer_list = []
        for dim in range(len(neuromod_network_dims) - 1):
            layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
            layer_list.append(torch.nn.Tanh())
        encoder = torch.nn.Sequential(*layer_list)
        
        w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
        w_policy = OrderedDict((k, v) for k, v in w.items() if 'neuromod' not in k)
        w_encoder = OrderedDict((k.split('.', 3)[-1], v) for k, v in w.items() if 'neuromod' in k)
    elif neuron_type == "LTC":
        raise NotImplementedError
    
    agent_net.load_state_dict(w_policy)
    encoder.load_state_dict(w_encoder)

    
    for name, param in encoder.named_parameters():
        param.requires_grad = False

    if adapt_mod_type == 'StandardRNN':
        adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptation, num_neurons_policy, seed = seed)
    else:
        raise NotImplementedError
    

    optimizer = torch.optim.Adam(adaptation_module.parameters(), lr = lr_adapt_mod, weight_decay = wd_adapt_mod)

    training_losses, training_total_rewards, validation_losses, validation_total_rewards, best_validation_reward, best_validation_reward_after, best_validation_loss, best_validation_loss_after = train_adaptation_module(env, num_training_eps, 200, agent_net, num_actions, evaluation_seeds, i, neuron_type, encoder, adaptation_module, optimizer, randomization_params=randomization_params, randomize_every=randomize_every, validate_every=validate_every, num_validation_eps=num_validation_eps)
    all_training_losses.append(training_losses)
    all_training_total_rewards.append(training_total_rewards)
    all_validation_losses.append(validation_losses)
    all_validation_total_rewards.append(validation_total_rewards)
    best_validation_rewards.append((best_validation_reward, best_validation_reward_after))
    best_validation_losses.append((best_validation_loss, best_validation_loss_after))
    
np.save(f"{results_dir}/all_training_losses.npy", all_training_losses)
np.save(f"{results_dir}/all_training_total_rewards.npy", all_training_total_rewards)
np.save(f"{results_dir}/all_validation_losses.npy", all_validation_losses)
np.save(f"{results_dir}/all_validation_total_rewards.npy", all_validation_total_rewards)


with open(f"{results_dir}/best_validation_reward_after.txt", "w") as f:
    for i, reward_results in enumerate(best_validation_rewards):
        f.write(f"{i}: {reward_results[0]} after {reward_results[1]}\n")

    f.write(f"Average training episodes: {np.mean([x[1] for x in best_validation_rewards])}\n")
    f.write(f"Mean average reward: {np.mean([x[0] for x in best_validation_rewards])}")

with open(f"{results_dir}/best_validation_loss_after.txt", "w") as f:
    for i, loss_results in enumerate(best_validation_losses):
        f.write(f"{i}: {loss_results[0]} after {loss_results[1]}\n")

    f.write(f"Average training episodes: {np.mean([x[1] for x in best_validation_losses])}\n")
    f.write(f"Mean average loss: {np.mean([x[0] for x in best_validation_losses])}")
