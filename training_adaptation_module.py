from collections import deque
import gym
import numpy as np
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from Master_Thesis_Code.Adaptation_Module import StandardRNN
import os
import argparse


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



def train_adaptation_module(env, num_training_episodes, max_steps, agent_net, num_outputs, evaluation_seeds, i_run, neuron_type, encoder, selection_method = "100 episode average", gamma = 0.99, max_reward = 200, env_name = "CartPole-v0", num_evaluation_episodes = 10, evaluate_every = 10, randomization_params = None, randomize_every = 5):



    for episode in range(1, num_training_episodes + 1):

        if randomization_params and episode % randomize_every == 0:
            env = gym.make(env_name)
            env = randomize_env_params(env, randomization_params)

        selected_actions = []
        states = []
        adaptation_module_outputs = []
        encoder_outputs = []

        # Randomly sample an action and state to 
        # feed as the first input to the adaptation
        # module.
        prev_action = env.action_space.sample()
        prev_action = torch.from_numpy(prev_action)
        prev_action = 
        prev_state = env.observation_space.sample()
        prev_state = torch.from_numpy(prev_state)
        prev_state = prev_state.unsqueeze(0).to(device)

        state = env.reset()
        for steps in range(max_steps):
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device)
            states.append(state)

            adaptation_module_input = 

            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            encoder_output = encoder(privileged_info)
            encoder_outputs.append(encoder_output)










def train_adaptation_module_old(env, num_training_episodes, max_steps, agent_net, num_outputs, evaluation_seeds, i_run, neuron_type, selection_method = "100 episode average", gamma = 0.99, max_reward = 200, env_name = "CartPole-v0", num_evaluation_episodes = 10, evaluate_every = 10, randomization_params = None, randomize_every = 5):
    best_average = -np.inf
    best_average_after = np.inf
    scores = []
    smoothed_scores = []
    scores_window = deque(maxlen = 100)

    entropy_term = 0

    for episode in range(1, num_training_episodes + 1):
        
        if randomization_params and episode % randomize_every == 0:
            env = gym.make(env_name)
            env = randomize_env_params(env, randomization_params)

        hidden_state = None

        score = 0

        log_probs = []
        values = []
        rewards = []
        states = []
        actions = []

        state = env.reset()
        for steps in range(max_steps):
            # Feed the state into the network
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device)

            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            policy_output, value, hidden_state = agent_net((state.float(), privileged_info), hidden_state)

            # Get distribution over the action space
            policy_dist = torch.softmax(policy_output, dim = 1)
            value = value.detach().cpu().numpy()[0, 0]
            dist = policy_dist.detach().cpu().numpy()

            # Sample an action from the distribution
            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            new_state, reward, done, _ = env.step(action)

            score += reward

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done or steps == max_steps - 1:
                new_state = torch.from_numpy(new_state)
                new_state = new_state.unsqueeze(0).to(device)
                privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
                _, Qval, hidden_state = agent_net((new_state.float(), privileged_info), hidden_state)
                Qval = Qval.detach().cpu().numpy()[0, 0]

                if ((selection_method == "evaluation") and (episode % evaluate_every == 0)):
                    evaluation_performance = np.mean(evaluate_agent_pole_length(agent_net, env_name, num_evaluation_episodes, evaluation_seeds, 1.0))
                    print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                    if evaluation_performance >= best_average:
                        best_average = evaluation_performance
                        best_average_after = episode
                        torch.save(agent_net.state_dict(),
                                       result_dir + f'/checkpoint_{neuron_type}_A2C_{i_run}.pt')

                    if best_average == max_reward:
                        print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                        best_average_after, f'. Model saved in folder {result_dir}')
                        return smoothed_scores, scores, best_average, best_average_after

                elif ((selection_method == "range_evaluation") and (episode % evaluate_every == 0)):
                    pole_length_mods = [0.1, 0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 17.0, 20.0]
                    # pole_length_mods = [0.55, 10.5]
                    eps_per_setting = 1
                    evaluation_performance = 0
                    for i, mod in enumerate(pole_length_mods):
                        # Get performance over one episode with this pole length modifier, 
                        # skip over the first i evaluation seeds so not all episodes have
                        # the same seed.
                        evaluation_performance += np.mean(evaluate_agent_pole_length(agent_net, env_name, eps_per_setting, evaluation_seeds[i+eps_per_setting:], mod))

                    evaluation_performance /= len(pole_length_mods)
                    print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                    if evaluation_performance >= best_average:
                        best_average = evaluation_performance
                        best_average_after = episode
                        torch.save(agent_net.state_dict(),
                                       result_dir + f'/checkpoint_{neuron_type}_A2C_{i_run}.pt')
                    
                    if best_average == max_reward:
                        print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                        best_average_after, f'. Model saved in folder {result_dir}')
                        return smoothed_scores, scores, best_average, best_average_after

                elif ((selection_method == "range_evaluation_all_params") and (episode % evaluate_every == 0)):
                    pole_length_mods = [0.55, 10.5]
                    pole_mass_mods = [3.0]
                    force_mag_mods = [0.6, 3.5]

                    eps_per_setting = 1
                    evaluation_performance = 0
                    total_eval_eps = 10
                    for i in range(total_eval_eps):
                        np.random.seed(evaluation_seeds[i+eps_per_setting-1])
                        pole_length_mod = np.random.choice(pole_length_mods)
                        pole_mass_mod = np.random.choice(pole_mass_mods)
                        force_mag_mod = np.random.choice(force_mag_mods)
                        evaluation_performance += np.mean(evaluate_agent_all_params(agent_net, env_name, eps_per_setting, evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                    evaluation_performance /= total_eval_eps
                    print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                    if evaluation_performance >= best_average:
                        best_average = evaluation_performance
                        best_average_after = episode
                        torch.save(agent_net.state_dict(),
                                       result_dir + f'/checkpoint_{neuron_type}_A2C_{i_run}.pt')
                        
                    if best_average == max_reward:
                        print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                        best_average_after, f'. Model saved in folder {result_dir}')
                        return smoothed_scores, scores, best_average, best_average_after
                    


                elif (selection_method == "100 episode average"):
                    scores_window.append(score)
                    scores.append(score)
                    smoothed_scores.append(np.mean(scores_window))

                    if smoothed_scores[-1] >= best_average:
                        best_average = smoothed_scores[-1]
                        best_average_after = episode
                        # SAVE MODEL HERE
                    
                    print("Episode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_window)), end='\r')

                    if episode % 100 == 0:
                        print("\rEpisode {}\tAverage Score: {:.2f}".
                            format(episode, np.mean(scores_window)))
                        
                break

        # Compute the Q-values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + gamma * Qval
            Qvals[t] = Qval

        # Update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        advantage = advantage.to(device)
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        # 0.001 IS A MAGIC NUMBER!!
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()
    
    print(f'Best {selection_method}: ', best_average, ' reached at episode ',
              best_average_after, f'. Model saved in folder {result_dir}')
    
    return smoothed_scores, scores, best_average, best_average_after







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
parser.add_argument('--num_training_eps', type=int, default=20000, help="Number of episodes to train the adaptation module")



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



results_dir = "CfC_a2c_result_296_202437_learningrate_0.0005_selectiomethod_range_evaluation_all_params_trainingmethod_original_numneurons_32_tausysextraction_True_mode_neuromodulated_neuromod_network_dims_3_192_96_32"
# os.mkdir(f"Master_Thesis_Code/LTC_A2C/adaptation_module/training_results/{results_dir}")
print("CURRENTLY NOT SAVING RESULTS!!!!!!!!!!!!!")


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


for i, w in enumerate(weights):
    if neuron_type == "CfC":
        agent_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims)
        w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))    
    elif neuron_type == "LTC":
        raise NotImplementedError
    
    agent_net.load_state_dict(w)

    privileged_encoder = agent_net.cfc_model.rnn_cell.get_neuromodulation_network()
    for param in privileged_encoder.named_parameters():
        param.requires_grad = False

    adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptation, num_neurons_policy, seed = seed)

    agent_net.cfc_model.rnn_cell.set_neuromodulation_network(adaptation_module)

    agent_net.cfc_model.rnn_cell.freeze_non_neuromodulation_parameters()

    train_adaptation_module(env, num_training_eps, 200, agent_net, num_actions, evaluation_seeds, i, neuron_type, privileged_encoder)


    
