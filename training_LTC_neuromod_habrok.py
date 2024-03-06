from datetime import date
import os
import random
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import deque
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from ncps_time_constant_extraction.ncps.wirings import AutoNCP
import argparse




device = "cpu"


def evaluate_agent_pole_length(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier):

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

def randomize_env_params(env, randomization_params):
    pole_length = env.unwrapped.length
    # gravity = env.unwrapped.gravity
    # masscart = env.unwrapped.masscart
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
    # env.unwrapped.gravity = params[1]
    # env.unwrapped.masscart = params[2]
    env.unwrapped.masspole = params[1]
    env.unwrapped.force_mag = params[2]

    return env
    



def train_agent(env, num_training_episodes, max_steps, agent_net, num_outputs, evaluation_seeds, i_run, neuron_type, selection_method = "100 episode average", gamma = 0.99, max_reward = 200, env_name = "CartPole-v0", num_evaluation_episodes = 10, evaluate_every = 10, randomization_params = None, randomize_every = 5):
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



parser = argparse.ArgumentParser(description='Train an A2C agent on the CartPole environment')
parser.add_argument('--num_neurons', type=int, default=32, help='Number of neurons in the hidden layer')
parser.add_argument('--randomization_factor', type=float, default=0.5, help='Factor to randomize the environment parameters')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the agent')
parser.add_argument('--training_method', type=str, default = "quarter_range", help='Method to train the agent')
parser.add_argument('--neuromod_network_dims', type=int, nargs='+', default = [3, 256, 128], help='Dimensions of the neuromodulation network, without output layer')
parser.add_argument('--selection_method', type=str, default = "range_evaluation_all_params", help='Method to select the best model')
parser.add_argument('--num_models', type=int, default=10, help='Number of models to train')
parser.add_argument('--num_training_episodes', type=int, default=20000, help='Number of episodes to train the agent')
args = parser.parse_args()


num_neurons = args.num_neurons
factor = args.randomization_factor
learning_rate = args.learning_rate
training_method = args.training_method
selection_method = args.selection_method
neuromod_network_dims = args.neuromod_network_dims
neuromod_network_dims.append(num_neurons)
num_models = args.num_models
num_training_episodes = args.num_training_episodes

# print(f"Num neurons: {num_neurons}, sparsity level: {sparsity_level}, learning rate: {learning_rate}")
print(f"Num neurons: {num_neurons}, learning rate: {learning_rate}, rand factor: {factor}")
device = "cpu"

gamma = 0.99
# num_neurons = 32
neuron_type = "CfC"
mode = "neuromodulated"

if training_method == "quarter_range":
    randomization_params = [(0.775, 5.75), (1.0, 2.0), (0.8, 2.25)]
else:
    randomization_params = 3*[factor]



tau_sys_extraction = True
sparsity_level = 0.5
seed = 5
# wiring = AutoNCP(num_neurons, 3, sparsity_level=sparsity_level, seed=seed)
wiring = None

dirs = os.listdir('Master_Thesis_Code/LTC_A2C/training_results/')
if not any('a2c_result' in d for d in dirs):
    result_id = 1
else:
    results = [d for d in dirs if 'a2c_result' in d]
    result_id = len(results) + 1
d = date.today()
result_dir = f'Master_Thesis_Code/LTC_A2C/training_results/{neuron_type}_a2c_result_' + str(result_id) + f'_{str(d.year)+str(d.month)+str(d.day)}_learningrate_{learning_rate}_selectiomethod_{selection_method}_gamma_{gamma}_trainingmethod_{training_method}_numneurons_{num_neurons}_tausysextraction_{tau_sys_extraction}'
if neuron_type == "CfC":
    result_dir += "_mode_" + mode
    if mode == "neuromodulated":
        result_dir += "_neuromod_network_dims_" + "_".join(map(str, neuromod_network_dims))
if wiring:
    result_dir += "_wiring_" + "AutoNCP" + f"_sparsity_{sparsity_level}"
if randomization_params:
    result_dir += "_randomization_params_" + str(randomization_params)
os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))




env = gym.make('CartPole-v0')
evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')
training_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/training_seeds.npy')


best_average_after_all = []
best_average_all = []
for i in range(num_models):
    print(f"Run # {i}")
    seed = int(training_seeds[i])

    torch.manual_seed(seed)
    random.seed(seed)
    
    if neuron_type == "LTC":
        raise NotImplementedError
        agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
    elif neuron_type == "CfC":
        agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)

    optimizer = torch.optim.Adam(agent_net.parameters(), lr=learning_rate)

    smoothed_scores, scores, best_average, best_average_after = train_agent(env, num_training_episodes, 200, agent_net, 2, evaluation_seeds, i, neuron_type, selection_method = selection_method, gamma = gamma, randomization_params=randomization_params)
    best_average_after_all.append(best_average_after)
    best_average_all.append(best_average)


with open(f"{result_dir}/best_average_after.txt", 'w') as f:
    for i, best_episode in enumerate(best_average_after_all):
        f.write(f"{i}: {best_average_all[i]} after {best_episode}\n")

    f.write(f"Average training episodes: {np.mean(best_average_after_all)}, std dev: {np.std(best_average_after_all)}\n")
    f.write(f"Mean average performance: {np.mean(best_average_all)}, std dev: {np.std(best_average_all)}")

