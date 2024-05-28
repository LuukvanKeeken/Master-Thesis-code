import gym
import numpy as np
import torch
from Master_Thesis_Code.backpropamine_A2C import Standard_RNetwork



def main():

    env = gym.make('CartPole-v0')

    pole_length_multipliers = [0.775, 5.75, 0.55, 10.5, 0.1, 20.0]
    pole_mass_multipliers = [1.0, 2.0, 2.0, 3.0, 5.0, 13.0]
    force_magnitude_multipliers = [0.8, 2.25, 0.6, 3.5, 0.2, 6.0]

    default_length = 0.5
    default_mass = 0.1
    default_force = 10.0

    
    num_episodes = 2000

    policy_weights = torch.load("Master_Thesis_Code/BP_A2C/training_results/Standard_RNN_a2c_result_99994999_2024522_entropycoef_0.0_valuepredcoef_1.0_learningrate_0.001_numtrainepisodes_40000_selectionmethod_evaluation_trainingmethod_original_numneurons_32/checkpoint_BP_A2C_0.pt")
    agent_net = Standard_RNetwork(4, 32, 2, 5)
    agent_net.load_state_dict(policy_weights)



    pole_length_means = []
    pole_length_stds = []
    pole_mass_means = []
    pole_mass_stds = []
    force_magnitude_means = []
    force_magnitude_stds = []
    for multiplier in pole_length_multipliers:
        all_rewards = []
        env.unwrapped.length = default_length * multiplier
        for i in range(num_episodes):
            print(f'\rEpisode: {i+1}/{num_episodes}', end='')
            hebbian_traces = agent_net.initialZeroHebb(1)
            hidden_activations = agent_net.initialZeroState(1)
            env.seed(i)
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)
                policy_output, _, (hidden_activations, hebbian_traces) = agent_net(state.float(), (hidden_activations, hebbian_traces))
                policy_dist = torch.softmax(policy_output, dim=1)
                action = torch.argmax(policy_dist)
                state, reward, done, _ = env.step(action.item())
                total_reward += reward
            all_rewards.append(total_reward)
        pole_length_means.append(np.mean(all_rewards))
        pole_length_stds.append(np.std(all_rewards))

    env.unwrapped.length = default_length

    for multiplier in pole_mass_multipliers:
        all_rewards = []
        env.unwrapped.masspole = default_mass * multiplier
        for i in range(num_episodes):
            print(f'\rEpisode: {i+1}/{num_episodes}', end='')
            hebbian_traces = agent_net.initialZeroHebb(1)
            hidden_activations = agent_net.initialZeroState(1)
            env.seed(i)
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)
                policy_output, _, (hidden_activations, hebbian_traces) = agent_net(state.float(), (hidden_activations, hebbian_traces))
                policy_dist = torch.softmax(policy_output, dim=1)
                action = torch.argmax(policy_dist)
                state, reward, done, _ = env.step(action.item())
                total_reward += reward
            all_rewards.append(total_reward)
        pole_mass_means.append(np.mean(all_rewards))
        pole_mass_stds.append(np.std(all_rewards))
    
    env.unwrapped.masspole = default_mass

    for multiplier in force_magnitude_multipliers:
        all_rewards = []
        env.unwrapped.force_mag = default_force * multiplier
        for i in range(num_episodes):
            hebbian_traces = agent_net.initialZeroHebb(1)
            hidden_activations = agent_net.initialZeroState(1)
            env.seed(i)
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)
                policy_output, _, (hidden_activations, hebbian_traces) = agent_net(state.float(), (hidden_activations, hebbian_traces))
                policy_dist = torch.softmax(policy_output, dim=1)
                action = torch.argmax(policy_dist)
                state, reward, done, _ = env.step(action.item())
                total_reward += reward
            all_rewards.append(total_reward)
        force_magnitude_means.append(np.mean(all_rewards))
        force_magnitude_stds.append(np.std(all_rewards))

    print(pole_length_means)
    print(pole_length_stds)
    print(pole_mass_means)
    print(pole_mass_stds)
    print(force_magnitude_means)
    print(force_magnitude_stds)
    test = 1

    



if __name__ == "__main__":
    main()