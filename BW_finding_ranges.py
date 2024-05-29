import gym
import numpy as np
import torch
from Master_Thesis_Code.backpropamine_A2C import BP_RNetwork, Standard_RNetwork

gym.envs.registration.register(
    id='AdjustableBipedalWalker-v3',
    entry_point='Master_Thesis_Code.AdjustableBipedalWalker:AdjustableBipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)

def main():
    with torch.no_grad():
        env = gym.make('AdjustableBipedalWalker-v3')
        state = env.reset()

        # policy_weights = torch.load("Master_Thesis_Code/BP_A2C/bipedal_walker/Continued_Standard_RNN_a2c_result_68024_202459_entropycoef_0.0001_valuepredcoef_0.0001_learningrate_1e-05_numtrainepisodes_1000000_selectionmethod_exp_BW_validation_trainingmethod_original_numneurons_96/checkpoint_BP_A2C_0.pt")
        # policy_weights = torch.load("Master_Thesis_Code/BP_A2C/29_BW_BPandRNN_original_96/Standard_RNN_a2c_result_68024_202459_entropycoef_0.0001_valuepredcoef_0.0001_learningrate_1e-05_numtrainepisodes_1000000_selectionmethod_exp_BW_validation_trainingmethod_original_numneurons_96/checkpoint_BP_A2C_0.pt")
        policy_weights = torch.load("Master_Thesis_Code/BP_A2C/29_BW_BPandRNN_original_96/BP_RNN_a2c_result_98_2024516_entropycoef_0.0001_valuepredcoef_0.0001_learningrate_1e-05_numtrainepisodes_1000000_selectionmethod_exp_BW_validation_trainingmethod_original_numneurons_96/checkpoint_BP_A2C_0.pt")

        # agent_net = Standard_RNetwork(24, 96, 4, 5, continuous_actions=True)
        agent_net = BP_RNetwork(24, 96, 4, 5, continuous_actions=True)
        agent_net.load_state_dict(policy_weights)
        
        # variables = ['left_leg_w_unscaled', 'left_leg_h_unscaled', 'right_leg_w_unscaled', 'right_leg_h_unscaled', 'terrain_friction', 'speed_hip', 'speed_knee', 'left_leg_density', 'right_leg_density', 'hull_density', 'hull_friction', 'lidar_range_unscaled']
        variables = ['left_leg_w_unscaled', 'left_leg_h_unscaled', 'right_leg_w_unscaled', 'right_leg_h_unscaled', 'terrain_friction', 'speed_hip', 'speed_knee', 'left_leg_density', 'right_leg_density', 'hull_density', 'lidar_range_unscaled']        
        # default_values = [8.0, 34.0, 8.0, 34.0, 2.5, 4.0, 6.0, 1.0, 1.0, 5.0, 0.1, 160.0]
        default_values = [8.0, 34.0, 8.0, 34.0, 2.5, 4.0, 6.0, 1.0, 1.0, 5.0, 160.0]
        percentages_llwu = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.5, 2.75, 3.0]
        percentages_llhu = [0.8125, 0.825, 0.8375, 0.85, 0.8625, 0.875, 0.8875, 0.9, 1.1, 1.125, 1.15, 1.175]
        percentages_rlwu = [0.125, 0.15, 0.175, 0.2, 1.5, 1.625, 1.75, 1.875, 2.0]
        percentages_rlhu = [0.93, 0.935, 0.94, 0.945, 0.95, 1.5, 1.625, 1.75, 1.875, 2.0]
        percentages_tf = [0.225, 0.25, 0.275, 0.3, 1.3, 1.325, 1.35, 1.375, 1.4, 1.425, 1.45, 1.475, 1.5]
        percentages_sh = [0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 1.5, 1.625, 1.75, 1.875, 2.0]
        percentages_sk = [0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 1.2, 1.225, 1.25, 1.275, 1.3]
        percentages_lld = [0.0625, 0.075, 0.0875, 0.1, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.5, 2.75, 3.0]
        percentages_rld = [0.0625, 0.075, 0.0875, 0.1, 1.3, 1.325, 1.35, 1.375, 1.4, 1.425, 1.45, 1.475]
        percentages_hd = [0.8125, 0.825, 0.8375, 0.85, 0.8625, 0.875, 0.8875, 0.9, 1.3, 1.325, 1.35, 1.375, 1.4, 1.425, 1.45, 1.475, 1.5]
        percentages_lru = [0.7125, 0.725, 0.7375, 0.75, 0.7625, 0.775, 0.7875, 0.8, 1.1, 1.125, 1.15, 1.175]
        all_percentages = [percentages_llwu, percentages_llhu, percentages_rlwu, percentages_rlhu, percentages_tf, percentages_sh, percentages_sk, percentages_lld, percentages_rld, percentages_hd, percentages_lru]
        
        all_results_means = []
        all_results_stds = []
        num_episodes = 150
        for variable, default_value, percentages in zip(variables, default_values, all_percentages):
            with open(f"Master_Thesis_Code/current_variable.txt", 'w') as f:
                f.write(f"{variable}")

            print(f"{variable}: {default_value}")
            all_means = []
            all_stds = []
            for percentage in percentages:
                print(f"{percentage}")
                setattr(env.unwrapped, variable, default_value * percentage)
                all_rewards = []
                for i in range(num_episodes):
                    env.seed(i)
                    state = env.reset()
                    done = False
                    hebbian_traces = agent_net.initialZeroHebb(1)
                    hidden_activations = agent_net.initialZeroState(1)
                    rewards_sum = 0
                    while not done:
                        # env.render()
                        state = torch.from_numpy(state)
                        state = state.unsqueeze(0)
                        policy_output, _, (hidden_activations, hebbian_traces) = agent_net.forward(state.float(), [hidden_activations, hebbian_traces])
                        means, _ = policy_output
                        action = means.detach()
                        state, r, done, _ = env.step(action[0].numpy())
                        rewards_sum += r
                        if done:
                            print(f"Episode {i} finished with reward {rewards_sum}")
                            all_rewards.append(rewards_sum)
                            env.reset()

                all_means.append(np.mean(all_rewards))
                all_stds.append(np.std(all_rewards))
                print(f"Mean: {np.mean(all_rewards)} +/- {np.std(all_rewards)}")
            
            all_results_means.append(all_means)
            all_results_stds.append(all_stds)
            setattr(env.unwrapped, variable, default_value)
            # np.save("Master_Thesis_Code/BW_finding_ranges_means_BP_detailed.npy", all_results_means, allow_pickle=True)
            # np.save("Master_Thesis_Code/BW_finding_ranges_stds_BP_detailed.npy", all_results_stds, allow_pickle=True)
        
        print(all_results_means)
        print(all_results_stds)
        

        env.close()
        
        
        
        
        
        
        # percentages = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 1.01, 1.025, 1.05, 1.075, 1.1, 1.2, 1.3, 1.5, 2.0, 3.5, 5.0, 10.0, 15.0, 20.0]
        
        # all_results_means = []
        # all_results_stds = []
        # num_episodes = 100
        # for variable, default_value in zip(variables, default_values):
        #     with open(f"Master_Thesis_Code/current_variable.txt", 'w') as f:
        #         f.write(f"{variable}")

        #     print(f"{variable}: {default_value}")

        #     all_means = []
        #     all_stds = []
        #     for percentage in percentages:
        #         print(f"{percentage}")
        #         setattr(env.unwrapped, variable, default_value * percentage)
        #         all_rewards = []
        #         for i in range(num_episodes):
        #             env.seed(i)
        #             state = env.reset()
        #             done = False
        #             hebbian_traces = agent_net.initialZeroHebb(1)
        #             hidden_activations = agent_net.initialZeroState(1)
        #             rewards_sum = 0
        #             while not done:
        #                 # env.render()
        #                 state = torch.from_numpy(state)
        #                 state = state.unsqueeze(0)
        #                 policy_output, _, (hidden_activations, hebbian_traces) = agent_net.forward(state.float(), [hidden_activations, hebbian_traces])
        #                 means, _ = policy_output
        #                 action = means.detach()
        #                 state, r, done, _ = env.step(action[0].numpy())
        #                 rewards_sum += r
        #                 if done:
        #                     print(f"Episode {i} finished with reward {rewards_sum}")
        #                     all_rewards.append(rewards_sum)
        #                     env.reset()

        #         all_means.append(np.mean(all_rewards))
        #         all_stds.append(np.std(all_rewards))
        #         print(f"Mean: {np.mean(all_rewards)} +/- {np.std(all_rewards)}")
            
        #     all_results_means.append(all_means)
        #     all_results_stds.append(all_stds)
        #     setattr(env.unwrapped, variable, default_value)
        #     np.save("Master_Thesis_Code/BW_finding_ranges_means.npy", all_results_means)
        #     np.save("Master_Thesis_Code/BW_finding_ranges_stds.npy", all_results_stds)
        
        # print(all_results_means)
        # print(all_results_stds)
        

        # env.close()

if __name__ == "__main__":
    main()