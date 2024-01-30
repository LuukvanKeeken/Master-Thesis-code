import numpy as np
from LTC_A2C import LTC_Network, CfC_Network
import torch
import gym


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            policy_output, value, hidden_state = agent_net(state.float(), hidden_state)
            
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

evaluation_seeds = np.load('rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')


weights_0 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_0.pt', map_location=torch.device(device))
weights_1 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_1.pt', map_location=torch.device(device))
weights_2 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_2.pt', map_location=torch.device(device))
weights_3 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_3.pt', map_location=torch.device(device))
weights_4 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_4.pt', map_location=torch.device(device))
weights_5 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_5.pt', map_location=torch.device(device))
weights_6 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_6.pt', map_location=torch.device(device))
weights_7 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_7.pt', map_location=torch.device(device))
weights_8 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_8.pt', map_location=torch.device(device))
weights_9 = torch.load('LTC_A2C/training_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/checkpoint_CfC_A2C_9.pt', map_location=torch.device(device))
weights = [weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9]

eraser = '\b \b'


original_eval_rewards = []
for i, w in enumerate(weights):
    print(f'Run #{i}', end='')
    agent_net = CfC_Network(4, 64, 2, 5).to(device)
    agent_net.load_state_dict(w)

    rewards = evaluate_LTC_agent_pole_length(agent_net, env_name, n_evaluations, evaluation_seeds, 1.0)
    original_eval_rewards.append(rewards)
    print(eraser*3 + '-> Avg reward: {:7.2f}'.format(np.mean(rewards)))

print(f"Mean avg reward: {np.mean(original_eval_rewards)}")


