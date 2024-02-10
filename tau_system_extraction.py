import numpy as np
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
import os
import torch
import gym
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_LTC_agent_and_extract_tau_sys_pole_length(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier):

    eval_rewards = []
    all_tau_sys_all_episodes = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
        
    for i_episode in range(num_episodes):
        all_tau_sys = []
        hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            policy_output, value, hidden_state, tau_sys = agent_net(state.float(), hidden_state)

            all_tau_sys.append(tau_sys)
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)
        all_tau_sys_all_episodes.append(all_tau_sys)

    return eval_rewards, all_tau_sys_all_episodes


neuron_type = "CfC"
mode = "pure"
num_neurons = 32
eval_pole_length_modifiers = [1.0, 6.0, 11.0, 16.0, 21.0]
num_evaluation_episodes = 100

evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')


results_dir = f"CfC_a2c_result_10_202428_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_fullyconnected_True_mode_pure"
os.mkdir(f"Master_Thesis_Code/LTC_A2C/time_constants/{results_dir}")

weights = torch.load(f"Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_0.pt", map_location=torch.device("cpu"))

if neuron_type == "CfC":
    agent_net = CfC_Network(4, num_neurons, 2, 5, extract_tau_sys=True, mode = mode).to(device)
elif neuron_type == "LTC":
    agent_net = LTC_Network(4, num_neurons, 2, 5, extract_tau_sys=True).to(device)


weights["cfc_model.rnn_cell.tau_system"] = torch.reshape(weights["cfc_model.rnn_cell.tau_system"], (32,))
agent_net.load_state_dict(weights)

all_tau_sys = []
for pole_length_mod in eval_pole_length_modifiers:
    print(f"Pole length mod: {pole_length_mod}")

    rewards, tau_sys = evaluate_LTC_agent_and_extract_tau_sys_pole_length(agent_net, "CartPole-v0", num_evaluation_episodes, evaluation_seeds, pole_length_mod)
    print(f"Mean performance: {np.mean(rewards)}")
    all_tau_sys.append(tau_sys)

all_results = {"pole_length_mods": eval_pole_length_modifiers, "all_tau_sys": all_tau_sys}

with open(f'Master_Thesis_Code/LTC_A2C/time_constants/{results_dir}/all_tau_sys.pkl', 'wb') as f:
    pickle.dump(all_results, f)



