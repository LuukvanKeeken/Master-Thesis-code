import time
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
import gym
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


neuron_type = "CfC"
mode = "pure"
num_neurons = 32
episode_idx = 70
pole_length_mod = 11.0
pole_mass_mod = 1.0
force_mag_mod = 1.0

evaluation_seeds = np.load('Master_Thesis_Code/rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')

results_dir = f"CfC_a2c_result_10_202428_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_fullyconnected_True_mode_pure"
weights = torch.load(f"Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_0.pt", map_location=torch.device("cpu"))

if neuron_type == "CfC":
    agent_net = CfC_Network(4, num_neurons, 2, 5, extract_tau_sys=True, mode = mode).to(device)
elif neuron_type == "LTC":
    agent_net = LTC_Network(4, num_neurons, 2, 5, extract_tau_sys=True).to(device)


weights["cfc_model.rnn_cell.tau_system"] = torch.reshape(weights["cfc_model.rnn_cell.tau_system"], (32,))
agent_net.load_state_dict(weights)


env = gym.make('CartPole-v0')
env.unwrapped.length *= pole_length_mod
env.unwrapped.masspole *= pole_mass_mod
env.unwrapped.force_mag *= force_mag_mod

while True:
    env.seed(int(evaluation_seeds[episode_idx]))
    

    state = env.reset()

    done = False
    hidden_state = None
    step_num = 0
    while not done:
        step_num += 1
        print(f"Step: {step_num:<3}", end="\r")
        env.render()
        time.sleep(0.1)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0).to(device) #This as well?
        policy_output, value, hidden_state, tau_sys = agent_net(state.float(), hidden_state)
        
        policy_dist = torch.softmax(policy_output, dim = 1)
        
        action = torch.argmax(policy_dist)
        

        state, r, done, _ = env.step(action.item())