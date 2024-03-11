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





parser = argparse.ArgumentParser(description='Train adaptation module for neuromodulated CfC')
parser.add_argument('--neuron_type', type=str, default='CfC', help='Type of neuron to train')
parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
parser.add_argument('--state_dims', type=int, default=4, help='Number of state dimensions')
parser.add_argument('--num_neurons_policy', type=int, default=32, help='Number of neurons in the policy network')
parser.add_argument('--num_neurons_adaptation', type=int, default=64, help='Number of neurons in the adaptation module')
parser.add_argument('--num_actions', type=int, default=2, help='Number of actions')
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--mode', type=str, default='neuromodulated', help='Mode of the CfC network')
parser.add_argument('--wiring', type=str, default='None', help='Wiring of the CfC network')
parser.add_argument('--neuromod_network_dims', type=int, nargs='+', default = [3, 192, 96], help='Dimensions of the neuromodulation network, without output layer')



args = parser.parse_args()
neuron_type = args.neuron_type
device = args.device
state_dims = args.state_dims
num_neurons_policy = args.num_neurons_policy
num_neurons_adaptation = args.num_neurons_adaptation
num_actions = args.num_actions
seed = args.seed
mode = args.mode
if args.wiring == 'None':
    wiring = None
neuromod_network_dims = args.neuromod_network_dims
neuromod_network_dims.append(num_neurons_policy)



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

    adaptation_module = StandardRNN(state_dims + num_actions, num_neurons_adaptation, num_neurons_policy, seed = seed)

    agent_net.cfc_model.rnn_cell.set_neuromodulation_network(adaptation_module)

    agent_net.cfc_model.rnn_cell.freeze_non_neuromodulation_parameters()
    

    
