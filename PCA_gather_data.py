import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
import numpy as np
import gym




device = 'cpu'
neuron_type = 'CfC'
num_neurons = 32
seed = 5
mode = 'neuromodulated'
wiring = None
neuromod_network_dims = [3, 256, 128, 32]
# results_dir = f"CfC_a2c_result_157_2024225_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated"
# results_dir = f"CfC_a2c_result_158_2024225_learningrate_0.0005_selectiomethod_range_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.2, 0.2, 0.2]"
# results_dir = f"CfC_a2c_result_161_2024225_learningrate_0.0005_selectiomethod_range_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.35, 0.35, 0.35]"
# results_dir = f"CfC_a2c_result_164_2024225_learningrate_0.0005_selectiomethod_range_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.5, 0.5, 0.5]"
# results_dir = f"CfC_a2c_result_117_2024216_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated"
# results_dir = f"CfC_a2c_result_173_2024227_learningrate_0.0005_selectiomethod_range_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated"
results_dir = f"CfC_a2c_result_157_2024225_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated"
# results_dir = f"CfC_a2c_result_174_2024227_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated"

weights = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))


agent_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring, neuromod_network_dims=neuromod_network_dims).to(device)
weights['cfc_model.rnn_cell.tau_system'] = torch.reshape(weights['cfc_model.rnn_cell.tau_system'], (num_neurons,))

agent_net.load_state_dict(weights)

# pole_length_mods = np.linspace(0.1, 20.0, 500)
# pole_mass_mods = np.linspace(5.0, 20.0, 500)
force_mag_mods = np.linspace(0.2, 6.0, 500)
all_output_vectors = []
for fm_mods in force_mag_mods:
    fm_val = 10.0*fm_mods
    env_info = torch.tensor([0.5, 0.1, fm_val], dtype=torch.float32)
    encoder_output = agent_net.cfc_model.rnn_cell.neuromod(env_info)
    all_output_vectors.append(encoder_output.detach().numpy())

np.save(f'Master_Thesis_Code/encoder_outputs_force_mag_UNtrainedencodernotonrange.npy', np.array(all_output_vectors))
