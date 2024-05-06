import numpy as np


nums_neurons = [48]
learning_rates = [0.001, 0.0001, 0.00001]
neuromod_nets = [[3, 256, 128], [3, 192, 96], [3, 128, 80]]
entropy_coefs = [0.0, 0.0001, 0.01, 1.0]
value_coefs = [0.0001, 0.01, 0.5, 1.0]
activation_functions = ["tanh"]


# types = ["BP", "StandardRNN", "StandardMLP"]

neuron_type = "CfC"
num_models = 10
all_results = []
results_id = 46000
date = 202453

# for neuron_type in types:
for num_neurons in nums_neurons:
    for learning_rate in learning_rates:
        for neuromodnet in neuromod_nets:
            for entropy_coef in entropy_coefs:
                for value_coef in value_coefs:
                    for func in activation_functions:
                        

                        if neuron_type == "CfC":
                            results_dir = f"CfC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_numneurons_{num_neurons}_encoutact_{func}_neuromod_network_dims_{'_'.join(map(str, neuromodnet))}_{num_neurons}"
                        elif neuron_type == "LTC":
                            # results_dir = f'LTC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_{num_neurons}_tausysextraction_True'
                            results_dir = f'LTC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_selectiomethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}_tausysextraction_True'
                        elif neuron_type == "BP":
                            # BP_a2c_result_42000_202452_learningrate_0.001_numneurons_48_encoutact_relu_neuromod_network_dims_3_256_128_48
                            results_dir = f"BP_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_numneurons_{num_neurons}_encoutact_{func}_neuromod_network_dims_{'_'.join(map(str, neuromodnet))}_{num_neurons}"

                        elif neuron_type == "StandardRNN":
                            results_dir = f"Standard_RNN_a2c_result_{results_id}_{date}_entropycoef_0.01_valuepredcoef_0.1_learningrate_{learning_rate}_numtrainepisodes_20000_selectionmethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}"
                        elif neuron_type == "StandardMLP":
                            results_dir = f"Standard_MLP_a2c_result_{results_id}_{date}_entropycoef_0.01_valuepredcoef_0.1_learningrate_{learning_rate}_numtrainepisodes_20000_selectionmethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}"


                        try:
                            with open(f"Master_Thesis_Code/LTC_A2C/7_NMCfC_tanh_48/{results_dir}/best_average_after.txt", "r") as file:
                                for i in range(num_models):
                                    file.readline()

                                file.readline()
                                last_line = file.readline()
                                last_line = last_line.split(" ")
                                all_results.append((results_id, num_neurons, learning_rate, neuromodnet, func, entropy_coef, value_coef, float(last_line[3].strip(',')), float(last_line[6])))
                                # all_results.append((results_id, neuron_type, num_neurons, learning_rate, float(last_line[3].strip(',')), float(last_line[6])))
                        except:
                            print(f"Could not find {results_dir}")

                        results_id += 1

print(all_results)