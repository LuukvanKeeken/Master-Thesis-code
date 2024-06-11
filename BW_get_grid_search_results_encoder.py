import numpy as np



nums_neurons = [96]
learning_rates = [0.00001, 0.000001]
neuromod_nets = [[11, 256, 128]]
entropy_coefs = [0.001, 0.01, 1.0]
value_coefs = [0.001, 0.01, 1.0]
activation_functions = ["relu"]




# types = ["BP", "StandardRNN", "StandardMLP"]
directory = "bipedal_walker"
neuron_type = "CfC"
if neuron_type == "CfC":
    top_dir = "LTC_A2C"
else:
    top_dir = "BP_A2C"
num_models = 1
all_results = []
results_id = 800000
date = 202465
fine_ids = []
not_fine_ids = []

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
                            results_dir = f'LTC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_selectiomethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}_tausysextraction_True'
                        elif neuron_type == "BP":
                            results_dir = f"BP_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_numneurons_{num_neurons}_encoutact_{func}_neuromod_network_dims_{'_'.join(map(str, neuromodnet))}_{num_neurons}"

                        elif neuron_type == "StandardRNN":
                            results_dir = f"Standard_RNN_a2c_result_{results_id}_{date}_entropycoef_0.01_valuepredcoef_0.1_learningrate_{learning_rate}_numtrainepisodes_20000_selectionmethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}"
                        elif neuron_type == "StandardMLP":
                            results_dir = f"Standard_MLP_a2c_result_{results_id}_{date}_entropycoef_0.01_valuepredcoef_0.1_learningrate_{learning_rate}_numtrainepisodes_20000_selectionmethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}"


                        try:
                            with open(f"Master_Thesis_Code/{top_dir}/{directory}/{results_dir}/best_average_after.txt", "r") as file:
                                for i in range(num_models):
                                    info = file.readline()
                                    info = info.split(":")
                                    total_eps = int(info[2].split(")")[0].strip())

                                num_eps_line = file.readline()
                                num_eps_line = num_eps_line.split(" ")
                                num_eps = float(num_eps_line[3].strip(','))
                                last_line = file.readline()
                                last_line = last_line.split(" ")
                                all_results.append((results_id, num_neurons, learning_rate, neuromodnet, func, entropy_coef, value_coef, float(last_line[3].strip(',')), float(last_line[6]), num_eps, total_eps))
                                # all_results.append((results_id, neuron_type, num_neurons, learning_rate, float(last_line[3].strip(',')), float(last_line[6])))
                                fine_ids.append(results_id)
                        except:
                            print(f"Could not find {results_dir}")
                            not_fine_ids.append(results_id)

                        results_id += 1

print(all_results)
print(f"Fine ids: {fine_ids}")
print(f"Not fine ids: {not_fine_ids}")