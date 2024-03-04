import numpy as np

nums_neurons = [64, 48, 32]
neuron_types = ['BP_RNN', 'Standard_RNN', 'Standard_MLP']
learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
num_models = 10
all_results = []
results_id = 21
date = 202434

for neuron_type in neuron_types:
    for num_neurons in nums_neurons:
        for learning_rate in learning_rates:

            results_dir = f'{neuron_type}_a2c_result_{results_id}_{date}_entropycoef_0.01_valuepredcoef_0.1_batchsize_1_maxsteps_200_maxgradnorm_4.0_gammaR_0.99_learningrate_{learning_rate}_numtrainepisodes_20000_selectionmethod_range_evaluation_all_params_trainingmethod_original'

            with open(f"Master_Thesis_Code/BP_A2C/training_results/{results_dir}/best_average_after.txt", "r") as file:
                for i in range(num_models):
                    file.readline()

                last_line = file.readline()
                last_line = last_line.split(" ")
                all_results.append((results_id, neuron_type, num_neurons, learning_rate, last_line[9], last_line[12]))
                

            results_id += 1

print(all_results)