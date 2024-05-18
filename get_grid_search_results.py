import os
import numpy as np

nums_neurons = [32, 48, 64]
types = ["CfC"]
learning_rates = [0.001, 0.0001, 0.00001,  0.000001]
entropy_coefs = [0.0, 0.0001, 0.01, 1.0]
value_coefs = [0.0001, 0.01, 0.5, 1.0]



directory = "33_CfC_range_32and48and64_correctEntropies"
training_method = "original"
selection_method = "true_range_eval_all_params"
num_models = 1
all_results = []
results_id = 72000
date = 2024517
num_train_eps = 40000
missing_ids = []
fine_ids = []
for neuron_type in types:
    for num_neurons in nums_neurons:
        for learning_rate in learning_rates:
            for entropy_coef in entropy_coefs:
                for value_coef in value_coefs:

                    if neuron_type == "CfC" or neuron_type == "LTC":
                        top_dir = "LTC_A2C"
                    else:
                        top_dir = "BP_A2C"    


                    if neuron_type == "CfC":
                        results_dir = f'CfC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_selectiomethod_true_range_eval_all_params_trainingmethod_{training_method}_numneurons_{num_neurons}_mode_pure'
                    elif neuron_type == "LTC":
                        results_dir = f'LTC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_selectiomethod_true_range_eval_all_params_trainingmethod_{training_method}_numneurons_{num_neurons}'
                    elif neuron_type == "BP_RNN":
                        results_dir = f'BP_RNN_a2c_result_{results_id}_{date}_entropycoef_{entropy_coef}_valuepredcoef_{value_coef}_learningrate_{learning_rate}_numtrainepisodes_{num_train_eps}_selectionmethod_{selection_method}_trainingmethod_{training_method}_numneurons_{num_neurons}'
                    elif neuron_type == "Standard_RNN":
                        results_dir = f'Standard_RNN_a2c_result_{results_id}_{date}_entropycoef_{entropy_coef}_valuepredcoef_{value_coef}_learningrate_{learning_rate}_numtrainepisodes_{num_train_eps}_selectionmethod_{selection_method}_trainingmethod_{training_method}_numneurons_{num_neurons}'
                    elif neuron_type == "Standard_MLP":
                        results_dir = f"Standard_MLP_a2c_result_{results_id}_{date}_entropycoef_{entropy_coef}_valuepredcoef_{value_coef}_learningrate_{learning_rate}_numtrainepisodes_{num_train_eps}_selectionmethod_true_range_eval_all_params_trainingmethod_{training_method}_numneurons_{num_neurons}"

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
                            all_results.append((results_id, neuron_type, num_neurons, learning_rate, entropy_coef, value_coef, float(last_line[3].strip(',')), float(last_line[6]), num_eps, total_eps))
                            fine_ids.append(results_id)
                    except:
                        files = os.listdir(f"Master_Thesis_Code/{top_dir}/{directory}/{results_dir}")
                        num_files = len(files)
                        if num_files <= 5:
                            print(f"Could not find {results_dir}, {num_files} models")
                        else:
                            print(f"Could not find {results_dir}, {num_files - 5} models")

                        missing_ids.append(results_id)


                    results_id += 1


print(all_results)
print(f"Missing IDs: {missing_ids}")
print(f"Fine IDs: {fine_ids}")