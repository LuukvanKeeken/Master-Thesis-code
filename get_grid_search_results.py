import os
import numpy as np

nums_neurons = [48]
types = ["Standard_MLP"]
learning_rates = [0.001, 0.00005, 0.000001]
entropy_coefs = [0.0, 0.0001, 0.01, 1.0]
value_coefs = [0.0001, 0.01, 0.5, 1.0]



training_method = "quarter_range"
num_models = 10
all_results = []
results_id = 44000
date = 202453
num_train_eps = 25000
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
                        # results_dir = f'LTC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_{num_neurons}_tausysextraction_True'
                        results_dir = f'LTC_a2c_result_{results_id}_{date}_learningrate_{learning_rate}_selectiomethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}_tausysextraction_True'
                    elif neuron_type == "BP_RNN":
                        results_dir = f'BP_RNN_a2c_result_{results_id}_{date}_entropycoef_{entropy_coef}_valuepredcoef_{value_coef}_learningrate_{learning_rate}_numtrainepisodes_{num_train_eps}_selectionmethod_true_range_eval_all_params_trainingmethod_{training_method}_numneurons_{num_neurons}'
                    elif neuron_type == "Standard_RNN":
                        results_dir = f"Standard_RNN_a2c_result_{results_id}_{date}_entropycoef_{entropy_coef}_valuepredcoef_{value_coef}_learningrate_{learning_rate}_numtrainepisodes_{num_train_eps}_selectionmethod_true_range_eval_all_params_trainingmethod_original_numneurons_{num_neurons}"
                    elif neuron_type == "Standard_MLP":
                        results_dir = f"Standard_MLP_a2c_result_{results_id}_{date}_entropycoef_{entropy_coef}_valuepredcoef_{value_coef}_learningrate_{learning_rate}_numtrainepisodes_{num_train_eps}_selectionmethod_true_range_eval_all_params_trainingmethod_{training_method}_numneurons_{num_neurons}"

                    try:
                        with open(f"Master_Thesis_Code/{top_dir}/5_MLP_range_48/{results_dir}/best_average_after.txt", "r") as file:
                            for i in range(num_models):
                                file.readline()

                            file.readline()
                            last_line = file.readline()
                            last_line = last_line.split(" ")
                            # all_results.append((results_id, neuron_type, num_neurons, learning_rate, float(last_line[3].strip(',')), float(last_line[6])))
                            all_results.append((results_id, neuron_type, num_neurons, learning_rate, entropy_coef, value_coef, float(last_line[3].strip(',')), float(last_line[6])))
                            fine_ids.append(results_id)
                    except:
                        files = os.listdir(f"Master_Thesis_Code/{top_dir}/5_MLP_range_48/{results_dir}")
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