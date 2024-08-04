import os
import numpy as np

nums_neurons = [96]
learning_rates = [0.01, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000001]
wd_terms = [0.0, 0.0001, 0.005, 0.01]



neuron_type = "CfC"
if neuron_type == "CfC":
    top_dir = "LTC_A2C"
else:
    top_dir = "BP_A2C"

num_models = 1
all_results = []
results_id = 5100000
date = 2024721
failed_idx = []

for num_neurons in nums_neurons:
    for learning_rate in learning_rates:
        for wd in wd_terms:
            
            if neuron_type == "CfC":
                results_dir = f"adaptation_module_StandardRNN_result_{results_id}_{date}_BP_a2c_result_2169_202448_numneuronsadaptmod_{num_neurons}_lradaptmod_{learning_rate}_wdadaptmod_{wd}"
            else:
                # adaptation_module_StandardRNN_result_2000_2024412_BP_a2c_result_2169_202448_numneuronsadaptmod_64_lradaptmod_0.001_wdadaptmod_0.0
                results_dir = f'adaptation_module_StandardRNN_result_{results_id}_{date}_BP_a2c_result_2169_202448_numneuronsadaptmod_{num_neurons}_lradaptmod_{learning_rate}_wdadaptmod_{wd}'
            
            try:
                with open(f"Master_Thesis_Code/{top_dir}/bipedal_walker/adaptation_module/training_results/{results_dir}/best_validation_loss_after.txt", "r") as file:
                    for i in range(num_models):
                        file.readline()

                    file.readline()
                    last_line = file.readline()
                    last_line = last_line.split(" ")
                    all_results.append((results_id, num_neurons, learning_rate, wd, float(last_line[3].strip(',')), float(last_line[5])))#, float(last_line[6])))
                    
                
                with open(f"Master_Thesis_Code/{top_dir}/bipedal_walker/adaptation_module/training_results/{results_dir}/best_validation_reward_after.txt", "r") as file:
                    for i in range(num_models):
                        file.readline()

                    file.readline()
                    last_line = file.readline()
                    last_line = last_line.split(" ")
                    all_results[-1] = all_results[-1] + (float(last_line[3].strip(',')), float(last_line[5]))
            except:
                num_files = len(os.listdir(f"Master_Thesis_Code/{top_dir}/bipedal_walker/adaptation_module/training_results/{results_dir}"))
                print(f"lr: {learning_rate}, wd: {wd}, num_neurons: {num_neurons} not found. {results_id} Models trained: {num_files/2}")
                failed_idx.append(results_id)
            
            
            results_id += 1

print(f"Good results: {all_results}")
print(f"Failed results: {failed_idx}")
print(len(failed_idx))