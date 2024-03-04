import numpy as np

nums_neurons = [64, 48, 32]
neuron_types = ['CfC', 'LTC']
learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
num_models = 10
all_results = []
results_id = 185

for neuron_type in neuron_types:
    for num_neurons in nums_neurons:
        for learning_rate in learning_rates:

            results_dir = f"{neuron_type}_a2c_result_{results_id}_202431_learningrate_{learning_rate}_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_{num_neurons}_tausysextraction_True"
            if neuron_type == 'CfC':
                results_dir += "_mode_pure"


            with open(f"Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/best_average_after.txt", "a+") as file:
                file.seek(0)

                performances = []
                count = 0
                for line in file.readlines():
                    count += 1
                    if count <= num_models:
                        performances.append(float(line.split(" ")[1]))

                # file.write(f"\nMean average performance: {np.mean(performances)}, std: {np.std(performances)}\n")

                all_results.append((results_id, neuron_type, num_neurons, learning_rate, np.mean(performances), np.std(performances)))
                results_id += 1

print(all_results)