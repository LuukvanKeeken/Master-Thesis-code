import os

os.makedirs("Master_Thesis_Code/all_jobscripts/")

nums_neurons = [64, 48, 32]
network_types = ['Standard_MLP']
learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]

for network_type in network_types:
    for num_neurons in nums_neurons:
        for learning_rate in learning_rates:
            template = f"""#!/bin/bash
#SBATCH --time=10:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_BP_habrok --num_neurons {num_neurons} --network_type {network_type} --learning_rate {learning_rate}

deactivate"""

            with open(f"Master_Thesis_Code/all_jobscripts/jobscript_{network_type}_{num_neurons}neurons_{learning_rate}lr.sh", "w") as file:
                file.write(template)

with open("Master_Thesis_Code/all_jobscripts/all_jobs.sh", "w") as file:
    file.write("#!/bin/bash\n")
    total_jobs = len(nums_neurons) * len(network_types) * len(learning_rates)
    current_job = 0
    for network_type in network_types:
        for num_neurons in nums_neurons:
            for learning_rate in learning_rates:
                file.write(f"sbatch jobscript_{network_type}_{num_neurons}neurons_{learning_rate}lr.sh\n")
                current_job += 1
                if current_job < total_jobs:
                    file.write("sleep 30\n")
            