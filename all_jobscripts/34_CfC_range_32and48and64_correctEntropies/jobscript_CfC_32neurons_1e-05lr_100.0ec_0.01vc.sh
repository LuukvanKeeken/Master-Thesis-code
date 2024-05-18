#!/bin/bash
#SBATCH --time=35:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_34_CfC_range_32and48and64_correctEntropies --network_type CfC --num_neurons 32 --learning_rate 1e-05 --result_id 73057 --entropy_coef 100.0 --value_pred_coef 0.01

deactivate