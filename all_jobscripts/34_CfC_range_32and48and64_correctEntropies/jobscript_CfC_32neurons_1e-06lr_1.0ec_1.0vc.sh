#!/bin/bash
#SBATCH --time=35:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_34_CfC_range_32and48and64_correctEntropies --network_type CfC --num_neurons 32 --learning_rate 1e-06 --result_id 73071 --entropy_coef 1.0 --value_pred_coef 1.0

deactivate