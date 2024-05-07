#!/bin/bash
#SBATCH --time=35:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_22_LTC_range_32and64 --network_type LTC --num_neurons 64 --learning_rate 1e-05 --result_id 61025 --entropy_coef 0.0 --value_pred_coef 0.01

deactivate