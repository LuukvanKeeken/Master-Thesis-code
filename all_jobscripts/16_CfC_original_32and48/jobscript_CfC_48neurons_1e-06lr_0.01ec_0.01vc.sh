#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_16_CfC_original_32and48 --network_type CfC --num_neurons 48 --learning_rate 1e-06 --result_id 55029 --entropy_coef 0.01 --value_pred_coef 0.01

deactivate