#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_21_LTC_original_32and64 --network_type LTC --num_neurons 32 --learning_rate 0.0001 --result_id 60002 --entropy_coef 0.0001 --value_pred_coef 0.0001

deactivate