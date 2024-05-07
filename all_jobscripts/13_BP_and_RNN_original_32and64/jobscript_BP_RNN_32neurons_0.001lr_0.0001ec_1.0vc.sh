#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_13_BP_and_RNN_original_32and64 --network_type BP_RNN --num_neurons 32 --learning_rate 0.001 --result_id 52039 --entropy_coef 0.0001 --value_pred_coef 1.0

deactivate