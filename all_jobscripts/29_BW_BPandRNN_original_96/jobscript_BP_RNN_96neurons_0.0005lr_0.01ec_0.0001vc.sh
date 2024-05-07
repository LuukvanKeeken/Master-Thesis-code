#!/bin/bash
#SBATCH --time=120:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_29_BW_BPandRNN_original_96 --network_type BP_RNN --num_neurons 96 --learning_rate 0.0005 --result_id 68064 --entropy_coef 0.01 --value_pred_coef 0.0001

deactivate