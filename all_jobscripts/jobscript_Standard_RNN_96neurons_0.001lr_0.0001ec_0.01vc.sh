#!/bin/bash
#SBATCH --time=100:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_BP_habrok --network_type Standard_RNN --num_neurons 96 --learning_rate 0.001 --result_id 632 --entropy_coef 0.0001 --value_pred_coef 0.01

deactivate