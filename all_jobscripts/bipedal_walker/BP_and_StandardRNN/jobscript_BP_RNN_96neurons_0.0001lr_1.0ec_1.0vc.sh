#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --mem=7000M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_LTC_batched --network_type BP_RNN --num_neurons 96 --learning_rate 0.0001 --result_id 400017 --entropy_coef 1.0 --value_pred_coef 1.0

deactivate