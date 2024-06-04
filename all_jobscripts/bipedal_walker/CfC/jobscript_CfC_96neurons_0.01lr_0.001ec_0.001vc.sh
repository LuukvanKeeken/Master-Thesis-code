#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --mem=4500M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_LTC_batched --network_type CfC --num_neurons 96 --learning_rate 0.01 --result_id 300000 --entropy_coef 0.001 --value_pred_coef 0.001

deactivate