#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_LTC_habrok_CfC_range_48 --network_type CfC --num_neurons 48 --learning_rate 1e-06 --result_id 40052 --entropy_coef 0.0001 --value_pred_coef 0.0001

deactivate