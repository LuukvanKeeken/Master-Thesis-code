#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_17_MLP_range_32and64 --network_type Standard_MLP --num_neurons 32 --learning_rate 5e-05 --result_id 56030 --entropy_coef 1.0 --value_pred_coef 0.5

deactivate