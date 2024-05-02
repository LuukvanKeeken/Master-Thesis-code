#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_6_MLP_original_48 --network_type Standard_MLP --num_neurons 48 --learning_rate 5e-05 --result_id 45021 --entropy_coef 0.0001 --value_pred_coef 0.01

deactivate