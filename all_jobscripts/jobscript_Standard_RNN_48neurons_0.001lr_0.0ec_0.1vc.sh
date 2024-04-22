#!/bin/bash
#SBATCH --time=20:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_BP_habrok --network_type Standard_RNN --num_neurons 48 --learning_rate 0.001 --result_id 30003 --entropy_coef 0.0 --value_pred_coef 0.1

deactivate