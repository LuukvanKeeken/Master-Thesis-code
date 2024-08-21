#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=8000M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_different_cartpole_envs --network_type Standard_RNN --num_neurons 32 --learning_rate 0.001 --result_id 110 --entropy_coef 0.01 --value_pred_coef 0.0001

deactivate