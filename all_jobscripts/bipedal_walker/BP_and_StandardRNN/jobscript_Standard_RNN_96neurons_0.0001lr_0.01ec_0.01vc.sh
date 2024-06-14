#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --mem=4000M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_BP_batched --network_type Standard_RNN --num_neurons 96 --learning_rate 0.0001 --result_id 2000003 --entropy_coef 0.01 --value_pred_coef 0.01

deactivate