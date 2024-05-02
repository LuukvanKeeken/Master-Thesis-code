#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_4_NMBP_tanh_48 --num_neurons 48 --learning_rate 1e-05 --result_id 43097 --neuromod_network_dims 3 256 128 --encoder_output_activation tanh --encoder_hidden_activation tanh --entropy_coef 0.0 --value_pred_coef 0.01

deactivate