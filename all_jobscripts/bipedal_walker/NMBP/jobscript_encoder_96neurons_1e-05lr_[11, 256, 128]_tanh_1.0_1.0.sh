#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --mem=10000M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_neuromod_batched --num_neurons 96 --learning_rate 1e-05 --result_id 700008 --neuromod_network_dims 11 256 128 --encoder_output_activation tanh --encoder_hidden_activation tanh --value_pred_coef 1.0 --entropy_coef 1.0 --neuron_type BP

deactivate