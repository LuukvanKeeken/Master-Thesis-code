#!/bin/bash
#SBATCH --time=236:00:00
#SBATCH --mem=10000M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_neuromod_batched --mode only_neuromodulated --num_neurons 96 --learning_rate 0.0001 --result_id 3400000 --neuromod_network_dims 11 256 128 --encoder_output_activation tanh --encoder_hidden_activation tanh --value_pred_coef 0.001 --entropy_coef 0.001 --neuron_type CfC

deactivate