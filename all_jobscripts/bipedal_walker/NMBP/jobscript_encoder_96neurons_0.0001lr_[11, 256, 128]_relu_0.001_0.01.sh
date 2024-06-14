#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --mem=10000M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_neuromod_batched --num_neurons 96 --learning_rate 0.0001 --result_id 2300009 --neuromod_network_dims 11 256 128 --encoder_output_activation relu --encoder_hidden_activation relu --value_pred_coef 0.001 --entropy_coef 0.01 --neuron_type BP

deactivate