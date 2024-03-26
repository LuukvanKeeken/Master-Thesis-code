#!/bin/bash
#SBATCH --time=10:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_LTC_neuromod_habrok --num_neurons 32 --learning_rate 0.0001 --result_id 1082 --neuromod_network_dims 3 192 96 --encoder_output_activation relu --encoder_hidden_activation relu

deactivate