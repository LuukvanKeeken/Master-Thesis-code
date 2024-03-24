#!/bin/bash
#SBATCH --time=10:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_LTC_neuromod_habrok --num_neurons 64 --learning_rate 1e-05 --result_id 401 --neuromod_network_dims 3 256 128 --encoder_output_activation tanh

deactivate