#!/bin/bash
#SBATCH --time=10:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_LTC_neuromod_habrok_old --num_neurons 48 --learning_rate 0.0001 --result_id 566 --neuromod_network_dims 3 256 128

deactivate