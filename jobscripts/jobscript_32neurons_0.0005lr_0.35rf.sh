#!/bin/bash
#SBATCH --time=05:30:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_LTC_neuromod_habrok --num_neurons 32 --randomization_factor 0.35 --learning_rate 0.0005

deactivate