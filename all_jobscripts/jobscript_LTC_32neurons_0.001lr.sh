#!/bin/bash
#SBATCH --time=10:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_LTC_habrok --num_neurons 32 --neuron_type LTC --learning_rate 0.001

deactivate