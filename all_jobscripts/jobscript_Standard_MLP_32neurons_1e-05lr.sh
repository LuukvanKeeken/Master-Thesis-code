#!/bin/bash
#SBATCH --time=05:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_BP_habrok --num_neurons 32 --network_type Standard_MLP --learning_rate 1e-05

deactivate