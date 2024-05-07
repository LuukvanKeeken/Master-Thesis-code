#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_27_adaptmod_BP_tanh_48 --num_neurons_adaptation 48 --lr_adapt_mod 0.0001 --result_id 66023 --wd_adapt_mod 0.01

deactivate