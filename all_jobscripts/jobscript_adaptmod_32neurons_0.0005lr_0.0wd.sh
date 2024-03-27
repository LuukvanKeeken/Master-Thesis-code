#!/bin/bash
#SBATCH --time=10:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_adaptation_module_batched --num_neurons_adaptation 32 --lr_adapt_mod 0.0005 --wd_adapt_mod 0.0 --result_id 723 

deactivate