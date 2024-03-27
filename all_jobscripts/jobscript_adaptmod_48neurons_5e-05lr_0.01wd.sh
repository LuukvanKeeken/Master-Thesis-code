#!/bin/bash
#SBATCH --time=10:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_adaptation_module_batched --num_neurons_adaptation 48 --lr_adapt_mod 5e-05 --result_id 671 --wd_adapt_mod 0.01

deactivate