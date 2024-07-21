#!/bin/bash
#SBATCH --time=240:00:00
#SBATCH --mem=6000M

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.BW_training_adaptation_module_batched --mode only_neuromodulated --neuron_type CfC --num_neurons_adaptation 96 --lr_adapt_mod 5e-05 --result_id 5200017 --wd_adapt_mod 0.0001

deactivate