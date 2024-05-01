#!/bin/bash
#SBATCH --time=4:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.test_script --id 3 --group 1

deactivate