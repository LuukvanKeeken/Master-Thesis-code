#!/bin/bash
#SBATCH --time=24:00:00

module purge
module load Python/3.8.16-GCCcore-11.2.0

source $HOME/venvs/LTC/bin/activate

python3 -m Master_Thesis_Code.training_23_ONMCfC_reluandtanh_32and64 --num_neurons 32 --learning_rate 0.001 --result_id 62021 --neuromod_network_dims 3 128 80 --encoder_output_activation tanh --encoder_hidden_activation tanh --entropy_coef 0.01 --value_pred_coef 0.0001

deactivate