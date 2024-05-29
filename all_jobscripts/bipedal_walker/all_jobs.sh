#!/bin/bash
sbatch 'jobscript_Standard_RNN_96neurons_0.0001lr_0.001ec_0.001vc.sh'
sleep 3
sbatch 'jobscript_Standard_RNN_96neurons_0.0001lr_0.001ec_1.0vc.sh'
sleep 3
sbatch 'jobscript_Standard_RNN_96neurons_0.0001lr_1.0ec_0.001vc.sh'
sleep 3
sbatch 'jobscript_Standard_RNN_96neurons_0.0001lr_1.0ec_1.0vc.sh'
sleep 3
sbatch 'jobscript_BP_RNN_96neurons_0.0001lr_0.001ec_0.001vc.sh'
sleep 3
sbatch 'jobscript_BP_RNN_96neurons_0.0001lr_0.001ec_1.0vc.sh'
sleep 3
sbatch 'jobscript_BP_RNN_96neurons_0.0001lr_1.0ec_0.001vc.sh'
sleep 3
sbatch 'jobscript_BP_RNN_96neurons_0.0001lr_1.0ec_1.0vc.sh'
