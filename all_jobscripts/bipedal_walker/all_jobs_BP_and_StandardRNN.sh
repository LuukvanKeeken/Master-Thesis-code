#!/bin/bash
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_Standard_RNN_96neurons_0.0001lr_0.001ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_Standard_RNN_96neurons_0.0001lr_0.001ec_0.01vc.sh'
sleep 3
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_Standard_RNN_96neurons_0.0001lr_0.01ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_Standard_RNN_96neurons_0.0001lr_0.01ec_0.01vc.sh'
sleep 3
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_BP_RNN_96neurons_0.0001lr_0.001ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_BP_RNN_96neurons_0.0001lr_0.001ec_0.01vc.sh'
sleep 3
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_BP_RNN_96neurons_0.0001lr_0.01ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/BP_and_StandardRNN/jobscript_BP_RNN_96neurons_0.0001lr_0.01ec_0.01vc.sh'
