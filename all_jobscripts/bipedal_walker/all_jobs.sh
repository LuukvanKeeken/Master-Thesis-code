#!/bin/bash
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_0.0001lr_0.001ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_0.0001lr_0.001ec_1.0vc.sh'
sleep 3
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_0.0001lr_1.0ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_0.0001lr_1.0ec_1.0vc.sh'
sleep 3
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_1e-05lr_0.001ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_1e-05lr_0.001ec_1.0vc.sh'
sleep 3
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_1e-05lr_1.0ec_0.001vc.sh'
sleep 3
sbatch 'bipedal_walker/CfC/jobscript_CfC_96neurons_1e-05lr_1.0ec_1.0vc.sh'
