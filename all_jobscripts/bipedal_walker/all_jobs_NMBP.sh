#!/bin/bash
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_1.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_2.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_3.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_4.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_5.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_6.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_7.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_8.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01_9.sh'
