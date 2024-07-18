#!/bin/bash
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_1.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_2.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_3.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_4.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_5.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_6.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_7.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_8.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001_9.sh'
