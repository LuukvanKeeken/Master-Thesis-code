#!/bin/bash
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.01_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.001_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_0.01_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_tanh_1.0_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_relu_0.001_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_relu_0.01_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_relu_1.0_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_relu_0.001_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_relu_0.01_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_0.0001lr_[11, 256, 128]_relu_1.0_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_tanh_0.001_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_tanh_0.01_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_tanh_1.0_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_tanh_0.001_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_tanh_0.01_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_tanh_1.0_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.001_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.01_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_1.0_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.001_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.01_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMBP/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_1.0_0.01.sh'
