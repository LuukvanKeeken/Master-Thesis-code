#!/bin/bash
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.001_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.01_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_1.0_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.001_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.01_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_1.0_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.001_1.0.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_0.01_1.0.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-05lr_[11, 256, 128]_relu_1.0_1.0.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_0.001_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_0.01_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_1.0_0.001.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_0.001_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_0.01_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_1.0_0.01.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_0.001_1.0.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_0.01_1.0.sh'
sleep 3
sbatch 'bipedal_walker/NMCfC/jobscript_encoder_96neurons_1e-06lr_[11, 256, 128]_relu_1.0_1.0.sh'
