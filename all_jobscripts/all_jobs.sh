#!/bin/bash
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_49neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_50neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_51neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_52neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_53neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_54neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_55neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_56neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_57neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 5 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_58neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'
