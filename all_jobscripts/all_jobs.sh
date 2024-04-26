#!/bin/bash
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 256, 128]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 192, 96]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0005lr_[3, 128, 80]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 256, 128]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 192, 96]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_5e-05lr_[3, 128, 80]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_1.0vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.1vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.5vc_relu.sh'

while [ $(squeue -u $USER | wc -l) -ge 120 ]; do
    sleep 60
done
sbatch 'jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_1.0vc_relu.sh'
