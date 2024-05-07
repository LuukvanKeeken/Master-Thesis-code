#!/bin/bash
jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.001lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_32neurons_0.0001lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.001lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 256, 128]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 192, 96]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.01ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.01ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_0.01ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_1.0ec_0.0001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_1.0ec_0.001vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/19_NMCfC_relu_32and64/jobscript_encoder_64neurons_0.0001lr_[3, 128, 80]_1.0ec_0.01vc_relu.sh" | awk '{print $4}')
echo $jobid >> jobids_19_NMCfC_relu_32and64.txt
