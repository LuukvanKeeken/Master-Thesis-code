#!/bin/bash
jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 256, 128]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 192, 96]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.001lr_[3, 128, 80]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 256, 128]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 192, 96]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_0.0001lr_[3, 128, 80]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 256, 128]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 192, 96]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.0001ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_0.01ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.0001vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.01vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_0.5vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch "$HOME/Thesis/7_NMCfC_tanh_48/jobscript_encoder_48neurons_1e-05lr_[3, 128, 80]_1.0ec_1.0vc_tanh.sh" | awk '{print $4}')
echo $jobid >> jobids_7_NMCfC_tanh_48.txt
