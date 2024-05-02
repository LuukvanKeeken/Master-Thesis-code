#!/bin/bash
jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_0.0001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/8_LTC_original_48/jobscript_LTC_48neurons_1e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_8_LTC_original_48.txt
