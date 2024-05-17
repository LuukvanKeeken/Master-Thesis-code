#!/bin/bash
jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/33_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_33_CfC_range_32and48and64_correctEntropies.txt
