#!/bin/bash
jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.001lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_0.0001lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-05lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_32neurons_1e-06lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.001lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_0.0001lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-05lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_48neurons_1e-06lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.001lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_0.0001lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-05lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_0.001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_10.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_10.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_10.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_10.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_100.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_100.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_100.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/34_CfC_range_32and48and64_correctEntropies/jobscript_CfC_64neurons_1e-06lr_100.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_34_CfC_range_32and48and64_correctEntropies.txt
