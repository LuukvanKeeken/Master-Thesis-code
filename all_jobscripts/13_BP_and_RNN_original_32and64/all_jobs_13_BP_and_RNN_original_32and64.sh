#!/bin/bash
jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_32neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_Standard_RNN_64neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_32neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/13_BP_and_RNN_original_32and64/jobscript_BP_RNN_64neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_13_BP_and_RNN_original_32and64.txt
