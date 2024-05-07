#!/bin/bash
jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_32neurons_5e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_0.001lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/17_MLP_range_32and64/jobscript_Standard_MLP_64neurons_5e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_17_MLP_range_32and64.txt
