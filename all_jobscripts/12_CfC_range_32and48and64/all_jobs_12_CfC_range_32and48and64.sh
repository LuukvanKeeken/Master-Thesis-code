#!/bin/bash
jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_32neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_32neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_48neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_48neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_64neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_64neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_64neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/12_CfC_range_32and48and64/jobscript_CfC_64neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_12_CfC_range_32and48and64.txt
