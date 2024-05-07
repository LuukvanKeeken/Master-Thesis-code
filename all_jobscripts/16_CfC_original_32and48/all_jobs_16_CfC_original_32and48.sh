#!/bin/bash
jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_32neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/16_CfC_original_32and48/jobscript_CfC_48neurons_1e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_16_CfC_original_32and48.txt
