#!/bin/bash
jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.001lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.001lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.001lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.0005lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.0005lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.0005lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.0001lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.0001lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_0.0001lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_5e-05lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_5e-05lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_5e-05lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_1e-05lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_1e-05lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_64neurons_1e-05lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.001lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.001lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.001lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.0005lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.0005lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.0005lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.0001lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.0001lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_0.0001lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_5e-05lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_5e-05lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_5e-05lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_1e-05lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_1e-05lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_48neurons_1e-05lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.001lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.001lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.001lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.0005lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.0005lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.0005lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.0001lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.0001lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_0.0001lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_5e-05lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_5e-05lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_5e-05lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_1e-05lr_0.0wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_1e-05lr_0.0001wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/31_adaptmod_CfC_tanh_32/jobscript_adaptmod_32neurons_1e-05lr_0.01wd.sh | awk '{print $4}')
echo $jobid >> jobids_31_adaptmod_CfC_tanh_32.txt
