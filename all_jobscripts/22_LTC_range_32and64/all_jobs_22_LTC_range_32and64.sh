#!/bin/bash
jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_32neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_0.0001lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_0.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_0.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/22_LTC_range_32and64/jobscript_LTC_64neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_22_LTC_range_32and64.txt
