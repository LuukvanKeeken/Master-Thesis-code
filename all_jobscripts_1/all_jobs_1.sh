#!/bin/bash

jobid=$(sbatch job_1_1.sh | awk '{print $4}')
echo $jobid >> jobids_1.txt

while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 60
done

jobid=$(sbatch job_1_2.sh | awk '{print $4}')
echo $jobid >> jobids_1.txt

while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 60
done

jobid=$(sbatch job_1_3.sh | awk '{print $4}')
echo $jobid >> jobids_1.txt

while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 60
done