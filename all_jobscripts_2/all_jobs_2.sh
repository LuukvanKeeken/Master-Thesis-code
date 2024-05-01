#!/bin/bash

jobid=$(sbatch job_2_1.sh | awk '{print $4}')
echo $jobid >> jobids_2.txt

while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 60
done

jobid=$(sbatch job_2_2.sh | awk '{print $4}')
echo $jobid >> jobids_2.txt

while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 60
done

jobid=$(sbatch job_2_3.sh | awk '{print $4}')
echo $jobid >> jobids_2.txt

while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 60
done