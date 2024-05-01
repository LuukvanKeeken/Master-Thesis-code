#!/bin/bash

jobid = $(sbatch job_1.sh | awk '{print $4}')
echo $jobid >> jobids_1.txt

while [$(squeue -u $USER | wc -l) -gt 2]; do
    sleep 60
done

jobid = $(sbatch job_2.sh | awk '{print $4}')
echo $jobid >> jobids_1.txt

while [$(squeue -u $USER | wc -l) -gt 2]; do
    sleep 60
done

jobid = $(sbatch job_3.sh | awk '{print $4}')
echo $jobid >> jobids_1.txt

while [$(squeue -u $USER | wc -l) -gt 2]; do
    sleep 60
done