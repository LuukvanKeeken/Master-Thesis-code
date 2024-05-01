#!/bin/bash

# File containing job IDs
jobids_file="jobids_2.txt"

# Directory to move
src_dir="~/Thesis/Master_Thesis_Code/LTC_A2C/all_jobs_2/"
dest_dir="/projects/s3512290/LTC_A2C/all_jobs_2/"

# Function to check if a job is running
is_job_running() {
    local jobid=$1
    local status=$(scontrol show jobid -dd $jobid | grep -oP 'JobState=\K\S+' || echo "UNKNOWN")
    [[ $status != "COMPLETED" && $status != "FAILED" && $status != "CANCELLED" && $status != "UNKNOWN" ]]
}


# Main loop
while true; do
    # Assume all jobs are done until proven otherwise
    all_jobs_done=true

    # Read the job IDs from the file
    while read -r jobid; do
        if is_job_running $jobid; then
            all_jobs_done=false
            break
        fi
    done < "$jobids_file"

    if $all_jobs_done; then
        # All jobs are done, move the directory
        mv "$src_dir" "$dest_dir"
        break
    else
        # Not all jobs are done, wait for a minute before checking again
        sleep 60
    fi
done