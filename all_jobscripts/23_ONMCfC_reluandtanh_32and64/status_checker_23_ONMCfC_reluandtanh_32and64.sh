#!/bin/bash

# File containing job IDs
jobids_file="jobids_23_ONMCfC_reluandtanh_32and64.txt"

# Directory to move
src_dir="$HOME/Thesis/Master_Thesis_Code/LTC_A2C/23_ONMCfC_reluandtanh_32and64/"
dest_dir="/projects/s3512290/LTC_A2C/23_ONMCfC_reluandtanh_32and64/"

# Function to check if a job is running
is_job_running() {
    local jobid=$1
    local status=$(scontrol show jobid -dd $jobid 2>/dev/null | grep -oP 'JobState=\K\S+' || echo "COMPLETED")
    [[ $status != "COMPLETED" && $status != "FAILED" && $status != "CANCELLED" ]]
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
        mv -v "$src_dir" "$dest_dir"
        break
    else
        # Not all jobs are done, wait for a minute before checking again
        sleep 60
    fi
done
