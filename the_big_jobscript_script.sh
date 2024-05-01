#!/bin/bash

chmod +x all_jobs_1.sh
chmod +x status_checker_1.sh
chmod +x all_jobs_2.sh
chmod +x status_checker_2.sh

./all_jobs_1.sh
./status_checker_1.sh &

./all_jobs_2.sh
./status_checker_2.sh &