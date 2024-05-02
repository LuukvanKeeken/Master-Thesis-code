#!/bin/bash

chmod +x 1_CfC_range_48/all_jobs_1_CfC_range_48.sh
chmod +x 1_CfC_range_48/status_checker_1_CfC_range_48.sh
chmod +x 2_BP_and_RNN_original_48/all_jobs_2_BP_and_RNN_original_48.sh
chmod +x 2_BP_and_RNN_original_48/status_checker_2_BP_and_RNN_original_48.sh
chmod +x 3_NMBP_relu_48/all_jobs_3_NMBP_relu_48.sh
chmod +x 3_NMBP_relu_48/status_checker_3_NMBP_relu_48.sh
chmod +x 4_NMBP_tanh_48/all_jobs_4_NMBP_tanh_48.sh
chmod +x 4_NMBP_tanh_48/status_checker_4_NMBP_tanh_48.sh
chmod +x 5_MLP_range_48/all_jobs_5_MLP_range_48.sh
chmod +x 5_MLP_range_48/status_checker_5_MLP_range_48.sh
chmod +x 6_MLP_original_48/all_jobs_6_MLP_original_48.sh
chmod +x 6_MLP_original_48/status_checker_6_MLP_original_48.sh
chmod +x 7_NMCfC_tanh_48/all_jobs_7_NMCfC_tanh_48.sh
chmod +x 7_NMCfC_tanh_48/status_checker_7_NMCfC_tanh_48.sh
chmod +x 8_LTC_original_48/all_jobs_8_LTC_original_48.sh
chmod +x 8_LTC_original_48/status_checker_8_LTC_original_48.sh
chmod +x 9_LTC_range_48/all_jobs_9_LTC_range_48.sh
chmod +x 9_LTC_range_48/status_checker_9_LTC_range_48.sh
chmod +x 10_ONMCfC_relu_48/all_jobs_10_ONMCfC_relu_48.sh
chmod +x 10_ONMCfC_relu_48/status_checker_10_ONMCfC_relu_48.sh
chmod +x 11_ONMCfC_tanh_48/all_jobs_11_ONMCfC_tanh_48.sh
chmod +x 11_ONMCfC_tanh_48/status_checker_11_ONMCfC_tanh_48.sh


./1_CfC_range_48/all_jobs_1_CfC_range_48.sh
./1_CfC_range_48/status_checker_1_CfC_range_48.sh &

./2_BP_and_RNN_original_48/all_jobs_2_BP_and_RNN_original_48.sh
./2_BP_and_RNN_original_48/status_checker_2_BP_and_RNN_original_48.sh &

./3_NMBP_relu_48/all_jobs_3_NMBP_relu_48.sh
./3_NMBP_relu_48/status_checker_3_NMBP_relu_48.sh &

./4_NMBP_tanh_48/all_jobs_4_NMBP_tanh_48.sh
./4_NMBP_tanh_48/status_checker_4_NMBP_tanh_48.sh &

./5_MLP_range_48/all_jobs_5_MLP_range_48.sh
./5_MLP_range_48/status_checker_5_MLP_range_48.sh &

./6_MLP_original_48/all_jobs_6_MLP_original_48.sh
./6_MLP_original_48/status_checker_6_MLP_original_48.sh &

./7_NMCfC_tanh_48/all_jobs_7_NMCfC_tanh_48.sh
./7_NMCfC_tanh_48/status_checker_7_NMCfC_tanh_48.sh &

./8_LTC_original_48/all_jobs_8_LTC_original_48.sh
./8_LTC_original_48/status_checker_8_LTC_original_48.sh &

./9_LTC_range_48/all_jobs_9_LTC_range_48.sh
./9_LTC_range_48/status_checker_9_LTC_range_48.sh &

./10_ONMCfC_relu_48/all_jobs_10_ONMCfC_relu_48.sh
./10_ONMCfC_relu_48/status_checker_10_ONMCfC_relu_48.sh &

./11_ONMCfC_tanh_48/all_jobs_11_ONMCfC_tanh_48.sh
./11_ONMCfC_tanh_48/status_checker_11_ONMCfC_tanh_48.sh &