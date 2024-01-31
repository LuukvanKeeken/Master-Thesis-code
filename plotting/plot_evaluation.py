import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

start = 0
end = 200
modified_parameter = "pole_mass"
statistic = "mean"

CfC_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/{modified_parameter}/{statistic}s.npy")
CfC_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/{modified_parameter}/stddevs.npy")
CfC_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard/{modified_parameter}/percentages.npy")
CfC_evaluation_percentages = (CfC_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(CfC_evaluation_percentages[start:end], CfC_evaluation_statistic[start:end], CfC_evaluation_std[start:end], capsize=5, label = "CfC")
elif statistic == "median":
    ax.plot(CfC_evaluation_percentages[start:end], CfC_evaluation_statistic[start:end], label = "CfC")


# BP_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_10models/{modified_parameter}/{statistic}s.npy")
# BP_evaluation_std = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_10models/{modified_parameter}/stddevs.npy")
# BP_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_10models/{modified_parameter}/percentages.npy")
# BP_evaluation_percentages = (BP_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(BP_evaluation_percentages[start:end], BP_evaluation_statistic[start:end], BP_evaluation_std[start:end], capsize=5, label = "BP")
# elif statistic == "median":
#     ax.plot(BP_evaluation_percentages[start:end], BP_evaluation_statistic[start:end], label = "BP")

# BPRANGE_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.91.1_10models/{modified_parameter}/{statistic}s.npy")
# BPRANGE_evaluation_std = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.91.1_10models/{modified_parameter}/stddevs.npy")
# BPRANGE_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.91.1_10models/{modified_parameter}/percentages.npy")
# BPRANGE_evaluation_percentages = (BPRANGE_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(BPRANGE_evaluation_percentages[start:end], BPRANGE_evaluation_statistic[start:end], BPRANGE_evaluation_std[start:end], capsize=5, label = "BP_0.9_1.1")
# elif statistic == "median":
#     ax.plot(BPRANGE_evaluation_percentages[start:end], BPRANGE_evaluation_statistic[start:end], label = "BP_0.9_1.1")

# BPRANGE2_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.82.0_10models/{modified_parameter}/{statistic}s.npy")
# BPRANGE2_evaluation_std = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.82.0_10models/{modified_parameter}/stddevs.npy")
# BPRANGE2_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.82.0_10models/{modified_parameter}/percentages.npy")
# BPRANGE2_evaluation_percentages = (BPRANGE2_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(BPRANGE2_evaluation_percentages[start:end], BPRANGE2_evaluation_statistic[start:end], BPRANGE2_evaluation_std[start:end], capsize=5, label = "BP_0.8_2.0")
# elif statistic == "median":
#     ax.plot(BPRANGE2_evaluation_percentages[start:end], BPRANGE2_evaluation_statistic[start:end], label = "BP_0.8_2.0")

# BPRANGE3_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.73.0_10models/{modified_parameter}/{statistic}s.npy")
# BPRANGE3_evaluation_std = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.73.0_10models/{modified_parameter}/stddevs.npy")
# BPRANGE3_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.73.0_10models/{modified_parameter}/percentages.npy")
# BPRANGE3_evaluation_percentages = (BPRANGE3_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(BPRANGE3_evaluation_percentages[start:end], BPRANGE3_evaluation_statistic[start:end], BPRANGE3_evaluation_std[start:end], capsize=5, label = "BP_0.7_3.0")
# elif statistic == "median":
#     ax.plot(BPRANGE3_evaluation_percentages[start:end], BPRANGE3_evaluation_statistic[start:end], label = "BP_0.7_3.0")

# BPRANGE4_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.79.0_10models/{modified_parameter}/{statistic}s.npy")
# BPRANGE4_evaluation_std = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.79.0_10models/{modified_parameter}/stddevs.npy")
# BPRANGE4_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_range0.79.0_10models/{modified_parameter}/percentages.npy")
# BPRANGE4_evaluation_percentages = (BPRANGE4_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(BPRANGE4_evaluation_percentages[start:end], BPRANGE4_evaluation_statistic[start:end], BPRANGE4_evaluation_std[start:end], capsize=5, label = "BP_0.7_9.0")
# elif statistic == "median":
#     ax.plot(BPRANGE4_evaluation_percentages[start:end], BPRANGE4_evaluation_statistic[start:end], label = "BP_0.7_9.0")


# Standard_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_RNN_10models/{modified_parameter}/{statistic}s.npy")
# Standard_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_RNN_10models/{modified_parameter}/stddevs.npy")
# Standard_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_RNN_10models/{modified_parameter}/percentages.npy")
# Standard_evaluation_percentages = (Standard_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(Standard_evaluation_percentages[start:end], Standard_evaluation_statistic[start:end], Standard_evaluation_std[start:end], capsize=5, label = "Standard")
# elif statistic == "median":
#     ax.plot(Standard_evaluation_percentages[start:end], Standard_evaluation_statistic[start:end], label = "Standard")

# MLP_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_10models/{modified_parameter}/{statistic}s.npy")
# MLP_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_10models/{modified_parameter}/stddevs.npy")
# MLP_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_10models/{modified_parameter}/percentages.npy")
# MLP_evaluation_percentages = (MLP_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(MLP_evaluation_percentages[start:end], MLP_evaluation_statistic[start:end], MLP_evaluation_std[start:end], capsize=5, label = "MLP")
# elif statistic == "median":
#     ax.plot(MLP_evaluation_percentages[start:end], MLP_evaluation_statistic[start:end], label = "MLP")

# MLPLIB_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLPLIBRARY_10models/{modified_parameter}/{statistic}s.npy")
# MLPLIB_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLPLIBRARY_10models/{modified_parameter}/stddevs.npy")
# MLPLIB_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLPLIBRARY_10models/{modified_parameter}/percentages.npy")
# MLPLIB_evaluation_percentages = (MLPLIB_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(MLPLIB_evaluation_percentages[start:end], MLPLIB_evaluation_statistic[start:end], MLPLIB_evaluation_std[start:end], capsize=5, label = "MLPLIB")
# elif statistic == "median":
#     ax.plot(MLPLIB_evaluation_percentages[start:end], MLPLIB_evaluation_statistic[start:end], label = "MLPLIB")

# MLPRANGE_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range_10models/{modified_parameter}/{statistic}s.npy")
# MLPRANGE_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range_10models/{modified_parameter}/stddevs.npy")
# MLPRANGE_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range_10models/{modified_parameter}/percentages.npy")
# MLPRANGE_evaluation_percentages = (MLPRANGE_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(MLPRANGE_evaluation_percentages[start:end], MLPRANGE_evaluation_statistic[start:end], MLPRANGE_evaluation_std[start:end], capsize=5, label = "MLP_0.9_1.1")
# elif statistic == "median":
#     ax.plot(MLPRANGE_evaluation_percentages[start:end], MLPRANGE_evaluation_statistic[start:end], label = "MLP_0.9_1.1")

# MLPRANGE2_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.82.0_10models/{modified_parameter}/{statistic}s.npy")
# MLPRANGE2_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.82.0_10models/{modified_parameter}/stddevs.npy")
# MLPRANGE2_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.82.0_10models/{modified_parameter}/percentages.npy")
# MLPRANGE2_evaluation_percentages = (MLPRANGE2_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(MLPRANGE2_evaluation_percentages[start:end], MLPRANGE2_evaluation_statistic[start:end], MLPRANGE2_evaluation_std[start:end], capsize=5, label = "MLP_0.8_2.0")
# elif statistic == "median":
#     ax.plot(MLPRANGE2_evaluation_percentages[start:end], MLPRANGE2_evaluation_statistic[start:end], label = "MLP_0.8_2.0")

# MLPRANGE3_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.73.0_10models/{modified_parameter}/{statistic}s.npy")
# MLPRANGE3_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.73.0_10models/{modified_parameter}/stddevs.npy")
# MLPRANGE3_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.73.0_10models/{modified_parameter}/percentages.npy")
# MLPRANGE3_evaluation_percentages = (MLPRANGE3_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(MLPRANGE3_evaluation_percentages[start:end], MLPRANGE3_evaluation_statistic[start:end], MLPRANGE3_evaluation_std[start:end], capsize=5, label = "MLP_0.7_3.0")
# elif statistic == "median":
#     ax.plot(MLPRANGE3_evaluation_percentages[start:end], MLPRANGE3_evaluation_statistic[start:end], label = "MLP_0.7_3.0")

# MLPRANGE4_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.64.0_10models/{modified_parameter}/{statistic}s.npy")
# MLPRANGE4_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.64.0_10models/{modified_parameter}/stddevs.npy")
# MLPRANGE4_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.64.0_10models/{modified_parameter}/percentages.npy")
# MLPRANGE4_evaluation_percentages = (MLPRANGE4_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(MLPRANGE4_evaluation_percentages[start:end], MLPRANGE4_evaluation_statistic[start:end], MLPRANGE4_evaluation_std[start:end], capsize=5, label = "MLP_0.6_4.0")
# elif statistic == "median":
#     ax.plot(MLPRANGE4_evaluation_percentages[start:end], MLPRANGE4_evaluation_statistic[start:end], label = "MLP_0.6_4.0")

# MLPRANGE5_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.79.0_10models/{modified_parameter}/{statistic}s.npy")
# MLPRANGE5_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.79.0_10models/{modified_parameter}/stddevs.npy")
# MLPRANGE5_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_range0.79.0_10models/{modified_parameter}/percentages.npy")
# MLPRANGE5_evaluation_percentages = (MLPRANGE5_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(MLPRANGE5_evaluation_percentages[start:end], MLPRANGE5_evaluation_statistic[start:end], MLPRANGE5_evaluation_std[start:end], capsize=5, label = "MLP_0.7_9.0")
# elif statistic == "median":
#     ax.plot(MLPRANGE5_evaluation_percentages[start:end], MLPRANGE5_evaluation_statistic[start:end], label = "MLP_0.7_9.0")

ax.grid(True)
ax.set_ylim(0, 250)
ax.set_xlabel(f"{modified_parameter} increase (%)")
ax.set_ylabel(f"{statistic} average reward")
ax.set_title(f"{statistic} average evaluation reward in modified environments ({modified_parameter})")
ax.set_xticks(CfC_evaluation_percentages[start:end])
ax.legend()


plt.show()