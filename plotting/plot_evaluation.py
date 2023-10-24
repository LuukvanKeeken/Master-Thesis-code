import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

start = 0
end = 200
modified_parameter = "pole_length"
statistic = "median"

BP_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_10models/{modified_parameter}/{statistic}s.npy")
BP_evaluation_std = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_10models/{modified_parameter}/stddevs.npy")
BP_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/BP_A2C_RNN_10models/{modified_parameter}/percentages.npy")
print(BP_evaluation_percentages)
exit()
BP_evaluation_percentages = (BP_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(BP_evaluation_percentages[start:end], BP_evaluation_statistic[start:end], BP_evaluation_std[start:end], capsize=5, label = "BP")
elif statistic == "median":
    ax.plot(BP_evaluation_percentages[start:end], BP_evaluation_statistic[start:end], label = "BP")


Standard_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_RNN_10models/{modified_parameter}/{statistic}s.npy")
Standard_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_RNN_10models/{modified_parameter}/stddevs.npy")
Standard_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_RNN_10models/{modified_parameter}/percentages.npy")
Standard_evaluation_percentages = (Standard_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(Standard_evaluation_percentages[start:end], Standard_evaluation_statistic[start:end], Standard_evaluation_std[start:end], capsize=5, label = "Standard")
elif statistic == "median":
    ax.plot(Standard_evaluation_percentages[start:end], Standard_evaluation_statistic[start:end], label = "Standard")

MLP_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_10models/{modified_parameter}/{statistic}s.npy")
MLP_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_10models/{modified_parameter}/stddevs.npy")
MLP_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLP_10models/{modified_parameter}/percentages.npy")
MLP_evaluation_percentages = (MLP_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(MLP_evaluation_percentages[start:end], MLP_evaluation_statistic[start:end], MLP_evaluation_std[start:end], capsize=5, label = "MLP")
elif statistic == "median":
    ax.plot(MLP_evaluation_percentages[start:end], MLP_evaluation_statistic[start:end], label = "MLP")

MLPLIB_evaluation_statistic = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLPLIBRARY_10models/{modified_parameter}/{statistic}s.npy")
MLPLIB_evaluation_std = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLPLIBRARY_10models/{modified_parameter}/stddevs.npy")
MLPLIB_evaluation_percentages = np.load(f"../BP_A2C/evaluation_results/Standard_A2C_MLPLIBRARY_10models/{modified_parameter}/percentages.npy")
MLPLIB_evaluation_percentages = (MLPLIB_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(MLPLIB_evaluation_percentages[start:end], MLPLIB_evaluation_statistic[start:end], MLPLIB_evaluation_std[start:end], capsize=5, label = "MLPLIB")
elif statistic == "median":
    ax.plot(MLPLIB_evaluation_percentages[start:end], MLPLIB_evaluation_statistic[start:end], label = "MLPLIB")


ax.grid(True)
ax.set_ylim(0, 250)
ax.set_xlabel(f"{modified_parameter} increase (%)")
ax.set_ylabel(f"{statistic} average reward")
ax.set_title(f"{statistic} average evaluation reward in modified environments ({modified_parameter})")
ax.set_xticks(BP_evaluation_percentages[start:end])
ax.legend()


plt.show()