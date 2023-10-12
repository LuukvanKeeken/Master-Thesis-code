import numpy as np
import matplotlib.pyplot as plt


BP_DQRNN_best_episodes_avg, BP_DQRNN_best_episodes_stddev = np.load("fine_tuning/BP_DQRNN_1000eps_best_episodes.npy")

Standard_DQRNN_best_episodes_avg, Standard_DQRNN_best_episodes_stddev = np.load("fine_tuning/Standard_DQRNN_1000eps_best_episodes.npy")
percentages = np.linspace(10, 100, 10)


fig = plt.figure()

plt.errorbar(percentages, BP_DQRNN_best_episodes_avg, BP_DQRNN_best_episodes_stddev, capsize=5, label = "Backpropamine")

plt.errorbar(percentages, Standard_DQRNN_best_episodes_avg, Standard_DQRNN_best_episodes_stddev, capsize=5, label = "Standard")
plt.plot(percentages, [35.4, 14.6, 60.4, 85.4, 56.3, 62.5, 72.9, 60.4, 47.9, 68.8], label = "r-STDP (estimated from plot)")
plt.grid(True)
plt.ylim(-30, 250)
plt.xlabel("Pole length increase (%)")
plt.ylabel("Average fine-tuning episodes")
plt.title("Number of fine-tuning episodes needed to reach perfect evaluation performance")
plt.xticks(percentages)
plt.legend()


plt.show()
