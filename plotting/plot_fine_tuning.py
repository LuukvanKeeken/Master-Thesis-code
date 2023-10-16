import numpy as np
import matplotlib.pyplot as plt


BP_DQRNN_best_episodes_avg, BP_DQRNN_best_episodes_stddev = np.load("../fine_tuning_results/BP_DQRNN_1000eps_best_episodes_correct.npy")

BP_A2C_RNN_best_episodes_avg, BP_A2C_RNN_best_episodes_stddev = np.load("../BP_A2C/fine_tuning_results/BP_A2C_RNN_fine_tuning_result_8_20231016/best_episodes.npy")
percentages = np.linspace(10, 100, 10)


fig = plt.figure()

plt.errorbar(percentages, BP_DQRNN_best_episodes_avg, BP_DQRNN_best_episodes_stddev, capsize=5, label = "BP DQRNN")

plt.errorbar(percentages, BP_A2C_RNN_best_episodes_avg, BP_A2C_RNN_best_episodes_stddev, capsize=5, label = "BP A2C RNN")
plt.plot(percentages, [35.4, 14.6, 60.4, 85.4, 56.3, 62.5, 72.9, 60.4, 47.9, 68.8], label = "r-STDP (estimated from plot)")
plt.grid(True)
plt.ylim(-30, 250)
plt.xlabel("Pole length increase (%)")
plt.ylabel("Average fine-tuning episodes")
plt.title("Number of fine-tuning episodes needed to reach perfect evaluation performance")
plt.xticks(percentages)
plt.legend()


plt.show()
