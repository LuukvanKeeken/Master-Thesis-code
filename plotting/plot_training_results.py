import numpy as np
import matplotlib.pyplot as plt


smoothed_scores_0 = np.load("simple_BP_DQRNN_training_1000eps/smoothed_scores_DQN_0.npy")
smoothed_scores_1 = np.load("simple_BP_DQRNN_training_1000eps/smoothed_scores_DQN_1.npy")
smoothed_scores_2 = np.load("simple_BP_DQRNN_training_1000eps/smoothed_scores_DQN_2.npy")
smoothed_scores_3 = np.load("simple_BP_DQRNN_training_1000eps/smoothed_scores_DQN_3.npy")
smoothed_scores_4 = np.load("simple_BP_DQRNN_training_1000eps/smoothed_scores_DQN_4.npy")

best_0 = np.argmax(smoothed_scores_0)
best_1 = np.argmax(smoothed_scores_1)
best_2 = np.argmax(smoothed_scores_2)
best_3 = np.argmax(smoothed_scores_3)
best_4 = np.argmax(smoothed_scores_4)

avg_steps_to_best = np.mean([best_0, best_1, best_2, best_3, best_4])

best_smoothed_scores_dqn = [smoothed_scores_0,
                            smoothed_scores_1,
                            smoothed_scores_2,
                            smoothed_scores_3,
                            smoothed_scores_4]
mean_smoothed_scores_dqn = np.mean(best_smoothed_scores_dqn, axis=0)
std_smoothed_scores = np.std(best_smoothed_scores_dqn, axis=0)

fig = plt.figure()
plt.plot(range(len(best_smoothed_scores_dqn[0])), mean_smoothed_scores_dqn, label = "Backpropamine")
plt.fill_between(range(len(best_smoothed_scores_dqn[0])),
                 np.nanpercentile(best_smoothed_scores_dqn, 2, axis=0),
                 np.nanpercentile(best_smoothed_scores_dqn, 97, axis=0), alpha=0.25)
plt.vlines(avg_steps_to_best, 0, 250, 'C0')
plt.ylim(0, 250)
plt.grid(True)


smoothed_scores_0 = np.load("Standard_DQRNN_training_1000eps/smoothed_scores_Standard_DQRNN_0.npy")
smoothed_scores_1 = np.load("Standard_DQRNN_training_1000eps/smoothed_scores_Standard_DQRNN_1.npy")
smoothed_scores_2 = np.load("Standard_DQRNN_training_1000eps/smoothed_scores_Standard_DQRNN_2.npy")
smoothed_scores_3 = np.load("Standard_DQRNN_training_1000eps/smoothed_scores_Standard_DQRNN_3.npy")
smoothed_scores_4 = np.load("Standard_DQRNN_training_1000eps/smoothed_scores_Standard_DQRNN_4.npy")

best_0 = np.argmax(smoothed_scores_0)
best_1 = np.argmax(smoothed_scores_1)
best_2 = np.argmax(smoothed_scores_2)
best_3 = np.argmax(smoothed_scores_3)
best_4 = np.argmax(smoothed_scores_4)

avg_steps_to_best = np.mean([best_0, best_1, best_2, best_3, best_4])

best_smoothed_scores_dqn = [smoothed_scores_0,
                            smoothed_scores_1,
                            smoothed_scores_2,
                            smoothed_scores_3,
                            smoothed_scores_4]
mean_smoothed_scores_dqn = np.mean(best_smoothed_scores_dqn, axis=0)
std_smoothed_scores = np.std(best_smoothed_scores_dqn, axis=0)


plt.plot(range(len(best_smoothed_scores_dqn[0])), mean_smoothed_scores_dqn, label = "Standard")
plt.fill_between(range(len(best_smoothed_scores_dqn[0])),
                 np.nanpercentile(best_smoothed_scores_dqn, 2, axis=0),
                 np.nanpercentile(best_smoothed_scores_dqn, 97, axis=0), alpha=0.25)
plt.vlines(avg_steps_to_best, 0, 250, 'orange')
plt.legend()
plt.xlabel("Training episodes")
plt.ylabel("Average smoothed score")
plt.title("Training performance BP DQRNN vs. standard DQRNN")

plt.show()