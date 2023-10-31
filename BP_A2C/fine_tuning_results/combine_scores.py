import numpy as np

f5_best_episodes_avg, f5_best_episodes_stddev = np.load("FIRST5_BP_A2C_RNN_fine_tuning_result_8_20231016/best_episodes.npy")
f5_b_r_avg, f5_b_r_stddev = np.load("FIRST5_BP_A2C_RNN_fine_tuning_result_8_20231016/best_rewards.npy")

o5_b_e_avg, o5_b_e_stddev = np.load("OTHER5_BP_A2C_RNN_fine_tuning_result_4_20231017/best_episodes.npy")
o5_b_r_avg, o5_b_r_stddev = np.load("OTHER5_BP_A2C_RNN_fine_tuning_result_4_20231017/best_rewards.npy")


f5_data = np.random.rand(10, 5)
o5_data = np.random.rand(10, 5)


f5_best_episodes_stddev = np.std(f5_data, axis = 1)
f5_best_episodes_avg = np.mean(f5_data, axis = 1)
o5_b_e_stddev = np.std(o5_data, axis = 1)
o5_b_e_avg = np.mean(o5_data, axis = 1)

combined_episodes_avg = []
combined_rewards_avg = []
combined_episodes_stddev = []
combined_rewards_stddev = []
for i in range(10):
    
    combined_episodes_avg.append((f5_best_episodes_avg[i] + o5_b_e_avg[i])/2)
    combined_rewards_avg.append((f5_b_r_avg[i] + o5_b_r_avg[i])/2)
    
    eps_stddev_first_term = (4 * np.power(f5_best_episodes_stddev[i], 2) + 4 * np.power(o5_b_e_stddev[i], 2))/(5 + 5 - 1)
    eps_stddev_second_term = (5*5*np.power(f5_best_episodes_avg[i] - o5_b_e_avg[i], 2))/((5+5)*(5+5-1))
    
    combined_episodes_stddev.append(np.sqrt(eps_stddev_first_term + eps_stddev_second_term))

print(combined_episodes_avg)
print()
print(combined_episodes_stddev)
print(np.std([1, 5, 6, 4, 3, 7, 4, 3, 1, 1, 1, 54, 6, 4, 35, 72, 4, 31, 1, 1]))




