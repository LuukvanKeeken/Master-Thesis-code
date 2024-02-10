import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

def plot_one_episode(all_tau_sys, pole_length_mods, results_dir, neuron_idx = 0, episode_idx = None):
    
    episodes = []
    if not episode_idx:
        num_episodes = len(all_tau_sys[0])
        rand_episode = random.randint(0, num_episodes - 1)
        episode_idx = rand_episode
    for i in range(len(all_tau_sys)):
        episodes.append(all_tau_sys[i][episode_idx])
    

    for i, pole_length_mod in enumerate(pole_length_mods):
        neuron_vals = [x[neuron_idx] for x in episodes[i]]
        plt.plot(neuron_vals, label = pole_length_mod)

    plt.xlabel("Time step")
    plt.ylabel("Tau system value")
    plt.legend()
    plt.title(f"Tau system values for neuron {neuron_idx} in episode {episode_idx}")
    plt.savefig(f"Master_Thesis_Code/LTC_A2C/time_constants/{results_dir}/tau_sys_neuron_{neuron_idx}_episode_{episode_idx}.png")
    plt.show()



def plot_average_episode(all_tau_sys, pole_length_mods, results_dir, neuron_idx = 0):
    averaged_episodes = []
    std_devs = []
    for i in range(len(all_tau_sys)):
        min_episode_length = min([len(x) for x in all_tau_sys[i]])
        trimmed_tau_sys = [x[:min_episode_length] for x in all_tau_sys[i]]

        averaged_episodes.append(np.mean(trimmed_tau_sys, axis = 0))
        std_devs.append(np.std(trimmed_tau_sys, axis = 0))


    for i, pole_length_mod in enumerate(pole_length_mods):
        neuron_vals = [x[neuron_idx] for x in averaged_episodes[i]]
        std_dev = [x[neuron_idx] for x in std_devs[i]]
        plt.plot(neuron_vals, label = pole_length_mod)
        plt.fill_between(range(len(neuron_vals)), np.array(neuron_vals) - np.array(std_dev), np.array(neuron_vals) + np.array(std_dev), alpha = 0.3)
    
    plt.xlabel("Time step")
    plt.ylabel("Tau system value")
    plt.legend()
    plt.title(f"Average tau system values for neuron {neuron_idx}")
    plt.savefig(f"Master_Thesis_Code/LTC_A2C/time_constants/{results_dir}/average_tau_sys_neuron_{neuron_idx}.png")
    plt.show()


    




results_dir = f"CfC_a2c_result_10_202428_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_fullyconnected_True_mode_pure"

with open(f"Master_Thesis_Code/LTC_A2C/time_constants/{results_dir}/all_tau_sys.pkl", 'rb') as f:
    all_tau_sys = pickle.load(f)

tau_sys = all_tau_sys["all_tau_sys"]
pole_length_mods = all_tau_sys["pole_length_mods"]

plot_one_episode(tau_sys, pole_length_mods, results_dir, neuron_idx = 10, episode_idx = 27)
# plot_average_episode(tau_sys, pole_length_mods, results_dir, neuron_idx = 12)