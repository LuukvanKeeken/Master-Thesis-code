import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

start = 0
end = 200
modified_parameter = "force_mag"
statistic = "median"

# CfC_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_64/{modified_parameter}/{statistic}s.npy")
# CfC_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_64/{modified_parameter}/stddevs.npy")
CfC_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_64/{modified_parameter}/percentages.npy")
CfC_evaluation_percentages = (CfC_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_evaluation_percentages[start:end], CfC_evaluation_statistic[start:end], CfC_evaluation_std[start:end], capsize=5, label = "CfC")
# elif statistic == "median":
#     ax.plot(CfC_evaluation_percentages[start:end], CfC_evaluation_statistic[start:end], label = "CfC_64")

# CfC_48_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_10_2024131_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_48/{modified_parameter}/{statistic}s.npy")
# CfC_48_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_10_2024131_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_48/{modified_parameter}/stddevs.npy")
# CfC_48_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_10_2024131_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_48/{modified_parameter}/percentages.npy")
# CfC_48_evaluation_percentages = (CfC_48_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_48_evaluation_percentages[start:end], CfC_48_evaluation_statistic[start:end], CfC_48_evaluation_std[start:end], capsize=5, label = "CfC_48")
# elif statistic == "median":
#     ax.plot(CfC_48_evaluation_percentages[start:end], CfC_48_evaluation_statistic[start:end], label = "CfC_48")

# CfC_32_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_8_2024131_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32/{modified_parameter}/{statistic}s.npy")
# CfC_32_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_8_2024131_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32/{modified_parameter}/stddevs.npy")
# CfC_32_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_8_2024131_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32/{modified_parameter}/percentages.npy")
# CfC_32_evaluation_percentages = (CfC_32_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_32_evaluation_percentages[start:end], CfC_32_evaluation_statistic[start:end], CfC_32_evaluation_std[start:end], capsize=5, label = "CfC_32")
# elif statistic == "median":
#     ax.plot(CfC_32_evaluation_percentages[start:end], CfC_32_evaluation_statistic[start:end], label = "CfC_32")

# LTC_AutoNCP_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_17_2024210_learningrate_0.001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_wiring_AutoNCP/{modified_parameter}/{statistic}s.npy")
# LTC_AutoNCP_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_17_2024210_learningrate_0.001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_wiring_AutoNCP/{modified_parameter}/stddevs.npy")
# LTC_AutoNCP_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_17_2024210_learningrate_0.001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_wiring_AutoNCP/{modified_parameter}/percentages.npy")
# LTC_AutoNCP_evaluation_percentages = (LTC_AutoNCP_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(LTC_AutoNCP_evaluation_percentages[start:end], LTC_AutoNCP_evaluation_statistic[start:end], LTC_AutoNCP_evaluation_std[start:end], capsize=5, label = "LTC_AutoNCP")
# elif statistic == "median":
#     ax.plot(LTC_AutoNCP_evaluation_percentages[start:end], LTC_AutoNCP_evaluation_statistic[start:end], label = "LTC_AutoNCP")

# best_LTC_AutoNCP_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_19_2024211_learningrate_0.005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_24_tausysextraction_True_wiring_AutoNCP/{modified_parameter}/{statistic}s.npy")
# best_LTC_AutoNCP_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_19_2024211_learningrate_0.005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_24_tausysextraction_True_wiring_AutoNCP/{modified_parameter}/stddevs.npy")
# best_LTC_AutoNCP_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_19_2024211_learningrate_0.005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_24_tausysextraction_True_wiring_AutoNCP/{modified_parameter}/percentages.npy")
# best_LTC_AutoNCP_evaluation_percentages = (best_LTC_AutoNCP_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(best_LTC_AutoNCP_evaluation_percentages[start:end], best_LTC_AutoNCP_evaluation_statistic[start:end], best_LTC_AutoNCP_evaluation_std[start:end], capsize=5, label = "best_LTC_AutoNCP")
# elif statistic == "median":
#     ax.plot(best_LTC_AutoNCP_evaluation_percentages[start:end], best_LTC_AutoNCP_evaluation_statistic[start:end], label = "best_LTC_AutoNCP")

best_fully_connected_LTC_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_92_2024214_learningrate_0.001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True/{modified_parameter}/{statistic}s.npy")
best_fully_connected_LTC_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_92_2024214_learningrate_0.001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True/{modified_parameter}/stddevs.npy")
best_fully_connected_LTC_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_92_2024214_learningrate_0.001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True/{modified_parameter}/percentages.npy")
best_fully_connected_LTC_evaluation_percentages = (best_fully_connected_LTC_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(best_fully_connected_LTC_evaluation_percentages[start:end], best_fully_connected_LTC_evaluation_statistic[start:end], best_fully_connected_LTC_evaluation_std[start:end], capsize=5, label = "best_fully_connected_LTC")
elif statistic == "median":
    ax.plot(best_fully_connected_LTC_evaluation_percentages[start:end], best_fully_connected_LTC_evaluation_statistic[start:end], label = "best_fully_connected_LTC")

CfC_neuromod_rand_035_selectiononvalranges = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_180_2024227_learningrate_5e-05_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.35, 0.35, 0.35]/{modified_parameter}/{statistic}s.npy")
CfC_neuromod_rand_035_selectiononvalranges_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_180_2024227_learningrate_5e-05_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.35, 0.35, 0.35]/{modified_parameter}/stddevs.npy")
CfC_neuromod_rand_035_selectiononvalranges_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_180_2024227_learningrate_5e-05_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.35, 0.35, 0.35]/{modified_parameter}/percentages.npy")
CfC_neuromod_rand_035_selectiononvalranges_percentages = (CfC_neuromod_rand_035_selectiononvalranges_percentages*100)-100
if statistic == "mean":
    ax.errorbar(CfC_neuromod_rand_035_selectiononvalranges_percentages[start:end], CfC_neuromod_rand_035_selectiononvalranges[start:end], CfC_neuromod_rand_035_selectiononvalranges_std[start:end], capsize=5, label = "CfC_neuromod_rand0.35_selectiononvalranges")
elif statistic == "median":
    ax.plot(CfC_neuromod_rand_035_selectiononvalranges_percentages[start:end], CfC_neuromod_rand_035_selectiononvalranges[start:end], label = "CfC_neuromod_rand0.35_selectiononvalranges")



# CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_153_2024222_learningrate_0.0001_selectiomethod_range_evaluatation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.5, 0.5, 0.5]/{modified_parameter}/{statistic}s.npy")
# CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_153_2024222_learningrate_0.0001_selectiomethod_range_evaluatation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.5, 0.5, 0.5]/{modified_parameter}/stddevs.npy")
# CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_153_2024222_learningrate_0.0001_selectiomethod_range_evaluatation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.5, 0.5, 0.5]/{modified_parameter}/percentages.npy")
# CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_percentages = (CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_percentages[start:end], CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_statistic[start:end], CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_std[start:end], capsize=5, label = "CfC_neuromod_rand0.5_3privparams_rangevalselection")
# elif statistic == "median":
#     ax.plot(CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_percentages[start:end], CfC_neuromod_rand_0_5_3privparams_rangevalselection_evaluation_statistic[start:end], label = "CfC_neuromod_rand0.5_3privparams_rangevalselection")


# CfC_neuromod_rand_0_5_3privilegedparams_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_147_2024220_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.5, 0.5, 0.5]/{modified_parameter}/{statistic}s.npy")
# CfC_neuromod_rand_0_5_3privilegedparams_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_147_2024220_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.5, 0.5, 0.5]/{modified_parameter}/stddevs.npy")
# CfC_neuromod_rand_0_5_3privilegedparams_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_147_2024220_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.5, 0.5, 0.5]/{modified_parameter}/percentages.npy")
# CfC_neuromod_rand_0_5_3privilegedparams_evaluation_percentages = (CfC_neuromod_rand_0_5_3privilegedparams_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_neuromod_rand_0_5_3privilegedparams_evaluation_percentages[start:end], CfC_neuromod_rand_0_5_3privilegedparams_evaluation_statistic[start:end], CfC_neuromod_rand_0_5_3privilegedparams_std[start:end], capsize=5, label = "CfC_neuromod_rand0.5_3privilegedparams")
# elif statistic == "median":
#     ax.plot(CfC_neuromod_rand_0_5_3privilegedparams_evaluation_percentages[start:end], CfC_neuromod_rand_0_5_3privilegedparams_evaluation_statistic[start:end], label = "CfC_neuromod_rand0.5_3privilegedparams")


# CfC_neuromod_rand_0_35_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_132_2024217_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.35, 0.35, 0.35, 0.35, 0.35]/{modified_parameter}/{statistic}s.npy")
# CfC_neuromod_rand_0_35_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_132_2024217_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.35, 0.35, 0.35, 0.35, 0.35]/{modified_parameter}/stddevs.npy")
# CfC_neuromod_rand_0_35_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_132_2024217_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.35, 0.35, 0.35, 0.35, 0.35]/{modified_parameter}/percentages.npy")
# CfC_neuromod_rand_0_35_evaluation_percentages = (CfC_neuromod_rand_0_35_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_neuromod_rand_0_35_evaluation_percentages[start:end], CfC_neuromod_rand_0_35_evaluation_statistic[start:end], CfC_neuromod_rand_0_35_evaluation_std[start:end], capsize=5, label = "CfC_neuromod_rand0.35")
# elif statistic == "median":
#     ax.plot(CfC_neuromod_rand_0_35_evaluation_percentages[start:end], CfC_neuromod_rand_0_35_evaluation_statistic[start:end], label = "CfC_neuromod_rand0.35")



# CfC_neuromod_rand_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_122_2024216_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.1, 0.1, 0.1, 0.1, 0.1]/{modified_parameter}/{statistic}s.npy")
# CfC_neuromod_rand_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_122_2024216_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.1, 0.1, 0.1, 0.1, 0.1]/{modified_parameter}/stddevs.npy")
# CfC_neuromod_rand_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_122_2024216_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated_randomization_params_[0.1, 0.1, 0.1, 0.1, 0.1]/{modified_parameter}/percentages.npy")
# CfC_neuromod_rand_evaluation_percentages = (CfC_neuromod_rand_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_neuromod_rand_evaluation_percentages[start:end], CfC_neuromod_rand_evaluation_statistic[start:end], CfC_neuromod_rand_evaluation_std[start:end], capsize=5, label = "CfC_neuromod_rand0.1")
# elif statistic == "median":
#     ax.plot(CfC_neuromod_rand_evaluation_percentages[start:end], CfC_neuromod_rand_evaluation_statistic[start:end], label = "CfC_neuromod_rand0.1")

# CfC_neuromod_norand_randomencoder_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_157_2024225_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/{statistic}s.npy")
# CfC_neuromod_norand_randomencoder_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_157_2024225_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/stddevs.npy")
# CfC_neuromod_norand_randomencoder_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_157_2024225_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/percentages.npy")
# CfC_neuromod_norand_randomencoder_evaluation_percentages = (CfC_neuromod_norand_randomencoder_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_neuromod_norand_randomencoder_evaluation_percentages[start:end], CfC_neuromod_norand_randomencoder_evaluation_statistic[start:end], CfC_neuromod_norand_randomencoder_evaluation_std[start:end], capsize=5, label = "CfC_neuromod_norand_randomencoder")
# elif statistic == "median":
#     ax.plot(CfC_neuromod_norand_randomencoder_evaluation_percentages[start:end], CfC_neuromod_norand_randomencoder_evaluation_statistic[start:end], label = "CfC_neuromod_norand_randomencoder")

# CfC_neuromod_norand_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_174_2024227_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/{statistic}s.npy")
# CfC_neuromod_norand_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_174_2024227_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/stddevs.npy")
# CfC_neuromod_norand_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_174_2024227_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/percentages.npy")
# CfC_neuromod_norand_evaluation_percentages = (CfC_neuromod_norand_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_neuromod_norand_evaluation_percentages[start:end], CfC_neuromod_norand_evaluation_statistic[start:end], CfC_neuromod_norand_evaluation_std[start:end], capsize=5, label = "CfC_neuromod_norand")
# elif statistic == "median":
#     ax.plot(CfC_neuromod_norand_evaluation_percentages[start:end], CfC_neuromod_norand_evaluation_statistic[start:end], label = "CfC_neuromod_norand")

# CfC_neuromod_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_117_2024216_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/{statistic}s.npy")
# CfC_neuromod_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_117_2024216_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/stddevs.npy")
# CfC_neuromod_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_117_2024216_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_neuromodulated/{modified_parameter}/percentages.npy")
# CfC_neuromod_evaluation_percentages = (CfC_neuromod_evaluation_percentages*100)-100
# if statistic == "mean":
#     ax.errorbar(CfC_neuromod_evaluation_percentages[start:end], CfC_neuromod_evaluation_statistic[start:end], CfC_neuromod_evaluation_std[start:end], capsize=5, label = "CfC_neuromod_norand")
# elif statistic == "median":
#     ax.plot(CfC_neuromod_evaluation_percentages[start:end], CfC_neuromod_evaluation_statistic[start:end], label = "CfC_neuromod_norand")

best_fully_connected_CfC_validationrangeselection_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_192_202431_learningrate_0.0001_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_48_tausysextraction_True_mode_pure/{modified_parameter}/{statistic}s.npy")
best_fully_connected_CfC_validationrangeselection_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_192_202431_learningrate_0.0001_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_48_tausysextraction_True_mode_pure/{modified_parameter}/stddevs.npy")
best_fully_connected_CfC_validationrangeselection_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_192_202431_learningrate_0.0001_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_48_tausysextraction_True_mode_pure/{modified_parameter}/percentages.npy")
best_fully_connected_CfC_validationrangeselection_evaluation_percentages = (best_fully_connected_CfC_validationrangeselection_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(best_fully_connected_CfC_validationrangeselection_evaluation_percentages[start:end], best_fully_connected_CfC_validationrangeselection_evaluation_statistic[start:end], best_fully_connected_CfC_validationrangeselection_evaluation_std[start:end], capsize=5, label = "best_fully_connected_CfC_validationrangeselection")
elif statistic == "median":
    ax.plot(best_fully_connected_CfC_validationrangeselection_evaluation_percentages[start:end], best_fully_connected_CfC_validationrangeselection_evaluation_statistic[start:end], label = "best_fully_connected_CfC_validationrangeselection")

best_fully_connected_LTC_validationrangeselection_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_206_202431_learningrate_0.0005_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_48_tausysextraction_True/{modified_parameter}/{statistic}s.npy")
best_fully_connected_LTC_validationrangeselection_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_206_202431_learningrate_0.0005_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_48_tausysextraction_True/{modified_parameter}/stddevs.npy")
best_fully_connected_LTC_validationrangeselection_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/LTC_a2c_result_206_202431_learningrate_0.0005_selectiomethod_range_evaluation_all_params_gamma_0.99_trainingmethod_standard_numneurons_48_tausysextraction_True/{modified_parameter}/percentages.npy")
best_fully_connected_LTC_validationrangeselection_evaluation_percentages = (best_fully_connected_LTC_validationrangeselection_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(best_fully_connected_LTC_validationrangeselection_evaluation_percentages[start:end], best_fully_connected_LTC_validationrangeselection_evaluation_statistic[start:end], best_fully_connected_LTC_validationrangeselection_evaluation_std[start:end], capsize=5, label = "best_fully_connected_LTC_validationrangeselection")
elif statistic == "median":
    ax.plot(best_fully_connected_LTC_validationrangeselection_evaluation_percentages[start:end], best_fully_connected_LTC_validationrangeselection_evaluation_statistic[start:end], label = "best_fully_connected_LTC_validationrangeselection")


best_fully_connected_CfC_evaluation_statistic = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_88_2024214_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_pure/{modified_parameter}/{statistic}s.npy")
best_fully_connected_CfC_evaluation_std = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_88_2024214_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_pure/{modified_parameter}/stddevs.npy")
best_fully_connected_CfC_evaluation_percentages = np.load(f"../LTC_A2C/evaluation_results/CfC_a2c_result_88_2024214_learningrate_0.0005_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard_numneurons_32_tausysextraction_True_mode_pure/{modified_parameter}/percentages.npy")
best_fully_connected_CfC_evaluation_percentages = (best_fully_connected_CfC_evaluation_percentages*100)-100
if statistic == "mean":
    ax.errorbar(best_fully_connected_CfC_evaluation_percentages[start:end], best_fully_connected_CfC_evaluation_statistic[start:end], best_fully_connected_CfC_evaluation_std[start:end], capsize=5, label = "best_fully_connected_CfC")
elif statistic == "median":
    ax.plot(best_fully_connected_CfC_evaluation_percentages[start:end], best_fully_connected_CfC_evaluation_statistic[start:end], label = "best_fully_connected_CfC")

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