import numpy as np
import matplotlib.pyplot as plt

param = 'force_mag'

if param == 'pole_length':
    orig_val = 0.5
elif param == 'pole_mass':
    orig_val = 0.1
elif param == 'force_mag':
    orig_val = 10.0

# no_range_train = np.load(f'../encoder_outputs_for_PCA/encoder_outputs_{param}_norangetrain_3privparams.npy')
# no_range_train_5 = np.load(f'../encoder_outputs_for_PCA/encoder_outputs_{param}_norangetrain_5privparams.npy')
no_range_trained = np.load(f'../encoder_outputs_for_PCA/encoder_outputs_{param}_trainedencodernotonrange.npy')
no_range_UNtrained = np.load(f'../encoder_outputs_for_PCA/encoder_outputs_{param}_UNtrainedencodernotonrange.npy')
RF_02 = np.load(f'../encoder_outputs_for_PCA/encoder_outputs_{param}_RF0.2.npy')
RF_035 = np.load(f'../encoder_outputs_for_PCA/encoder_outputs_{param}_RF0.35.npy')
RF_05 = np.load(f'../encoder_outputs_for_PCA/encoder_outputs_{param}_RF0.5.npy')
fig, ax = plt.subplots()

# ax.errorbar(np.linspace(0.1, 20.0, 500), np.mean(test, axis=1), np.std(test, axis=1))
ax.plot(np.linspace(0.1, 20.0, 500)*orig_val, np.mean(RF_02, axis=1), label='20% training range')
ax.plot(np.linspace(0.1, 20.0, 500)*orig_val, np.mean(RF_035, axis=1), label='35% training range')
ax.plot(np.linspace(0.1, 20.0, 500)*orig_val, np.mean(RF_05, axis=1), label='50% training range')
ax.plot(np.linspace(0.1, 20.0, 500)*orig_val, np.mean(no_range_UNtrained, axis=1), label='Orig range, untrained')
ax.plot(np.linspace(0.1, 20.0, 500)*orig_val, np.mean(no_range_trained, axis=1), label='Orig range, trained')

ax.set_title(f'Mean encoder output activation over range of {param}, other params fixed')
ax.set_xlabel(f'{param}')
ax.set_ylabel('Mean activation')
ax.legend()

plt.show()