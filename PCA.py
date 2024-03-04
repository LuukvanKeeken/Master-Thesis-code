from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

param = 'force_mag'

if param == 'pole_length':
    mods = np.linspace(0.1, 20.0, 500)*0.5
elif param == 'pole_mass':
    mods = np.linspace(5.0, 20.0, 500)*0.1
elif param == 'force_mag':
    mods = np.linspace(0.2, 6.0, 500)*10.0


encoder_outputs = np.load(f'Master_Thesis_Code/encoder_outputs_for_PCA/encoder_outputs_{param}_UNtrainedencodernotonrange.npy')

pca = PCA(n_components=2)
principal_comps = pca.fit_transform(encoder_outputs)

print(pca.explained_variance_ratio_)

pc1 = principal_comps[:, 0]
pc2 = principal_comps[:, 1]


plt.scatter(pc1, pc2, c=mods)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label=f'{param}')
plt.title(f'First two principal components of original env untrained encoder outputs for range of {param}')
plt.show()
