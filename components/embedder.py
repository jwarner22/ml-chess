# %%
import umap
import pandas as pd
import matplotlib.pyplot as plt

# %%
features = pd.read_csv('./features.zip')

# %%
labels = pd.read_csv('./labels.zip')
labels = labels.iloc[:,0]

# %%
trans = umap.UMAP(n_neighbors=10, n_components = 2, random_state=42, verbose=True).fit(features, y=labels)


# %%

plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=labels,  cmap='Spectral')
plt.title('Embedding of the training set by UMAP', fontsize=24);