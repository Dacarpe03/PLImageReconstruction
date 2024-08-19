#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ami_analysis_constants import PATHS_DICTIONARY
from data_utils import save_numpy_array
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# # UMAP

# In[3]:


import umap

def flatten_complex_fields(complex_fields):
    real_part = np.real(complex_fields)
    imag_part = np.imag(complex_fields)

    # Flatten the real and imaginary parts and concatenate them
    flattened_complex_fields = np.concatenate(
        (real_part.reshape(real_part.shape[0], -1), imag_part.reshape(imag_part.shape[0], -1)), 
        axis=1
    )
    return flattened_complex_fields

# ### Intensities

# In[4]:


a = np.array([[2+1j,3-2j],[4-4j, 8+8j]])
a = flatten_complex_fields(a)
print(a)

for key, minidataset_dict in PATHS_DICTIONARY.items():

    intensities = np.load(minidataset_dict['intensities_file_path'])
    umap_reducer = umap.UMAP(n_neighbors=15,
                             min_dist=0.3,
                             n_components=19,
                             metric='euclidean')
    umap_intensities = umap_reducer.fit_transform(intensities)
    save_numpy_array(umap_intensities, minidataset_dict["umap_intensities_path"])


    complex_fields = np.load(minidataset_dict['complex_fields_path'])
    complex_fields = flatten_complex_fields(complex_fields)
    umap_reducer = umap.UMAP(n_neighbors=15,
                             min_dist=0.3,
                             n_components=19,
                             metric='euclidean')

    umap_complex_fields = umap_reducer.fit_transform(complex_fields)
    save_numpy_array(umap_complex_fields, minidataset_dict["umap_complex_fields_path"])

