#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nmi_analysis_constants import PATHS_DICTIONARY
from data_utils import save_numpy_array
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# # UMAP

# In[3]:


import umap


# ### Intensities

# In[4]:
import tensorflow as tf
 
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

key = "NMI_ANALYSIS_BIG_9M"
minidataset_dict = PATHS_DICTIONARY[key]

intensities = np.load(minidataset_dict['intensities_file_path'])

umap_reducer = umap.UMAP(n_neighbors=15,
                         min_dist=0.3,
                         n_components=19,
                         metric='euclidean')
umap_intensities = umap_reducer.fit_transform(intensities)
save_numpy_array(umap_intensities, minidataset_dict["umap_intensities_path"])

