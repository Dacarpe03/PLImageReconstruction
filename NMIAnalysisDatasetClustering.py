#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nmi_analysis_constants import PATHS_DICTIONARY
from data_utils import save_numpy_array
from plot_utils import plot_clusters_from_labels, \
                       plot_cluster_labels_count, \
                       plot_grid_clusters, \
                       plot_kneighbours,\
                       get_number_of_clusters

from sklearn.cluster import KMeans

import numpy as np

import os
# In[3]:


number_of_clusters = [8000, 10000, 20000]

data_keys = ["umap_intensities_path",
             "zernike_mode_coefficients_path",
             "output_fluxes_path"]

labels_keys = ["complex_fields_labels_path",
                "zernike_mode_coefficients_labels_path",
                "output_fluxes_labels_path"]

key = "NMI_ANALYSIS_BIG_9M"
minidataset_dict = PATHS_DICTIONARY[key]

for n_clusters in number_of_clusters:
    cluster_suffix = f"_{n_clusters}kmeans.npy"
    print(f"Computing {n_clusters} K-Means cluters for dataset {key}")

    for data_path_key, labels_path_key in zip(data_keys, labels_keys):
            
        print(f"    Computing {data_path_key}")
        data_path = minidataset_dict[data_path_key]
        print(data_path)
        labels_path = minidataset_dict[labels_path_key]
        
        complete_labels_path = f"{labels_path}{cluster_suffix}"

        if not os.path.isfile(complete_labels_path):
            data = np.load(data_path)
            if data_path_key == "lp_modes_path":
                data = data.reshape(75000, 38)
                    
            kmeans = KMeans(n_clusters=n_clusters,
                            n_init=100)
            data_labels = kmeans.fit_predict(data)
            save_numpy_array(data_labels, complete_labels_path, single_precision=False)
        else:
            print(f"{complete_labels_path} already exists")

