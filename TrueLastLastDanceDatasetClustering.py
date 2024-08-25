#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ami_analysis_constants import ARBITRARY_EXPERIMENT_PATHS_DICTIONARY
from data_utils import save_numpy_array
from plot_utils import plot_clusters_from_labels, \
                       plot_cluster_labels_count, \
                       plot_grid_clusters, \
                       plot_kneighbours,\
                       get_number_of_clusters

from sklearn.cluster import KMeans

import numpy as np


# In[3]:

data_keys = ["umap_complex_fields_path",
             "umap_intensities_path",
             "zernike_mode_coefficients_path",
             "lp_modes_path",
             "output_fluxes_path",
             "complex_output_fluxes_path"]

labels_keys = ["complex_fields_labels_path",
                "intensities_labels_path",
                "zernike_mode_coefficients_labels_path",
                "lp_modes_labels_path",
                "output_fluxes_labels_path",
                "complex_output_fluxes_labels_path"]

for key, minidataset_dict in ARBITRARY_EXPERIMENT_PATHS_DICTIONARY.items():
    number_of_clusters = minidataset_dict['n_clusters']
    n_samples = minidataset_dict['n_samples']
    for n_clusters in number_of_clusters:
        cluster_suffix = f"_{n_clusters}kmeans.npy"
        print(f"Computing {n_clusters} K-Means cluters for dataset {key}")

        for data_path_key, labels_path_key in zip(data_keys, labels_keys):
            
            print(f"    Computing {data_path_key}")
            data_path = minidataset_dict[data_path_key]
            print(data_path)
            labels_path = minidataset_dict[labels_path_key]

            data = np.load(data_path)
            if data_path_key in ["lp_modes_path", "complex_output_fluxes_path"]:
                data = data.reshape(n_samples, 38)
                
            kmeans = KMeans(n_clusters=n_clusters,
                            n_init=200)
            data_labels = kmeans.fit_predict(data)
            
            complete_labels_path = f"{labels_path}{cluster_suffix}"
            save_numpy_array(data_labels, complete_labels_path, single_precision=False)