#!/usr/bin/env python
# coding: utf-8

# Dimensionality reduction for predicted PSF


from psf_constants import FC_PREDICTED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, \
                          FC_PREDICTED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, \
                          FC_PREDICTED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, \
                          FC_PREDICTED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, \
                          FC_PREDICTED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, \
                          LOW_DIMENSIONALITY_PREDICTED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH, \
                          LOW_DIMENSIONALITY_PREDICTED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH, \
                          LOW_DIMENSIONALITY_PREDICTED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH, \
                          LOW_DIMENSIONALITY_PREDICTED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH, \
                          LOW_DIMENSIONALITY_PREDICTED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH, \
                          PSF_TRAIN_FILE_SUFFIXES, \
                          NUMPY_SUFFIX

import umap
import numpy as np


PATH_DICTIONARY = {
    "TR2": [FC_PREDICTED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, 
            LOW_DIMENSIONALITY_PREDICTED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH],
    "TR5": [FC_PREDICTED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, 
            LOW_DIMENSIONALITY_PREDICTED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH],
    "TR9": [FC_PREDICTED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, 
            LOW_DIMENSIONALITY_PREDICTED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH],
    "TR14": [FC_PREDICTED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, 
             LOW_DIMENSIONALITY_PREDICTED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH],
    "TR20": [FC_PREDICTED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX,
             LOW_DIMENSIONALITY_PREDICTED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH]
}


for dataset_name, ds_info in PATH_DICTIONARY.items():
    
    reducer = umap.UMAP(n_neighbors=500,
                        min_dist=0.3,
                        n_components=3,
                        metric='euclidean')
    psf_path = ds_info[0]
    embedding_path = ds_info[1]
    
    if dataset_name.startswith("TR"):
        
        print("    Generating UMAP for", psf_path)
        psf_list = []
        for fnumber in PSF_TRAIN_FILE_SUFFIXES:
            psf_subpath = f"{psf_path}{fnumber}{NUMPY_SUFFIX}"
            psf_sublist = np.load(psf_subpath)
            psf_list.append(psf_sublist)
            
        psf_data = np.concatenate(psf_list)
        psf_list = []
        embedding = reducer.fit_transform(psf_data)
        print(embedding.shape)
        np.save(embedding_path, embedding)
        
    else:
        print("    Generating UMAP for", psf_path)
        psf_data = np.load(psf_path)
        embedding = reducer.fit_transform(psf_data)
        np.save(embedding_path, embedding)