#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nmi_analysis_constants import PATHS_DICTIONARY
from psf_constants import COMPLEX_NUMBER_NORMALIZATION_CONSTANT
import numpy as np
from data_utils import save_numpy_array


# In[2]:


def process_fc_complex_field_intensity(filepath):
    
    complex_arrays = np.load(filepath)

    complex_arrays = complex_arrays/COMPLEX_NUMBER_NORMALIZATION_CONSTANT
    intensities_arrays = np.abs(complex_arrays)**2
    print(intensities_arrays.shape)
    start_row = (128 - 64) // 2
    end_row = start_row + 64
    start_col = (128 - 64) // 2
    end_col = start_col + 64

    intensities_arrays = intensities_arrays.reshape(5000, 128*128)
    return intensities_arrays


# In[3]:


for key, minidataset_dict in PATHS_DICTIONARY.items():
    print(key)
    intensities = process_fc_complex_field_intensity(minidataset_dict['complex_fields_path'])
    save_numpy_array(intensities, minidataset_dict['intensities_file_path'])

