#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nmi_analysis_constants import PATHS_DICTIONARY
from psf_constants import COMPLEX_NUMBER_NORMALIZATION_CONSTANT
import numpy as np
from data_utils import save_numpy_array


# In[2]:


def process_fc_complex_field_intensity(filepath, start, end):
    
    complex_arrays = np.load(filepath)[start:end]

    complex_arrays = complex_arrays/COMPLEX_NUMBER_NORMALIZATION_CONSTANT
    intensities_arrays = np.abs(complex_arrays)**2
    print(intensities_arrays.shape)
    start_row = (128 - 64) // 2
    end_row = start_row + 64
    start_col = (128 - 64) // 2
    end_col = start_col + 64

    intensities_arrays = intensities_arrays.reshape(10000, 128*128)
    return intensities_arrays


# In[3]:


key = "NMI_ANALYSIS_BIG_9M"
minidataset_dict = PATHS_DICTIONARY[key]
start = 0
end = 10000
for i in range(10):
	intensities = process_fc_complex_field_intensity(minidataset_dict['complex_fields_path'], start, end)
	save_numpy_array(intensities, f"{str(i)}{minidataset_dict['intensities_file_path']}")
	start += 10000
	end += 10000

