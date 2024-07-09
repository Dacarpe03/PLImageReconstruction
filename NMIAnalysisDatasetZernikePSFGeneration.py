#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nmi_analysis_constants import NUMPY_SUFFIX, \
                                  PATHS_DICTIONARY

from data_utils import generate_zernike_psf_complex_fields,\
                       generate_psf_complex_fields, \
                       compute_output_fluxes_from_complex_field, \
                       compute_output_fluxes_from_complex_field_using_arbitrary_transfer_matrix, \
                       compute_lp_modes_from_complex_field


# ### Output fluxes

# In[2]:
dataset_name = "NMI_ANALYSIS_BIG_9M"
ds_info = PATHS_DICTIONARY[dataset_name]

print(f"Generating {dataset_name}")

# Save intensities
psf_path = ds_info["complex_fields_path"]

zernike_coeffs_path = ds_info["zernike_mode_coefficients_path"]
lp_coeffs_path = ds_info["lp_modes_path"]
flux_path = ds_info["output_fluxes_path"]
n_zernike_modes = len(ds_info["zernike_mode_indexes"])
coefficients_range = ds_info["zernike_coefficients_range"]
coefficients_can_be_negative = ds_info["zernike_can_be_negative"]
n_samples = ds_info["n_samples"]

print()
print("PSF:", psf_path)
print("Zernike:", zernike_coeffs_path)
print("Flux:", flux_path)
print("LP:", lp_coeffs_path)
print(ds_info["zernike_mode_indexes"])
print(coefficients_range)
print("    Generating PSFs")
generate_zernike_psf_complex_fields(psf_path,
                                    zernike_coeffs_path,
                                    zernike_modes=n_zernike_modes,
                                    coefficients_range=coefficients_range,
                                    coefficients_can_be_negative=coefficients_can_be_negative,
                                    n_samples=n_samples,
                                    overwrite=False,
                                    save_complex_fields=True,
                                    save_intensities=False)

print("    Generating Fluxes and LP coefficients")
compute_output_fluxes_from_complex_field(psf_path,
                                         lp_coeffs_path,
                                         flux_path,
                                         overwrite=False)

