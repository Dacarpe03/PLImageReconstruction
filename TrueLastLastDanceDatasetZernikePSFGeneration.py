#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ami_analysis_constants import NUMPY_SUFFIX, \
                                   ARBITRARY_EXPERIMENT_PATHS_DICTIONARY

from psf_constants import ARBITRARY_MATRIX
from data_utils import generate_zernike_psf_complex_fields,\
                       generate_psf_complex_fields, \
                       compute_output_fluxes_from_complex_field, \
                       compute_output_fluxes_from_complex_field_using_arbitrary_transfer_matrix, \
                       compute_lp_modes_from_complex_field


# ### Output fluxes

# In[2]:


for dataset_name, ds_info in ARBITRARY_EXPERIMENT_PATHS_DICTIONARY.items():
    print()
    print(f"Generating {dataset_name}")
    psf_path = ds_info["complex_fields_path"]
    zernike_coeffs_path = ds_info["zernike_mode_coefficients_path"]
    lp_coeffs_path = ds_info["lp_modes_path"]
    flux_path = ds_info["output_fluxes_path"]
    complex_flux_path = ds_info["complex_output_fluxes_path"]
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
                                        overwrite=False)

    print("    Generating Fluxes and LP coefficients")
    compute_output_fluxes_from_complex_field_using_arbitrary_transfer_matrix(psf_path,
                                                                             lp_coeffs_path,
                                                                             flux_path,
                                                                             ARBITRARY_MATRIX,
                                                                             complex_output_fluxes_file_path=complex_flux_path,
                                                                             overwrite=False)