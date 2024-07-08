NUMPY_SUFFIX = ".npy"

HOME = "/suphys/dcar0877"

PSF_MINIDATASET_PATH = f"{HOME}/DaniProjects/SAIL/PhotonicLanternProjects/Data/PSFReconstructionMinidataset"

ZERNIKE_COMPLEX_FIELDS_FILE_NAME = "zernike_complex_fields"
ZERNIKE_INTENSITIES_FILE_NAME = "zernike_intensities"
ZERNIKE_MODE_COEFFICIENTS_FILE_NAME = "zernike_mode_coefficients"
PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME = "processed_zernike_mode_coefficients"
ZERNIKE_OUTPUT_FLUXES_FILE_NAME = "zernike_output_fluxes"
PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME = "processed_zernike_output_fluxes"
ZERNIKE_PSF_LP_MODES_FILE_NAME = "lp_modes_from_zernike_psf"
PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME = "processed_lp_modes_from_zernike_psf"

# Folders paths
ZERNIKE_COMPLEX_FIELDS_MINIDATASET_FOLDER_PATH = f"{PSF_MINIDATASET_PATH}/ZernikeComplexFields"
ZERNIKE_MODES_MINIDATASET_FOLDER_PATH = f"{PSF_MINIDATASET_PATH}/ZernikeModeCoefficients"
ZERNIKE_PSF_LP_MODES_MINIDATASET_FOLDER_PATH = f"{PSF_MINIDATASET_PATH}/LPModesFromZernikePSF"
ZERNIKE_OUTPUT_FLUXES_MINIDATASET_FOLDER_PATH = f"{PSF_MINIDATASET_PATH}/ZernikeOutputFluxes"

PROCESSED_ZERNIKE_MODE_COEFFICIENTS_MINIDATSET_PATH = f"{PSF_MINIDATASET_PATH}/ProcessedZernikeModeCoefficients"
PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH = f"{PSF_MINIDATASET_PATH}/ProcessedLPModesFromZernikePSF"
PROCESSED_ZERNIKE_COMPLEX_FIELDS_MINIDATASET_PATH = f"{PSF_MINIDATASET_PATH}/ProcessedZernikeComplexFields"
PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH = f"{PSF_MINIDATASET_PATH}/ProcessedZernikeOutputFluxes"

# 2 modes Files paths
ZERNIKE_2M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_MINIDATASET_FOLDER_PATH}/2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH = f"{ZERNIKE_MODES_MINIDATASET_FOLDER_PATH}/2M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_LP_MODES_MINIDATASET_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_MINIDATASET_FOLDER_PATH}/2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_MINIDATASET_FOLDER_PATH}/2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_2M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_MINIDATASET_PATH}/2M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_2M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/pca_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/umap_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/pca_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/umap_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/pca_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/umap_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

# 5 modes Files paths
ZERNIKE_5M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_MINIDATASET_FOLDER_PATH}/5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH = f"{ZERNIKE_MODES_MINIDATASET_FOLDER_PATH}/5M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_LP_MODES_MINIDATASET_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_MINIDATASET_FOLDER_PATH}/5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_MINIDATASET_FOLDER_PATH}/5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_5M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_MINIDATASET_PATH}/5M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_5M_PSF_ZERNIKE_MODES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_MINIDATSET_PATH}/5M_pca_{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_ZERNIKE_MODES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_MINIDATSET_PATH}/5M_umap_{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/5M_pca_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/5M_umap_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/5M_pca_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/5M_umap_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/5M_pca_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/5M_umap_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

# 9 modes Files paths
ZERNIKE_9M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_MINIDATASET_FOLDER_PATH}/9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH = f"{ZERNIKE_MODES_MINIDATASET_FOLDER_PATH}/9M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_LP_MODES_MINIDATASET_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_MINIDATASET_FOLDER_PATH}/9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_MINIDATASET_FOLDER_PATH}/9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_MINIDATASET_PATH}/9M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_PSF_ZERNIKE_MODES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_MINIDATSET_PATH}/9M_pca_{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_ZERNIKE_MODES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_MINIDATSET_PATH}/9M_umap_{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/9M_pca_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/9M_umap_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/9M_pca_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/9M_umap_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/9M_pca_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/9M_umap_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

# 14 modes Files paths
ZERNIKE_14M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_MINIDATASET_FOLDER_PATH}/14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH = f"{ZERNIKE_MODES_MINIDATASET_FOLDER_PATH}/14M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_LP_MODES_MINIDATASET_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_MINIDATASET_FOLDER_PATH}/14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_MINIDATASET_FOLDER_PATH}/14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_14M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_MINIDATASET_PATH}/14M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_14M_PSF_ZERNIKE_MODES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_MINIDATSET_PATH}/14M_pca_{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_ZERNIKE_MODES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_MINIDATSET_PATH}/14M_umap_{PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/14M_pca_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_PSF_LP_MODES_MINIDATASET_PATH}/14M_umap_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/14M_pca_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/14M_umap_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/14M_pca_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH= f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_MINIDATASET_PATH}/14M_umap_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"



PATHS_DICTIONARY = {
	"MINI_2M":{
		"complex_field_path": ZERNIKE_2M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_2M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH,
		"pca_intensities_path": ZERNIKE_2M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH,
		"umap_intensities_path": ZERNIKE_2M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_2M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH,
		"lp_modes_path": ZERNIKE_2M_PSF_LP_MODES_MINIDATASET_FILE_PATH,
		"pca_lp_modes_path": ZERNIKE_2M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH,
		"umap_lp_modes_path": ZERNIKE_2M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH,
		"pca_output_fluxes_path": ZERNIKE_2M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH,
		"umap_output_fluxes_path": ZERNIKE_2M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH,
		"output_fluxes": ZERNIKE_2M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH,
		"zernike_mode_indexes": [2, 3],
		"zernike_coefficients_range": [[1.8, 2],
									   [1.8, 2]],

		"zernike_can_be_negative": [True,
									True,
									True,
									True,
									True],
		"labels_dictionary":{
			"pp":1,
			"pn":2,
			"np":3,
			"nn":4
		},
		"n_samples": 1000
	},

	"MINI_5M":{
		"complex_field_path": ZERNIKE_5M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_5M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH,
		"pca_intensities_path": ZERNIKE_5M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH,
		"umap_intensities_path": ZERNIKE_5M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_5M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH,
		"pca_zernike_mode_coefficients_path": ZERNIKE_5M_PSF_ZERNIKE_MODES_PCA_MINIDATASET_FILE_PATH,
		"umap_zernike_mode_coefficients_path": ZERNIKE_5M_PSF_ZERNIKE_MODES_UMAP_MINIDATASET_FILE_PATH,
		"lp_modes_path": ZERNIKE_5M_PSF_LP_MODES_MINIDATASET_FILE_PATH,
		"pca_lp_modes_path": ZERNIKE_5M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH,
		"umap_lp_modes_path": ZERNIKE_5M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH,
		"pca_output_fluxes_path": ZERNIKE_5M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH,
		"umap_output_fluxes_path": ZERNIKE_5M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH,
		"output_fluxes": ZERNIKE_5M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6],
		"zernike_coefficients_range": [[1.8, 2],
									   [1.8, 2],
									   [0.8, 1],
									   [0.8, 1],
									   [0.8, 1],
									   [0.8, 1]],
		"zernike_can_be_negative": [True,
									True,
									True,
									True,
									True],
		"n_samples": 3200,
		"labels_dictionary":{
			"ppppp":1,
			"ppppn":2,
			"pppnp":3,
			"pppnn":4,
			"ppnpp":5,
			"ppnpn":6,
			"ppnnp":7,
			"ppnnn":8,
			"pnppp":9,
			"pnppn":10,
			"pnpnp":11,
			"pnpnn":12,
			"pnnpp":13,
			"pnnpn":14,
			"pnnnp":15,
			"pnnnn":16,
			"npppp":17,
			"npppn":18,
			"nppnp":19,
			"nppnn":20,
			"npnpp":21,
			"npnpn":22,
			"npnnp":23,
			"npnnn":24,
			"nnppp":25,
			"nnppn":26,
			"nnpnp":27,
			"nnpnn":28,
			"nnnpp":29,
			"nnnpn":30,
			"nnnnp":31,
			"nnnnn":32
		}
	},

	"MINI_9M":{
		"complex_field_path": ZERNIKE_9M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_9M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH,
		"pca_intensities_path": ZERNIKE_9M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH,
		"umap_intensities_path": ZERNIKE_9M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_9M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH,
		"pca_zernike_mode_coefficients_path": ZERNIKE_9M_PSF_ZERNIKE_MODES_PCA_MINIDATASET_FILE_PATH,
		"umap_zernike_mode_coefficients_path": ZERNIKE_9M_PSF_ZERNIKE_MODES_UMAP_MINIDATASET_FILE_PATH,
		"lp_modes_path": ZERNIKE_9M_PSF_LP_MODES_MINIDATASET_FILE_PATH,
		"pca_lp_modes_path": ZERNIKE_9M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH,
		"umap_lp_modes_path": ZERNIKE_9M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH,
		"pca_output_fluxes_path": ZERNIKE_9M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH,
		"umap_output_fluxes_path": ZERNIKE_9M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH,
		"output_fluxes": ZERNIKE_9M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6, 7, 8, 9, 10],
		"zernike_coefficients_range": [[1.8, 2],
									   [1.8, 2],
									   [0.8, 1],
									   [0.8, 1],
									   [0.8, 1],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5]],
		"zernike_can_be_negative": [True,
									True,
									False,
									False,
									False,
									True,
									True,
									True,
									True],
		"n_samples": 6400,
		"labels_dictionary":{
			"ppppppppp":1,
			"ppppppppn":2,
			"pppppppnp":3,
			"pppppppnn":4,
			"ppppppnpp":5,
			"ppppppnpn":6,
			"ppppppnnp":7,
			"ppppppnnn":8,
			"pppppnppp":9,
			"pppppnppn":10,
			"pppppnpnp":11,
			"pppppnpnn":12,
			"pppppnnpp":13,
			"pppppnnpn":14,
			"pppppnnnp":15,
			"pppppnnnn":16,
			"pnppppppp":17,
			"pnppppppn":18,
			"pnpppppnp":19,
			"pnpppppnn":20,
			"pnppppnpp":21,
			"pnppppnpn":22,
			"pnppppnnp":23,
			"pnppppnnn":24,
			"pnpppnppp":25,
			"pnpppnppn":26,
			"pnpppnpnp":27,
			"pnpppnpnn":28,
			"pnpppnnpp":29,
			"pnpppnnpn":30,
			"pnpppnnnp":31,
			"pnpppnnnn":32,
			"npppppppp":33,
			"npppppppn":34,
			"nppppppnp":35,
			"nppppppnn":36,
			"npppppnpp":37,
			"npppppnpn":38,
			"npppppnnp":39,
			"npppppnnn":40,
			"nppppnppp":41,
			"nppppnppn":42,
			"nppppnpnp":43,
			"nppppnpnn":44,
			"nppppnnpp":45,
			"nppppnnpn":46,
			"nppppnnnp":47,
			"nppppnnnn":48,
			"nnppppppp":49,
			"nnppppppn":50,
			"nnpppppnp":51,
			"nnpppppnn":52,
			"nnppppnpp":53,
			"nnppppnpn":54,
			"nnppppnnp":55,
			"nnppppnnn":56,
			"nnpppnppp":57,
			"nnpppnppn":58,
			"nnpppnpnp":59,
			"nnpppnpnn":60,
			"nnpppnnpp":61,
			"nnpppnnpn":62,
			"nnpppnnnp":63,
			"nnpppnnnn":64,
		}
	},

	"MINI_14M":{
		"complex_field_path": ZERNIKE_14M_COMPLEX_FIELDS_MINIDATASET_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_14M_COMPLEX_FIELDS_INTENSITY_MINIDATASET_FILE_PATH,
		"pca_intensities_path": ZERNIKE_14M_PSF_INTENSITIES_PCA_MINIDATASET_FILE_PATH,
		"umap_intensities_path": ZERNIKE_14M_PSF_INTENSITIES_UMAP_MINIDATASET_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_14M_MODE_COEFFICIENTS_MINIDATASET_FILE_PATH,
		"pca_zernike_mode_coefficients_path": ZERNIKE_14M_PSF_ZERNIKE_MODES_PCA_MINIDATASET_FILE_PATH,
		"umap_zernike_mode_coefficients_path": ZERNIKE_14M_PSF_ZERNIKE_MODES_UMAP_MINIDATASET_FILE_PATH,
		"lp_modes_path": ZERNIKE_14M_PSF_LP_MODES_MINIDATASET_FILE_PATH,
		"pca_lp_modes_path": ZERNIKE_14M_PSF_LP_MODES_PCA_MINIDATASET_FILE_PATH,
		"umap_lp_modes_path": ZERNIKE_14M_PSF_LP_MODES_UMAP_MINIDATASET_FILE_PATH,
		"pca_output_fluxes_path": ZERNIKE_14M_PSF_OUTPUT_FLUXES_PCA_MINIDATASET_FILE_PATH,
		"umap_output_fluxes_path": ZERNIKE_14M_PSF_OUTPUT_FLUXES_UMAP_MINIDATASET_FILE_PATH,
		"output_fluxes": ZERNIKE_14M_OUTPUT_FLUXES_MINIDATASET_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
		"zernike_coefficients_range": [[1.8, 2],
									   [1.8, 2],
									   [0.8, 1],
									   [0.8, 1],
									   [0.8, 1],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5],
									   [0.3, 0.5]],
		"zernike_can_be_negative": [True,
									True,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									True,
									True,
									True,
									True,
									True],
		"n_samples": 12800,
		"labels_dictionary":{
			"pppppppppppppp":1,
			"pppppppppppppn":2,
			"ppppppppppppnp":3,
			"ppppppppppppnn":4,
			"pppppppppppnpp":5,
			"pppppppppppnpn":6,
			"pppppppppppnnp":7,
			"pppppppppppnnn":8,
			"ppppppppppnppp":9,
			"ppppppppppnppn":10,
			"ppppppppppnpnp":11,
			"ppppppppppnpnn":12,
			"ppppppppppnnpp":13,
			"ppppppppppnnpn":14,
			"ppppppppppnnnp":15,
			"ppppppppppnnnn":16,
			"pppppppppnpppp":17,
			"pppppppppnpppn":18,
			"pppppppppnppnp":19,
			"pppppppppnppnn":20,
			"pppppppppnpnpp":21,
			"pppppppppnpnpn":22,
			"pppppppppnpnnp":23,
			"pppppppppnpnnn":24,
			"pppppppppnnppp":25,
			"pppppppppnnppn":26,
			"pppppppppnnpnp":27,
			"pppppppppnnpnn":28,
			"pppppppppnnnpp":29,
			"pppppppppnnnpn":30,
			"pppppppppnnnnp":31,
			"pppppppppnnnnn":32,
			"pnpppppppppppp":33,
			"pnpppppppppppn":34,
			"pnppppppppppnp":35,
			"pnppppppppppnn":36,
			"pnpppppppppnpp":37,
			"pnpppppppppnpn":38,
			"pnpppppppppnnp":39,
			"pnpppppppppnnn":40,
			"pnppppppppnppp":41,
			"pnppppppppnppn":42,
			"pnppppppppnpnp":43,
			"pnppppppppnpnn":44,
			"pnppppppppnnpp":45,
			"pnppppppppnnpn":46,
			"pnppppppppnnnp":47,
			"pnppppppppnnnn":48,
			"pnpppppppnpppp":49,
			"pnpppppppnpppn":50,
			"pnpppppppnppnp":51,
			"pnpppppppnppnn":52,
			"pnpppppppnpnpp":53,
			"pnpppppppnpnpn":54,
			"pnpppppppnpnnp":55,
			"pnpppppppnpnnn":56,
			"pnpppppppnnppp":57,
			"pnpppppppnnppn":58,
			"pnpppppppnnpnp":59,
			"pnpppppppnnpnn":60,
			"pnpppppppnnnpp":61,
			"pnpppppppnnnpn":62,
			"pnpppppppnnnnp":63,
			"pnpppppppnnnnn":64,
			"nppppppppppppp":65,
			"nppppppppppppn":66,
			"npppppppppppnp":67,
			"npppppppppppnn":68,
			"nppppppppppnpp":69,
			"nppppppppppnpn":70,
			"nppppppppppnnp":71,
			"nppppppppppnnn":72,
			"npppppppppnppp":73,
			"npppppppppnppn":74,
			"npppppppppnpnp":75,
			"npppppppppnpnn":76,
			"npppppppppnnpp":77,
			"npppppppppnnpn":78,
			"npppppppppnnnp":79,
			"npppppppppnnnn":80,
			"nppppppppnpppp":81,
			"nppppppppnpppn":82,
			"nppppppppnppnp":83,
			"nppppppppnppnn":84,
			"nppppppppnpnpp":85,
			"nppppppppnpnpn":86,
			"nppppppppnpnnp":87,
			"nppppppppnpnnn":88,
			"nppppppppnnppp":89,
			"nppppppppnnppn":90,
			"nppppppppnnpnp":91,
			"nppppppppnnpnn":92,
			"nppppppppnnnpp":93,
			"nppppppppnnnpn":94,
			"nppppppppnnnnp":95,
			"nppppppppnnnnn":96,
			"nnpppppppppppp":97,
			"nnpppppppppppn":98,
			"nnppppppppppnp":99,
			"nnppppppppppnn":100,
			"nnpppppppppnpp":101,
			"nnpppppppppnpn":102,
			"nnpppppppppnnp":103,
			"nnpppppppppnnn":104,
			"nnppppppppnppp":105,
			"nnppppppppnppn":106,
			"nnppppppppnpnp":107,
			"nnppppppppnpnn":108,
			"nnppppppppnnpp":109,
			"nnppppppppnnpn":110,
			"nnppppppppnnnp":111,
			"nnppppppppnnnn":112,
			"nnpppppppnpppp":113,
			"nnpppppppnpppn":114,
			"nnpppppppnppnp":115,
			"nnpppppppnppnn":116,
			"nnpppppppnpnpp":117,
			"nnpppppppnpnpn":118,
			"nnpppppppnpnnp":119,
			"nnpppppppnpnnn":120,
			"nnpppppppnnppp":121,
			"nnpppppppnnppn":122,
			"nnpppppppnnpnp":123,
			"nnpppppppnnpnn":124,
			"nnpppppppnnnpp":125,
			"nnpppppppnnnpn":126,
			"nnpppppppnnnnp":127,
			"nnpppppppnnnnn":128
		}
	}
}