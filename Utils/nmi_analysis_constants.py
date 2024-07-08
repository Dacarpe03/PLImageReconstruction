NUMPY_SUFFIX = ".npy"

HOME = "/suphys/dcar0877"

NMI_ANALYSIS_DATASET_PATH = f"{HOME}/DaniProjects/SAIL/PhotonicLanternProjects/Data/NMIAnalysisDatasets"

ZERNIKE_COMPLEX_FIELDS_FILE_NAME = "zernike_complex_fields"
ZERNIKE_INTENSITIES_FILE_NAME = "zernike_intensities"
ZERNIKE_MODE_COEFFICIENTS_FILE_NAME = "zernike_mode_coefficients"
PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME = "processed_zernike_mode_coefficients"
ZERNIKE_OUTPUT_FLUXES_FILE_NAME = "zernike_output_fluxes"
PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME = "processed_zernike_output_fluxes"
ZERNIKE_PSF_LP_MODES_FILE_NAME = "lp_modes_from_zernike_psf"
PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME = "processed_lp_modes_from_zernike_psf"


# Folders paths
ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ZernikeComplexFields"
ZERNIKE_MODES_FOLDER_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ZernikeModeCoefficients"
ZERNIKE_PSF_LP_MODES_FOLDER_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/LPModesFromZernikePSF"
ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ZernikeOutputFluxes"

PROCESSED_ZERNIKE_MODE_COEFFICIENTS_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ProcessedZernikeModeCoefficients"
PROCESSED_ZERNIKE_PSF_LP_MODES_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ProcessedLPModesFromZernikePSF"
PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ProcessedZernikeComplexFields"
PROCESSED_ZERNIKE_OUTPUT_FLUXES_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ProcessedZernikeOutputFluxes"

CLUSTER_LABELS_FOLDER_PATH = f"{NMI_ANALYSIS_DATASET_PATH}/ClusterLabels"


# 2 modes Files paths
ZERNIKE_2M_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/2M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_2M_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/2M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_2M_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_2M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_2M_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_2M_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_2M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_2M_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_2M_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"

# 2 modes Files paths
ZERNIKE_5M_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/5M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_5M_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/5M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_5M_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_5M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_5M_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_5M_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_5M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_5M_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_5M_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"

# 2 modes Files paths
ZERNIKE_9M_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/9M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/9M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_9M_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_9M_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_9M_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"

# 9 modes Files paths
BIG_ZERNIKE_9M_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/big_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
BIG_ZERNIKE_9M_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/big_9M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
BIG_ZERNIKE_9M_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/big_9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
BIG_ZERNIKE_9M_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/big_9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

BIG_PROCESSED_ZERNIKE_9M_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/big_9M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
BIG_ZERNIKE_9M_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_big_9M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

BIG_ZERNIKE_9M_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_big_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
BIG_ZERNIKE_9M_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_big_9M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
BIG_ZERNIKE_9M_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_big_9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
BIG_ZERNIKE_9M_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_big_9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"

# 2 modes Files paths
ZERNIKE_14M_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/14M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_14M_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/14M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_14M_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_14M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_14M_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_14M_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_14M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_14M_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_14M_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"

# 2 modes Files paths
ZERNIKE_20M_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_20M_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/20M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_20M_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_20M_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_20M_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/20M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_20M_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_20M_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_20M_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_20M_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_20M_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_20M_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_20M_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"

PATHS_DICTIONARY = {
	"NMI_ANALYSIS_2M":{
		"complex_fields_path": ZERNIKE_2M_COMPLEX_FIELDS_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_2M_COMPLEX_FIELDS_INTENSITY_FILE_PATH,
		"umap_intensities_path": ZERNIKE_2M_PSF_INTENSITIES_UMAP_FILE_PATH,
		"complex_fields_labels_path": ZERNIKE_2M_COMPLEX_FIELDS_LABELS_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_2M_MODE_COEFFICIENTS_FILE_PATH,
		"zernike_mode_coefficients_labels_path": ZERNIKE_2M_MODE_COEFFICIENTS_LABELS_FILE_PATH,
		"lp_modes_path": ZERNIKE_2M_PSF_LP_MODES_FILE_PATH,
		"lp_modes_labels_path": ZERNIKE_2M_PSF_LP_MODES_LABELS_FILE_PATH,
		"output_fluxes_path": ZERNIKE_2M_OUTPUT_FLUXES_FILE_PATH,
		"output_fluxes_labels_path": ZERNIKE_2M_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"zernike_mode_indexes": [2, 3],
		"zernike_coefficients_range": [[-2, 2],
									   [-2, 2]],
		"zernike_can_be_negative": [False,
									False],
		"n_samples": 5000
	},


	"NMI_ANALYSIS_5M":{
		"complex_fields_path": ZERNIKE_5M_COMPLEX_FIELDS_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_5M_COMPLEX_FIELDS_INTENSITY_FILE_PATH,
		"umap_intensities_path": ZERNIKE_5M_PSF_INTENSITIES_UMAP_FILE_PATH,
		"complex_fields_labels_path": ZERNIKE_5M_COMPLEX_FIELDS_LABELS_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_5M_MODE_COEFFICIENTS_FILE_PATH,
		"zernike_mode_coefficients_labels_path": ZERNIKE_5M_MODE_COEFFICIENTS_LABELS_FILE_PATH,
		"lp_modes_path": ZERNIKE_5M_PSF_LP_MODES_FILE_PATH,
		"lp_modes_labels_path": ZERNIKE_5M_PSF_LP_MODES_LABELS_FILE_PATH,
		"output_fluxes_path": ZERNIKE_5M_OUTPUT_FLUXES_FILE_PATH,
		"output_fluxes_labels_path": ZERNIKE_5M_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6],
		"zernike_coefficients_range": [[-2, 2],
									   [-2, 2],
									   [-1.5, 1.5],
									   [-1.5, 1.5],
									   [-1.5, 1.5]],
		"zernike_can_be_negative": [False,
									False,
									False,
									False,
									False],
		"n_samples": 5000
	},


	"NMI_ANALYSIS_9M":{
		"complex_fields_path": ZERNIKE_9M_COMPLEX_FIELDS_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_9M_COMPLEX_FIELDS_INTENSITY_FILE_PATH,
		"umap_intensities_path": ZERNIKE_9M_PSF_INTENSITIES_UMAP_FILE_PATH,
		"complex_fields_labels_path": ZERNIKE_9M_COMPLEX_FIELDS_LABELS_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_9M_MODE_COEFFICIENTS_FILE_PATH,
		"zernike_mode_coefficients_labels_path": ZERNIKE_9M_MODE_COEFFICIENTS_LABELS_FILE_PATH,
		"lp_modes_path": ZERNIKE_9M_PSF_LP_MODES_FILE_PATH,
		"lp_modes_labels_path": ZERNIKE_9M_PSF_LP_MODES_LABELS_FILE_PATH,
		"output_fluxes_path": ZERNIKE_9M_OUTPUT_FLUXES_FILE_PATH,
		"output_fluxes_labels_path": ZERNIKE_9M_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6, 7, 8, 9, 10],
		"zernike_coefficients_range": [[-2, 2],
									   [-2, 2],
									   [-1.5, 1.5],
									   [-1.5, 1.5],
									   [-1.5, 1.5],
									   [-1, 1],
									   [-1, 1],
									   [-1, 1],
									   [-1, 1]],
		"zernike_can_be_negative": [False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False],
		"n_samples": 5000
	},

	"NMI_ANALYSIS_14M":{
		"complex_fields_path": ZERNIKE_14M_COMPLEX_FIELDS_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_14M_COMPLEX_FIELDS_INTENSITY_FILE_PATH,
		"umap_intensities_path": ZERNIKE_14M_PSF_INTENSITIES_UMAP_FILE_PATH,
		"complex_fields_labels_path": ZERNIKE_14M_COMPLEX_FIELDS_LABELS_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_14M_MODE_COEFFICIENTS_FILE_PATH,
		"zernike_mode_coefficients_labels_path": ZERNIKE_14M_MODE_COEFFICIENTS_LABELS_FILE_PATH,
		"lp_modes_path": ZERNIKE_14M_PSF_LP_MODES_FILE_PATH,
		"lp_modes_labels_path": ZERNIKE_14M_PSF_LP_MODES_LABELS_FILE_PATH,
		"output_fluxes_path": ZERNIKE_14M_OUTPUT_FLUXES_FILE_PATH,
		"output_fluxes_labels_path": ZERNIKE_14M_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
		"zernike_coefficients_range": [[-2, 2],
									   [-2, 2],
									   [-1.5, 1.5],
									   [-1.5, 1.5],
									   [-1.5, 1.5],
									   [-1, 1],
									   [-1, 1],
									   [-1, 1],
									   [-1, 1],
									   [-0.5, 0.5],
									   [-0.5, 0.5],
									   [-0.5, 0.5],
									   [-0.5, 0.5],
									   [-0.5, 0.5]],
		"zernike_can_be_negative": [False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False],
		"n_samples": 5000
	},

	"NMI_ANALYSIS_20M":{
		"complex_fields_path": ZERNIKE_20M_COMPLEX_FIELDS_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_20M_COMPLEX_FIELDS_INTENSITY_FILE_PATH,
		"umap_intensities_path": ZERNIKE_20M_PSF_INTENSITIES_UMAP_FILE_PATH,
		"complex_fields_labels_path": ZERNIKE_20M_COMPLEX_FIELDS_LABELS_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_20M_MODE_COEFFICIENTS_FILE_PATH,
		"zernike_mode_coefficients_labels_path": ZERNIKE_20M_MODE_COEFFICIENTS_LABELS_FILE_PATH,
		"lp_modes_path": ZERNIKE_20M_PSF_LP_MODES_FILE_PATH,
		"lp_modes_labels_path": ZERNIKE_20M_PSF_LP_MODES_LABELS_FILE_PATH,
		"output_fluxes_path": ZERNIKE_20M_OUTPUT_FLUXES_FILE_PATH,
		"output_fluxes_labels_path": ZERNIKE_20M_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
		"zernike_coefficients_range": [[-2, 2],
									   [-2, 2],
									   [-1.5, 1.5],
									   [-1.5, 1.5],
									   [-1.5, 1.5],
									   [-1, 1],
									   [-1, 1],
									   [-1, 1],
									   [-1, 1],
									   [-0.5, 0.5],
									   [-0.5, 0.5],
									   [-0.5, 0.5],
									   [-0.5, 0.5],
									   [-0.5, 0.5],
									   [-0.2, 0.2],
									   [-0.2, 0.2],
									   [-0.2, 0.2],
									   [-0.2, 0.2],
									   [-0.2, 0.2],
									   [-0.2, 0.2]],
		"zernike_can_be_negative": [False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False],
		"n_samples": 5000
	},


	"NMI_ANALYSIS_BIG_9M":{
		"complex_fields_path": BIG_ZERNIKE_9M_COMPLEX_FIELDS_FILE_PATH,
		"intensities_file_path": BIG_PROCESSED_ZERNIKE_9M_COMPLEX_FIELDS_INTENSITY_FILE_PATH,
		"umap_intensities_path": BIG_ZERNIKE_9M_PSF_INTENSITIES_UMAP_FILE_PATH,
		"complex_fields_labels_path": BIG_ZERNIKE_9M_COMPLEX_FIELDS_LABELS_FILE_PATH,
		"zernike_mode_coefficients_path": BIG_ZERNIKE_9M_MODE_COEFFICIENTS_FILE_PATH,
		"zernike_mode_coefficients_labels_path": BIG_ZERNIKE_9M_MODE_COEFFICIENTS_LABELS_FILE_PATH,
		"lp_modes_path": BIG_ZERNIKE_9M_PSF_LP_MODES_FILE_PATH,
		"lp_modes_labels_path": BIG_ZERNIKE_9M_PSF_LP_MODES_LABELS_FILE_PATH,
		"output_fluxes_path": BIG_ZERNIKE_9M_OUTPUT_FLUXES_FILE_PATH,
		"output_fluxes_labels_path": BIG_ZERNIKE_9M_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6, 7, 8, 9, 10],
		"zernike_coefficients_range": [[-0.8, 0.8],
									   [-0.8, 0.8],
									   [-0.6, 0.6],
									   [-0.6, 0.6],
									   [-0.6, 0.6],
									   [-0.4, 0.4],
									   [-0.4, 0.4],
									   [-0.4, 0.4],
									   [-0.4, 0.4]],
		"zernike_can_be_negative": [False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False],
		"n_samples": 100000
	},
}