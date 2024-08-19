NUMPY_SUFFIX = ".npy"

HOME = "/home/dani"

NMI_ANALYSIS_DATASET_PATH = f"{HOME}/DaniProjects/SAIL/PhotonicLanternProjects/Data/LastDance"

ZERNIKE_COMPLEX_FIELDS_FILE_NAME = "zernike_complex_fields"
ZERNIKE_INTENSITIES_FILE_NAME = "zernike_intensities"
ZERNIKE_MODE_COEFFICIENTS_FILE_NAME = "zernike_mode_coefficients"
PROCESSED_ZERNIKE_MODE_COEFFICIENTS_FILE_NAME = "processed_zernike_mode_coefficients"
ZERNIKE_OUTPUT_FLUXES_FILE_NAME = "zernike_output_fluxes"
ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME = "zernike_complex_output_fluxes"
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

# 9 modes Files paths, 500 points
ZERNIKE_9M_500_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_500_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_500_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/9M_500_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_500_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_500_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_500_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_500_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_500_COMPLEX_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_500_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_500_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/9M_500_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_500_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_500_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_500_PSF_COMPLEX_FIELDS_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_500_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_500_COMPLEX_FIELDS_INTENSITY_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_500_{ZERNIKE_INTENSITIES_FILE_NAME}"
ZERNIKE_9M_500_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_500_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_9M_500_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_500_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_9M_500_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_500_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_9M_500_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_500_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
ZERNIKE_9M_500_COMPLEX_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_500_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}"

# 9 modes Files paths, 1000 points
ZERNIKE_9M_1000_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_1000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_1000_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/9M_1000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_1000_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_1000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_1000_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_1000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_1000_COMPLEX_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_1000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_1000_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/9M_1000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_1000_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_1000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_1000_PSF_COMPLEX_FIELDS_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_1000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_1000_COMPLEX_FIELDS_INTENSITY_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_1000_{ZERNIKE_INTENSITIES_FILE_NAME}"
ZERNIKE_9M_1000_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_1000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_9M_1000_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_1000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_9M_1000_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_1000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_9M_1000_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_1000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
ZERNIKE_9M_1000_COMPLEX_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_1000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}"


# 9 modes Files paths, 2000 points
ZERNIKE_9M_2000_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_2000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_2000_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/9M_2000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_2000_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_2000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_2000_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_2000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_2000_COMPLEX_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_2000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_2000_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/9M_2000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_2000_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_2000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_2000_PSF_COMPLEX_FIELDS_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_2000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_2000_COMPLEX_FIELDS_INTENSITY_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_2000_{ZERNIKE_INTENSITIES_FILE_NAME}"
ZERNIKE_9M_2000_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_2000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_9M_2000_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_2000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_9M_2000_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_2000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_9M_2000_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_2000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
ZERNIKE_9M_2000_COMPLEX_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_2000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}"


# 9 modes Files paths, 5000 points
ZERNIKE_9M_5000_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_5000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_5000_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/9M_5000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_5000_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_5000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_5000_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_5000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_5000_COMPLEX_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_5000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_5000_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/9M_5000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_5000_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_5000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_5000_PSF_COMPLEX_FIELDS_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_5000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_5000_COMPLEX_FIELDS_INTENSITY_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_5000_{ZERNIKE_INTENSITIES_FILE_NAME}"
ZERNIKE_9M_5000_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_5000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_9M_5000_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_5000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_9M_5000_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_5000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_9M_5000_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_5000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
ZERNIKE_9M_5000_COMPLEX_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_5000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}"


# 9 modes Files paths, 10000 points
ZERNIKE_9M_10000_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_10000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_10000_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/9M_10000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_10000_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_10000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_10000_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_10000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_10000_COMPLEX_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_10000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_10000_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/9M_10000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_10000_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_10000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_10000_PSF_COMPLEX_FIELDS_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_10000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_10000_COMPLEX_FIELDS_INTENSITY_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_10000_{ZERNIKE_INTENSITIES_FILE_NAME}"
ZERNIKE_9M_10000_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_10000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_9M_10000_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_10000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_9M_10000_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_10000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_9M_10000_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_10000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
ZERNIKE_9M_10000_COMPLEX_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_10000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}"


# 9 modes Files paths, 20000 points
ZERNIKE_9M_20000_COMPLEX_FIELDS_FILE_PATH = f"{ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_20000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_20000_MODE_COEFFICIENTS_FILE_PATH = f"{ZERNIKE_MODES_FOLDER_PATH}/9M_20000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_20000_PSF_LP_MODES_FILE_PATH = f"{ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_20000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_20000_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_20000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_20000_COMPLEX_OUTPUT_FLUXES_FILE_PATH = f"{ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_20000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}{NUMPY_SUFFIX}"

PROCESSED_ZERNIKE_9M_20000_COMPLEX_FIELDS_INTENSITY_FILE_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/9M_20000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_20000_PSF_INTENSITIES_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_20000_{ZERNIKE_INTENSITIES_FILE_NAME}{NUMPY_SUFFIX}"
ZERNIKE_9M_20000_PSF_COMPLEX_FIELDS_UMAP_FILE_PATH= f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_PATH}/umap_9M_20000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{NUMPY_SUFFIX}"

ZERNIKE_9M_20000_COMPLEX_FIELDS_INTENSITY_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_20000_{ZERNIKE_INTENSITIES_FILE_NAME}"
ZERNIKE_9M_20000_COMPLEX_FIELDS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_20000_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
ZERNIKE_9M_20000_MODE_COEFFICIENTS_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_20000_{ZERNIKE_MODE_COEFFICIENTS_FILE_NAME}"
ZERNIKE_9M_20000_PSF_LP_MODES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_20000_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
ZERNIKE_9M_20000_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_20000_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
ZERNIKE_9M_20000_COMPLEX_OUTPUT_FLUXES_LABELS_FILE_PATH = f"{CLUSTER_LABELS_FOLDER_PATH}/labels_9M_20000_{ZERNIKE_COMPLEX_OUTPUT_FLUXES_FILE_NAME}"

PATHS_DICTIONARY = {
	"NMI_ANALYSIS_9M_500":{
		"complex_fields_path": ZERNIKE_9M_500_COMPLEX_FIELDS_FILE_PATH,
		"umap_complex_fields_path": ZERNIKE_9M_500_PSF_COMPLEX_FIELDS_UMAP_FILE_PATH,
		"intensities_file_path": PROCESSED_ZERNIKE_9M_500_COMPLEX_FIELDS_INTENSITY_FILE_PATH,
		"umap_intensities_path": ZERNIKE_9M_500_PSF_INTENSITIES_UMAP_FILE_PATH,
		"complex_fields_labels_path": ZERNIKE_9M_500_COMPLEX_FIELDS_LABELS_FILE_PATH,
		"intensities_labels_path": ZERNIKE_9M_500_COMPLEX_FIELDS_INTENSITY_LABELS_FILE_PATH,
		"zernike_mode_coefficients_path": ZERNIKE_9M_500_MODE_COEFFICIENTS_FILE_PATH,
		"zernike_mode_coefficients_labels_path": ZERNIKE_9M_500_MODE_COEFFICIENTS_LABELS_FILE_PATH,
		"lp_modes_path": ZERNIKE_9M_500_PSF_LP_MODES_FILE_PATH,
		"lp_modes_labels_path": ZERNIKE_9M_500_PSF_LP_MODES_LABELS_FILE_PATH,
		"complex_output_fluxes_path": ZERNIKE_9M_500_COMPLEX_OUTPUT_FLUXES_FILE_PATH,
		"output_fluxes_path": ZERNIKE_9M_500_OUTPUT_FLUXES_FILE_PATH,
		"complex_output_fluxes_labels_path": ZERNIKE_9M_500_COMPLEX_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"output_fluxes_labels_path": ZERNIKE_9M_500_OUTPUT_FLUXES_LABELS_FILE_PATH,
		"zernike_mode_indexes": [2, 3, 4, 5, 6, 7, 8, 9, 10],
		"zernike_coefficients_range": [
		[-0.22, 0.22],
		[-0.22, 0.22],
		[-0.22, 0.22],
		[-0.22, 0.22],
		[-0.22, 0.22],
		[-0.22, 0.22],
		[-0.22, 0.22],
		[-0.22, 0.22],
		[-0.22, 0.22]],
		"zernike_can_be_negative": [False,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False],
		"n_samples": 1,
		"n_clusters": [10, 20, 50, 100, 200]
	}
}