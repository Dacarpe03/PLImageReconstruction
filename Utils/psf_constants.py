NUMPY_SUFFIX = ".npy"

HOME = "/home/dani"

PSF_DATA_PATH = f"{HOME}/DaniProjects/SAIL/PhotonicLanternProjects/Data/PSFReconstruction"
COMPLEX_FIELDS_DATA_PATH = f"{PSF_DATA_PATH}/ComplexFields"
OUTPUT_FLUXES_DATA_PATH = f"{PSF_DATA_PATH}/OutputFluxes"

PSF_TRAIN_FILE_SUFFIXES = ["00", "01", "02", "03", "04", "05", "06"]
PSF_VALIDATION_FILE_SUFFIX = "07"
PSF_TEST_FILE_SUFFIX = "08"

LANTERN_FIBER_FILENAME = f"/extractedvals_probeset_19LP__Good202107.npz"
TRANSFER_MATRIX_42_MODES = f"{PSF_DATA_PATH}/transfer_matrix_42_modes{NUMPY_SUFFIX}"

TRAIN_DATA_FOLDER_NAME = "TrainData"
VALIDATION_DATA_FOLDER_NAME = "ValidationData"
TEST_DATA_FOLDER_NAME = "TestData"

ZERNIKE_COMPLEX_FIELDS_FILE_NAME = "zernike_complex_fields"
COMPLEX_FIELDS_FILE_NAME = "complex_fields"
OUTPUT_FLUXES_FILE_NAME = "output_fluxes"
ZERNIKE_OUTPUT_FLUXES_FILE_NAME = "zernike_output_fluxes"
PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME = "processed_zernike_output_fluxes"
ZERNIKE_PSF_LP_MODES_FILE_NAME = "lp_modes_from_zernike_psf"
PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME = "processed_lp_modes_from_zernike_psf"
EUCLIDEAN_DISTANCES_FILE_NAME = "euclidean_distances"
ZERNIKE_EUCLIDEAN_DISTANCES_FILE_NAME = "zernike_euclidean_distances"
EUCLIDEAN_DISTANCES_PAIRS_FILE_NAME = "pairs"
ZERNIKE_EUCLIDEAN_DISTANCES_PAIRS_FILE_NAME = "zernike_pairs"

# Data folders for complex fields
TRAIN_COMPLEX_FIELDS_FOLDER_PATH = f"{COMPLEX_FIELDS_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
VALIDATION_COMPLEX_FIELDS_FOLDER_PATH = f"{COMPLEX_FIELDS_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
TEST_COMPLEX_FIELDS_FOLDER_PATH = f"{COMPLEX_FIELDS_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data folders for output fluxes
TRAIN_OUTPUT_FLUXES_FOLDER_PATH = f"{OUTPUT_FLUXES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
VALIDATION_OUTPUT_FLUXES_FOLDER_PATH = f"{OUTPUT_FLUXES_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
TEST_OUTPUT_FLUXES_FOLDER_PATH = f"{OUTPUT_FLUXES_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for complex fields
TRAIN_COMPLEX_FIELDS_PREFIX = f"{TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/{COMPLEX_FIELDS_FILE_NAME}"
VALIDATION_COMPLEX_FIELDS_PATH = f"{VALIDATION_COMPLEX_FIELDS_FOLDER_PATH}/{COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_COMPLEX_FIELDS_PATH = f"{TEST_COMPLEX_FIELDS_FOLDER_PATH}/{COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

# Data filenames for output fluxes
TRAIN_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_OUTPUT_FLUXES_FOLDER_PATH}/{OUTPUT_FLUXES_FILE_NAME}"
VALIDATION_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_OUTPUT_FLUXES_FOLDER_PATH}/{OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_OUTPUT_FLUXES_FILE_PATH = f"{TEST_OUTPUT_FLUXES_FOLDER_PATH}/{OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

# Processed data
PROCESSED_COMPLEX_FIELDS_DATA_PATH = f"{PSF_DATA_PATH}/ProcessedComplexFields"
PROCESSED_OUTPUT_FLUXES_DATA_PATH = f"{PSF_DATA_PATH}/ProcessedOutputFluxes"

# Data folders for complex fields
PROCESSED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH = f"{PROCESSED_COMPLEX_FIELDS_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
PROCESSED_VALIDATION_COMPLEX_FIELDS_FOLDER_PATH = f"{PROCESSED_COMPLEX_FIELDS_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
PROCESSED_TEST_COMPLEX_FIELDS_FOLDER_PATH = f"{PROCESSED_COMPLEX_FIELDS_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for fully connected complex fields data
FC_PROCESSED_TRAIN_COMPLEX_FIELDS_PREFIX = f"{PROCESSED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_{COMPLEX_FIELDS_FILE_NAME}"
FC_PROCESSED_VALIDATION_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_COMPLEX_FIELDS_FOLDER_PATH}/fc_{COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
FC_PROCESSED_TEST_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_COMPLEX_FIELDS_FOLDER_PATH}/fc_{COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

# Data filenames for cropped fully connected fields data
FC_CROPPED_TRAIN_COMPLEX_FIELDS_PREFIX = f"{PROCESSED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_{COMPLEX_FIELDS_FILE_NAME}"
FC_CROPPED_VALIDATION_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_{COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
FC_CROPPED_TEST_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_{COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

#   data
PREDICTED_COMPLEX_FIELDS_DATA_PATH = f"{PSF_DATA_PATH}/PredictedComplexFields"

# Data folders for predicted complex fields
FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH = f"{PREDICTED_COMPLEX_FIELDS_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"

# Data filenames 
FC_PREDICTED_TRAIN_COMPLEX_FIELDS_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_{COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_CROPPED_TRAIN_COMPLEX_FIELDS_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_cropped_{COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_CROPPED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_cropped_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_CROPPED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_cropped_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_CROPPED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_cropped_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_CROPPED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_cropped_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
FC_PREDICTED_CROPPED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{FC_PREDICTED_TRAIN_COMPLEX_FIELDS_FOLDER_PATH}/fc_predicted_cropped_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"

# Data folders for output fluxes
PROCESSED_TRAIN_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_OUTPUT_FLUXES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
PROCESSED_VALIDATION_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_OUTPUT_FLUXES_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
PROCESSED_TEST_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_OUTPUT_FLUXES_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for fully connected output_fluxes fields data
FC_PROCESSED_TRAIN_OUTPUT_FLUXES_PREFIX = f"{PROCESSED_TRAIN_OUTPUT_FLUXES_FOLDER_PATH}/fc_{OUTPUT_FLUXES_FILE_NAME}"
FC_PROCESSED_VALIDATION_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_VALIDATION_OUTPUT_FLUXES_FOLDER_PATH}/fc_{OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
FC_PROCESSED_TEST_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_TEST_OUTPUT_FLUXES_FOLDER_PATH}/fc_{OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

# Data path for euclidean distances
EUCLIDEAN_DISTANCES_DATA_PATH = f"{PSF_DATA_PATH}/EuclideanDistances"

# Folder path for euclidean distances
TRAIN_EUCLIDEAN_DISTANCES_FOLDER_PATH = f"{EUCLIDEAN_DISTANCES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"

# Data filenames for euclidean distances
TRAIN_EUCLIDEAN_DISTANCES_PREFIX = f"{TRAIN_EUCLIDEAN_DISTANCES_FOLDER_PATH}/{EUCLIDEAN_DISTANCES_FILE_NAME}"
ZERNIKE_TRAIN_EUCLIDEAN_DISTANCES_PREFIX = f"{TRAIN_EUCLIDEAN_DISTANCES_FOLDER_PATH}/{ZERNIKE_EUCLIDEAN_DISTANCES_FILE_NAME}"
TRAIN_EUCLIDEAN_DISTANCES_PAIRS_PREFIX = f"{TRAIN_EUCLIDEAN_DISTANCES_FOLDER_PATH}/{EUCLIDEAN_DISTANCES_PAIRS_FILE_NAME}"
ZERNIKE_TRAIN_EUCLIDEAN_DISTANCES_PAIRS_PREFIX = f"{TRAIN_EUCLIDEAN_DISTANCES_FOLDER_PATH}/{ZERNIKE_EUCLIDEAN_DISTANCES_PAIRS_FILE_NAME}"

# Zernike complex fields
# Processed data
ZERNIKE_COMPLEX_FIELDS_DATA_PATH = f"{PSF_DATA_PATH}/ZernikeComplexFields"
PROCESSED_ZERNIKE_COMPLEX_FIELDS_DATA_PATH = f"{PSF_DATA_PATH}/ProcessedZernikeComplexFields"
ZERNIKE_OUTPUT_FLUXES_DATA_PATH = f"{PSF_DATA_PATH}/ZernikeOutputFluxes"
PROCESSED_ZERNIKE_OUTPUT_FLUXES_DATA_PATH = f"{PSF_DATA_PATH}/ProcessedZernikeOutputFluxes"
ZERNIKE_PSF_LP_MODES_DATA_PATH = f"{PSF_DATA_PATH}/LPModesFromZernikePSF"
PROCESSED_ZERNIKE_PSF_LP_MODES_DATA_PATH = f"{PSF_DATA_PATH}/ProcessedLPModesFromZernikePSF"

# Data folders for zernike complex fields
TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH = f"{ZERNIKE_COMPLEX_FIELDS_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH = f"{ZERNIKE_COMPLEX_FIELDS_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH = f"{ZERNIKE_COMPLEX_FIELDS_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for zernike complex fields
TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
VALIDATION_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
VALIDATION_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
VALIDATION_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
VALIDATION_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
VALIDATION_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

# Data folders for processed zernike complex fields
PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH = f"{PROCESSED_ZERNIKE_COMPLEX_FIELDS_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for processed zernike complex fields
PROCESSED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
PROCESSED_VALIDATION_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
PROCESSED_VALIDATION_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
PROCESSED_VALIDATION_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
PROCESSED_VALIDATION_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
PROCESSED_VALIDATION_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

# Data filenames for cropped processed zernike complex fields
CROPPED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
CROPPED_VALIDATION_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
CROPPED_TEST_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_2M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

CROPPED_TRAIN_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
CROPPED_VALIDATION_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
CROPPED_TEST_5M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_5M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

CROPPED_TRAIN_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
CROPPED_VALIDATION_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
CROPPED_TEST_9M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_9M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

CROPPED_TRAIN_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
CROPPED_VALIDATION_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
CROPPED_TEST_14M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_14M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

CROPPED_TRAIN_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}"
CROPPED_VALIDATION_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
CROPPED_TEST_20M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_COMPLEX_FIELDS_FOLDER_PATH}/fc_cropped_20M_{ZERNIKE_COMPLEX_FIELDS_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

# Data folders for zernike output fluxes
TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH = f"{ZERNIKE_OUTPUT_FLUXES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH = f"{ZERNIKE_OUTPUT_FLUXES_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH = f"{ZERNIKE_OUTPUT_FLUXES_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

PROCESSED_TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
PROCESSED_VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
PROCESSED_TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_ZERNIKE_OUTPUT_FLUXES_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for zernike complex fields
TRAIN_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
VALIDATION_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PL42_VALIDATION_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_2M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
VALIDATION_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PL42_VALIDATION_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_5M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
VALIDATION_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PL42_VALIDATION_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_9M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
VALIDATION_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PL42_VALIDATION_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_14M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
VALIDATION_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PL42_VALIDATION_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/PL42_20M_{ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/2M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PROCESSED_VALIDATION_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/2M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/2M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/5M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PROCESSED_VALIDATION_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/5M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_5M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/5M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PROCESSED_VALIDATION_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_9M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/9M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/14M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PROCESSED_VALIDATION_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/14M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_14M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/14M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/20M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}"
PROCESSED_VALIDATION_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/20M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_20M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_OUTPUT_FLUXES_FOLDER_PATH}/20M_{PROCESSED_ZERNIKE_OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"


# Data folders for zernike PSF LP modes
TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH = f"{ZERNIKE_PSF_LP_MODES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH = f"{ZERNIKE_PSF_LP_MODES_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH = f"{ZERNIKE_PSF_LP_MODES_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

PROCESSED_TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH = f"{PROCESSED_ZERNIKE_PSF_LP_MODES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
PROCESSED_VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH = f"{PROCESSED_ZERNIKE_PSF_LP_MODES_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
PROCESSED_TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH = f"{PROCESSED_ZERNIKE_PSF_LP_MODES_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for zernike complex fields
TRAIN_2M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
VALIDATION_2M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_2M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_2M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PL42_VALIDATION_2M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_2M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_2M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_5M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
VALIDATION_5M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_5M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_5M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PL42_VALIDATION_5M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_5M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_5M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_9M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
VALIDATION_9M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_9M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_9M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PL42_VALIDATION_9M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_9M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_9M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_14M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
VALIDATION_14M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_14M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_14M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PL42_VALIDATION_14M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_14M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_14M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

TRAIN_20M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
VALIDATION_20M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
TEST_20M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TRAIN_20M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PL42_VALIDATION_20M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PL42_TEST_20M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/PL42_20M_{ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_2M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/2M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PROCESSED_VALIDATION_2M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/2M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_2M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/2M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_5M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/5M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PROCESSED_VALIDATION_5M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/5M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_5M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/5M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_9M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PROCESSED_VALIDATION_9M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_9M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/9M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_14M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/14M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PROCESSED_VALIDATION_14M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/14M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_14M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/14M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

PROCESSED_TRAIN_20M_ZERNIKE_PSF_LP_MODES_FILE_PREFIX = f"{PROCESSED_TRAIN_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/20M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}"
PROCESSED_VALIDATION_20M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_VALIDATION_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/20M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
PROCESSED_TEST_20M_ZERNIKE_PSF_LP_MODES_FILE_PATH = f"{PROCESSED_TEST_ZERNIKE_PSF_LP_MODES_FOLDER_PATH}/20M_{PROCESSED_ZERNIKE_PSF_LP_MODES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"


SUBFILE_SAMPLES = 10000
COMPLEX_NUMBER_NORMALIZATION_CONSTANT = 50000

# CODE PATH
PSF_CODE_PATH = f"{HOME}/DaniProjects/SAIL/PhotonicLanternProjects/PLImageReconstruction/PSFReconstruction/"

# MODELS PATH
PSF_MODELS_FOLDER_PATH = f"{HOME}/DaniProjects/SAIL/PhotonicLanternProjects/PLImageReconstruction/PSFReconstruction/Models"
PSF_MODELS_DESCRIPTION_FILE_PATH = f"{HOME}/DaniProjects/SAIL/PhotonicLanternProjects/PLImageReconstruction/PSFReconstruction/Models/models_descriptions.txt"

# Temp images folder
PSF_TEMP_IMAGES = f"{PSF_CODE_PATH}/temp_images"