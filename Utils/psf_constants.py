NUMPY_SUFFIX = ".npy"

PSF_DATA_PATH = "~/DaniProjects/SAIL/PhotonicLanternProjects/Data/PSFReconstruction"
COMPLEX_FIELDS_DATA_PATH = f"{PSF_DATA_PATH}/ComplexFields"
OUTPUT_FLUXES_DATA_PATH = f"{PSF_DATA_PATH}/OutputFluxes"

PSF_TRAIN_FILE_SUFFIXES = ["00", "01", "02", "03", "04", "05", "06"]
PSF_VALIDATION_FILE_SUFFIX = "07"
PSF_TEST_FILE_SUFFIX = "08"

LANTERN_FIBER_FILENAME = f"/extractedvals_probeset_19LP__Good202107.npz"

TRAIN_DATA_FOLDER_NAME = "TrainData"
VALIDATION_DATA_FOLDER_NAME = "ValidationData"
TEST_DATA_FOLDER_NAME = "TestData"

COMPLEX_FIELDS_FILE_NAME = "complex_fields"
OUTPUT_FLUXES_FILE_NAME = "output_fluxes"

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

# Data folders for output fluxes
PROCESSED_TRAIN_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_OUTPUT_FLUXES_DATA_PATH}/{TRAIN_DATA_FOLDER_NAME}"
PROCESSED_VALIDATION_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_OUTPUT_FLUXES_DATA_PATH}/{VALIDATION_DATA_FOLDER_NAME}"
PROCESSED_TEST_OUTPUT_FLUXES_FOLDER_PATH = f"{PROCESSED_OUTPUT_FLUXES_DATA_PATH}/{TEST_DATA_FOLDER_NAME}"

# Data filenames for fully connected output_fluxes fields data
FC_PROCESSED_TRAIN_OUTPUT_FLUXES_PREFIX = f"{PROCESSED_TRAIN_OUTPUT_FLUXES_FOLDER_PATH}/fc_{OUTPUT_FLUXES_FILE_NAME}"
FC_PROCESSED_VALIDATION_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_VALIDATION_OUTPUT_FLUXES_FOLDER_PATH}/fc_{OUTPUT_FLUXES_FILE_NAME}{PSF_VALIDATION_FILE_SUFFIX}{NUMPY_SUFFIX}"
FC_PROCESSED_TEST_OUTPUT_FLUXES_FILE_PATH = f"{PROCESSED_TEST_OUTPUT_FLUXES_FOLDER_PATH}/fc_{OUTPUT_FLUXES_FILE_NAME}{PSF_TEST_FILE_SUFFIX}{NUMPY_SUFFIX}"

SUBFILE_SAMPLES = 10000
COMPLEX_NUMBER_NORMALIZATION_CONSTANT = 50000

# CODE PATH
PSF_CODE_PATH = "~/DaniProjects/SAIL/PhotonicLanternProjects/PLImageReconstruction/PSFReconstruction/"

# MODELS PATH
PSF_MODELS_FOLDER_PATH = "~/DaniProjects/SAIL/PhotonicLanternProjects/PLImageReconstruction/PSFReconstruction/Models"
PSF_MODELS_DESCRIPTION_FILE_PATH = "~/DaniProjects/SAIL/PhotonicLanternProjects/PLImageReconstruction/PSFReconstruction/Models/models_descriptions.txt"

# Temp images folder
PSF_TEMP_IMAGES = f"{PSF_CODE_PATH}/temp_images"