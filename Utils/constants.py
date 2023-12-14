# FILE SUFFIXES
KERAS_SUFFIX = ".keras"
NUMPY_SUFFIX = ".npy"
PICKLE_SUFFIX = ".pkl"

# DATA  FOLDER
PL_DATA_PATH = "../Data"

# ORIGINAL DATA FOLDERS
ORIGINAL_FOLDER_DATA_PATH = f"{PL_DATA_PATH}/OriginalData"

ORIGINAL_FLUXES_FOLDER = f"{ORIGINAL_FOLDER_DATA_PATH}/plfluxes_simplepoly__pllabdata_20230625a_superK_slmcube_20230625_complsines-01sp_04"
ORIGINAL_FLUXES_FILE = f"{ORIGINAL_FLUXES_FOLDER}/all_fluxes{NUMPY_SUFFIX}"

ORIGINAL_SLM_FOLDER = f"{ORIGINAL_FOLDER_DATA_PATH}/slmcube_20230625_complsines-01sp_04_PSFWFs_file"
ORIGINAL_AMPLITUDE_FILENAME = "complexsine_pupamp.npy"
ORIGINAL_PHASE_FILENAME = "complexsine_pupphase.npy"

# PROCESSED DATA FOLDERS
PROCESSED_DATA_FOLDER = f"{PL_DATA_PATH}/ProcessedData"

TRAIN_FOLDER = f"{PROCESSED_DATA_FOLDER}/TrainData"
VALIDATION_FOLDER = f"{PROCESSED_DATA_FOLDER}/ValidationData"
TEST_FOLDER = f"{PROCESSED_DATA_FOLDER}/TestData"
SCALERS_FOLDER = f"{PROCESSED_DATA_FOLDER}/Scalers"

# PROCESSED DATA FILENAMES
FC_FLUXES_FILENAME = f"fully_connected_fluxes{NUMPY_SUFFIX}"
CNN_FLUXES_FILENAME = f"convolutional_fluxes{NUMPY_SUFFIX}"
AUTOENCODER_FLUXES_FILENAME = f"autoencoder_fluxes{NUMPY_SUFFIX}"

FC_FLUX_SCALER_FILENAME = f"fc_flux_scaler{PICKLE_SUFFIX}"
CNN_FLUX_SCALER_FILENAME = f"convolutional_flux_scaler{PICKLE_SUFFIX}"
AUTOENCODER_FLUX_SCALER_FILENAME = f"autoencoder_flux_scaler{PICKLE_SUFFIX}"

FC_AMP_PHASE_FILENAME_PREFIX = f"fully_connected_amp_phase"
CNN_AMP_PHASE_FILENAME_PREFIX = f"convolutional_amp_phase"
AUTOENCODER_FLUXES_FILENAME_PREFIX = "autoencoder_amp_phase"

FC_AMP_SCALER_FILENAME = f"fc_amp_scaler{PICKLE_SUFFIX}"
CNN_AMP_SCALER_FILENAME = f"convolutional_amp_scaler{PICKLE_SUFFIX}"
AUTOENCODER_AMP_SCALER_FILENAME = f"autoencoder_amp_scaler{PICKLE_SUFFIX}"

FC_PHASE_SCALER_FILENAME = f"fc_phase_scaler{PICKLE_SUFFIX}"
CNN_PHASE_SCALER_FILENAME = f"convolutional_phase_scaler{PICKLE_SUFFIX}"
AUTOENCODER_PHASE_SCALER_FILENAME = f"autoencoder_phase_scaler{PICKLE_SUFFIX}"

# TRAIN, VALIDATION AND TEST SUFFIXES FOR THE AMPLITUDE AND PHASE FILES
TRAIN_AMP_PHASE_FILE_SUFFIXES = ["00", "01", "02", "03", "04", "05", "06"]
VAL_AMP_PHASE_FILE_SUFFIX = f"07"
TEST_AMP_PHASE_FILE_SUFFIX = f"08"

# PROCESSED DATA PATHS
# - Fully connected architecture
TRAIN_FC_FLUXES_PATH = f"{TRAIN_FOLDER}/{FC_FLUXES_FILENAME}"
VALIDATION_FC_FLUXES_PATH = f"{VALIDATION_FOLDER}/{FC_FLUXES_FILENAME}"
TEST_FC_FLUXES_PATH = f"{TEST_FOLDER}/{FC_FLUXES_FILENAME}"

TRAIN_FC_AMP_PHASE_PATH = f"{TRAIN_FOLDER}/{FC_AMP_PHASE_FILENAME_PREFIX}"
VALIDATION_FC_AMP_PHASE_PATH = f"{VALIDATION_FOLDER}/{FC_AMP_PHASE_FILENAME_PREFIX}{NUMPY_SUFFIX}"
TEST_FC_AMP_PHASE_PATH = f"{TEST_FOLDER}/{FC_AMP_PHASE_FILENAME_PREFIX}{NUMPY_SUFFIX}"

FC_FLUX_SCALER_PATH = f"{SCALERS_FOLDER}/{FC_FLUX_SCALER_FILENAME}"
FC_AMP_SCALER_PATH = f"{SCALERS_FOLDER}/{FC_AMP_SCALER_FILENAME}"
FC_PHASE_SCALER_PATH = f"{SCALERS_FOLDER}/{FC_PHASE_SCALER_FILENAME}"

# - Convolutional architecture
TRAIN_CNN_FLUXES_PATH = f"{TRAIN_FOLDER}/{CNN_FLUXES_FILENAME}"
VALIDATION_CNN_FLUXES_PATH = f"{VALIDATION_FOLDER}/{CNN_FLUXES_FILENAME}"
TEST_CNN_FLUXES_PATH = f"{TEST_FOLDER}/{CNN_FLUXES_FILENAME}"

TRAIN_CNN_AMP_PHASE_PATH = f"{TRAIN_FOLDER}/{CNN_AMP_PHASE_FILENAME_PREFIX}"
VALIDATION_CNN_AMP_PHASE_PATH = f"{TRAIN_FOLDER}/{CNN_AMP_PHASE_FILENAME_PREFIX}{NUMPY_SUFFIX}"
TEST_CNN_AMP_PHASE_PATH = f"{TRAIN_FOLDER}/{CNN_AMP_PHASE_FILENAME_PREFIX}{NUMPY_SUFFIX}"

CNN_FLUX_SCALER_PATH = f"{SCALERS_FOLDER}/{CNN_FLUX_SCALER_FILENAME}"
CNN_AMP_SCALER_PATH = f"{SCALERS_FOLDER}/{CNN_AMP_SCALER_FILENAME}"
CNN_PHASE_SCALER_PATH = f"{SCALERS_FOLDER}/{CNN_PHASE_SCALER_FILENAME}"

# - Autoencoder architecture
TRAIN_AUTOENCODER_FLUXES_PATH = f"{TRAIN_FOLDER}/{AUTOENCODER_FLUXES_FILENAME}"
VALIDATION_AUTOENCODER_FLUXES_PATH = f"{VALIDATION_FOLDER}/{AUTOENCODER_FLUXES_FILENAME}"
TEST_AUTOENCODER_FLUXES_PATH = f"{TEST_FOLDER}/{AUTOENCODER_FLUXES_FILENAME}"


AUTOENCODER_FLUX_SCALER_PATH = f"{SCALERS_FOLDER}/{AUTOENCODER_FLUX_SCALER_FILENAME}"

#OLD STUFF
FLUXES_FOLDER = f"{PL_DATA_PATH}plfluxes_simplepoly__pllabdata_20230625a_superK_slmcube_20230625_complsines-01sp_04"
SLM_FOLDER = f"{PL_DATA_PATH}/slmcube_20230625_complsines-01sp_04_PSFWFs_file"

FLUXES_FILE = "all_fluxes.npy"
AMPLITUDE_FILE = "complexsine_pupamp.npy"
PHASE_FILE = "complexsine_pupphase.npy"


MODELS_FOLDER_PATH = "./Models"
MODELS_DESCRIPTION_FILE_PATH = "./models_descriptions.txt"

