#!/usr/bin/env python
# coding: utf-8

from psf_constants import PROCESSED_TRAIN_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX, \
                          CROPPED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX, \
                          PROCESSED_VALIDATION_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH, \
                          CROPPED_VALIDATION_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH

from data_utils import load_numpy_data

from modeling_utils import create_fully_connected_architecture_for_amplitude_and_phase_reconstruction, \
                           compile_model, \
                           train_model, \
                           train_model_with_generator, \
                           store_model

from kobol_configurations import ZernikeCroppedFC as ModelConfig

from plot_utils import plot_amplitude_phase_fully_connected_prediction_from_electric_field, \
                       plot_model_history

import tensorflow as tf
 
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


validation_fluxes_array = load_numpy_data(PROCESSED_VALIDATION_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PATH)
validation_complex_fields_array = load_numpy_data(CROPPED_VALIDATION_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PATH)


validation_fluxes_array.shape


validation_complex_fields_array.shape


model_configuration = ModelConfig(name="SuperBigCroppedZernike2MFC", n_samples=70000)
print(model_configuration.get_description())


model = create_fully_connected_architecture_for_amplitude_and_phase_reconstruction(
    *model_configuration.unpack_architecture_hyperparameters()
)


compile_model(
    model,
    *model_configuration.unpack_compilation_hyperparameters()
)


history = train_model_with_generator(
    model,
    PROCESSED_TRAIN_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX,
    CROPPED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX,
    validation_fluxes_array,
    validation_complex_fields_array,
    *model_configuration.unpack_training_hyperparameters()
)


plot_model_history(history, model.name, show_plot=False, save_image=True)


store_model(model,
            model.name,
            model_configuration.get_description(),
            history.history['mean_squared_error'][-1],
            history.history['val_mean_squared_error'][-1],
            psf_model=True)


n = 106

plot_amplitude_phase_fully_connected_prediction_from_electric_field(model,
                                                                    validation_fluxes_array[n],
                                                                    validation_complex_fields_array[n],
                                                                    cropped=True,
                                                                    save_image=True,
                                                                    validation=True,
                                                                    show_plot=False)


n = 2
tf = load_numpy_data(f"{PROCESSED_TRAIN_2M_ZERNIKE_OUTPUT_FLUXES_FILE_PREFIX}00.npy")
tc = load_numpy_data(f"{CROPPED_TRAIN_2M_ZERNIKE_COMPLEX_FIELDS_FILE_PREFIX}00.npy")

plot_amplitude_phase_fully_connected_prediction_from_electric_field(model,
                                                                    tf[n],
                                                                    tc[n],
                                                                    cropped=True,
                                                                    save_image=True,
                                                                    train=True,
                                                                    show_plot=False)

