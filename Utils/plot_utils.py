import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_utils import compute_amplitude_and_phase_from_electric_field, \
                       reshape_fc_electric_field_to_real_imaginary_matrix

def plot_map(
	whatever_map
	):
	"""
	Plots an amplitude, phase of flux map

	Input:
		whatever_map (np.array): A 2D array containing the map

	Returns:
		None
	"""
	# Input the data into the figure
	fig = go.Figure(data=go.Heatmap(
                    z=whatever_map))
	fig.show()
	return None


def plot_fluxes(original_flux,
                process_flux):
    process_flux = process_flux.reshape(original_flux.shape)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Flux", "Processed  Flux"))

    og_flux_heatmap = go.Heatmap(
                                z=original_flux,
                                colorbar_x=-0.2, 
                                colorbar_y=0.8,
                                colorbar=dict(len=0.5))

    process_flux_heatmap = go.Heatmap(
                            z=process_flux, 
                            colorbar_y = 0.8,
                            colorbar=dict(len=0.5))

    fig.add_trace(og_flux_heatmap, row=1, col=1)
    fig.add_trace(process_flux_heatmap, row=1, col=2)

    fig.update_layout(
    title_text=f"Flux Comparison",
    height=800,  # Set the height of the figure
    width=800    # Set the width of the figure
    )

    # Show the plot
    fig.show()


def plot_model_history(
    history,
    top_y_lim=0.5
    ):
    """
    Plots the history of the model in a graph

    Input:
        history (): The training history of the model

    Returns:
        None
    """
    # Create a dataframe with the results
    results = pd.DataFrame(history.history)
    # Create a plot from the dataframe
    results.plot(figsize=(8,5))
    # Make grid lines visible
    plt.grid(True)
    # Set the x axis title
    plt.xlabel('Epochs')
    # Set the y axis title
    plt.ylabel('Mean Squared Error')
    # Limit the error
    plt.ylim(top=top_y_lim, bottom=0)
    # Show the plot
    plt.show()

    return None


def plot_fully_connected_amp_phase_prediction(
    model,
    input_flux,
    original_amplitude,
    original_phase
    ):
    """
    Plots a 4 figure diagram with the predictions of the model

    Input:
        model (keras.models): A trained neural network
        input_flux (np.array): A data point to feed the neural network
        original_amplitude (np.array): Original 2d array containing the amplitude information in the pupil
        original_phase (np.array): Original 2d array containing the phase information in the pupil

    Returns:
        None
    """

    reshaped_input_flux = input_flux.reshape(1,len(input_flux))
    prediction = model.predict(reshaped_input_flux)
    amplitude_prediction = prediction[0][0]
    phase_prediction = prediction[0][1]

    plot_amp_phase_prediction(
        amplitude_prediction,
        phase_prediction,
        original_amplitude,
        original_phase,
        model.name
    )


def plot_conv_amp_phase_prediction(
    model,
    input_flux,
    original_amp_phase
    ):
    """
    Plots a 4 figure diagram with the predictions of the convolutional model

    Input:
        model (keras.models): A trained neural network
        input_flux (np.array): A data point to feed the neural network
        original_amplitude (np.array): Original 2d array containing the amplitude information in the pupil
        original_phase (np.array): Original 2d array containing the phase information in the pupil

    Returns:
        None
    """

    reshaped_input_flux = np.array([input_flux])
    prediction = model.predict(reshaped_input_flux)[0]
    reshaped_prediction = np.transpose(prediction, (2,0,1))
    amplitude_prediction = reshaped_prediction[0]
    phase_prediction = reshaped_prediction[1]
    original_amplitude = np.transpose(original_amp_phase, (2,0,1))[0]
    original_phase = np.transpose(original_amp_phase, (2,0,1))[1]

    plot_amp_phase_prediction(
        amplitude_prediction,
        phase_prediction,
        original_amplitude,
        original_phase,
        model.name
    )


def plot_amp_phase_prediction(
    predicted_amplitude,
    predicted_phase,
    original_amplitude,
    original_phase,
    model_name
    ):
    """
    Creates a four figure plot with the predicted and original maps of amplitude and phase

    Input:
        predicted_amplitude (np.array): The model amplitude map reconstruction
        predicted_phase (np.array): The model phase map reconstruction
        original_amplitude (np.array): The original amplitude map
        original_phase (np.array): The original phase map
 
    """
    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(rows=2, cols=3, subplot_titles=("Original Amplitude", "Reconstructed Amplitude", "Amplitude Residual", "Original Phase", "Reconstructed Phase", "Phase Residual"))


    og_amplitude_heatmap = go.Heatmap(
                                z=original_amplitude,
                                colorbar_x=-0.2, 
                                colorbar_y=0.8,
                                colorbar=dict(len=0.5))

    og_phase_heatmap = go.Heatmap(
                            z=original_phase,
                            colorbar_x=-0.2,
                            colorbar_y = 0.2,
                            colorbar=dict(len=0.5))

    ai_amplitude_heatmap = go.Heatmap(
                                z=predicted_amplitude, 
                                showscale=False)

    ai_phase_heatmap = go.Heatmap(
                            z=predicted_phase,
                            showscale=False)

    re_amplitude_heatmap = go.Heatmap(
                            z=original_amplitude - predicted_amplitude,
                            colorbar_y=0.8,
                            colorbar=dict(len=0.5))

    re_phase_heatmap = go.Heatmap(
                            z=original_phase - predicted_phase,
                            colorbar_y=0.2,
                            colorbar=dict(len=0.5))

    fig.add_trace(og_amplitude_heatmap, row=1, col=1)
    fig.add_trace(og_phase_heatmap, row=2, col=1)
    fig.add_trace(ai_amplitude_heatmap, row=1, col=2)
    fig.add_trace(ai_phase_heatmap, row=2, col=2)
    fig.add_trace(re_amplitude_heatmap, row=1, col=3)
    fig.add_trace(re_phase_heatmap, row=2, col=3)


    fig.update_layout(
    title_text=f"Amplitude and Phase Reconstruction from {model_name} model",
    height=800,  # Set the height of the figure
    width=800    # Set the width of the figure
    )

    # Show the plot
    fig.show()


def plot_autoencoder(
    model,
    original_flux
    ):
    """
    Plots the autoencoder input and output

    Input:
        model (keras.models): A trained neural network
        original_flux (np.array): A data point to feed the neural network

    Returns:
        None
    """
    reshaped_input_flux = np.array([original_flux])
    prediction = model.predict(reshaped_input_flux)
    reconstructed_flux = prediction[0].reshape(original_flux.shape)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Flux", "Decoded Flux"))

    og_amplitude_heatmap = go.Heatmap(
                                z=original_flux,
                                colorbar_x=-0.2, 
                                colorbar_y=0.8,
                                colorbar=dict(len=0.5))

    og_phase_heatmap = go.Heatmap(
                            z=reconstructed_flux, 
                            colorbar_y = 0.8,
                            colorbar=dict(len=0.5))

    fig.add_trace(og_amplitude_heatmap, row=1, col=1)
    fig.add_trace(og_phase_heatmap, row=1, col=2)

    fig.update_layout(
    title_text=f"Flux AutoEncoder {model.name}",
    height=800,  # Set the height of the figure
    width=800    # Set the width of the figure
    )

    # Show the plot
    fig.show()


def plot_enc_conv_amp_phase_prediction(
    model,
    input_flux,
    original_amp_phase
    ):
    """
    Plots a 4 figure diagram with the predictions of the encoder+convolutional model

    Input:
        model (keras.models): A trained neural network
        input_flux (np.array): A data point to feed the neural network
        original_amplitude_phase (np.array): Original 3d array containing the amplitude and phase information in the pupil

    Returns:
        None
    """

    reshaped_input_flux = np.array([input_flux])
    prediction = model.predict(reshaped_input_flux)
    prediction = np.swapaxes(prediction, 1, 2)
    prediction = np.swapaxes(prediction, 1, 3)

    original_amp_phase = np.swapaxes(original_amp_phase, 0, 1)
    original_amp_phase = np.swapaxes(original_amp_phase, 0, 2)
    
    amplitude_prediction = prediction[0][0]
    phase_prediction = prediction[0][1]

    original_amplitude = original_amp_phase[0]
    original_phase = original_amp_phase [1]
    
    plot_amp_phase_prediction(
        amplitude_prediction,
        phase_prediction,
        original_amplitude,
        original_phase,
        model.name
    )


def plot_diffusion_output(
    original_amp_phase,
    diffusion_output):

    pred_amp = diffusion_output[0]
    pred_phase = diffusion_output[1]
    original_amp = original_amp_phase[0]
    original_phase = original_amp_phase[1]

    plot_amp_phase_prediction(pred_amp,
                              pred_phase,
                              original_amp,
                              original_phase,
                              "Diffusion model")


def plot_amplitude_phase_from_electric_field(
    original_electric_field,
    predicted_electric_field,
    model_name,
    log_scale=True):
    """
    Fuction that from an electric field represented by a matrix of complex numbers, computes amplitude, phase and intensity and plots them in heatmap
    
    Input:
        complex_field (np.array): A numpy array containing the electric field complex numbers

    Returns:
        None

    """
    original_amplitudes, original_phases = compute_amplitude_and_phase_from_electric_field(original_electric_field)    
    predicted_amplitudes, predicted_phases = compute_amplitude_and_phase_from_electric_field(predicted_electric_field)


    fig = make_subplots(rows=2, cols=3, subplot_titles=("Original Amplitude", "Predicted Amplitude", "Amplitude residual",
                                                        "Original Phase", "Predicted Phase", "Phase residual"))

    if log_scale:
        original_amplitudes = np.log10((original_amplitudes/original_amplitudes.max()))
        predicted_amplitudes = np.log10((predicted_amplitudes/predicted_amplitudes.max()))
        
    original_amplitude_heatmap = go.Heatmap(
                                            z=original_amplitudes,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.14,
                                                y=0.47,
                                                len=0.3,
                                                thickness=15
                                            ))

    predicted_amplitude_heatmap = go.Heatmap(
                                            z=predicted_amplitudes,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.5,
                                                y=0.47,
                                                len=0.3,
                                                thickness=15
                                    ))

    residual_amplitude_heatmap = go.Heatmap(
                                            z=original_amplitudes - predicted_amplitudes,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.86,
                                                y=0.47,
                                                len=0.3,
                                                thickness=15
                                        ))

    original_phase_heatmap = go.Heatmap(
                                        z=original_phases,
                                        colorscale='viridis',
                                        colorbar=dict(
                                                orientation='h',
                                                x=0.14,
                                                y=-0.14,
                                                len=0.3,
                                                thickness=15
                                            )
                                        )

    predicted_phase_heatmap = go.Heatmap(
                                        z=predicted_phases,
                                        colorscale='viridis',
                                        colorbar=dict(
                                                orientation='h',
                                                x=0.5,
                                                y=-0.14,
                                                len=0.3,
                                                thickness=15
                                            )
                                        )


    residual_phase_heatmap = go.Heatmap(
                                z=original_phases-predicted_phases,
                                colorscale='viridis',
                                colorbar=dict(
                                    orientation='h',
                                    x=0.86,
                                    y=-0.14,
                                    len=0.3,
                                    thickness=15
                                ))

    fig.add_trace(original_amplitude_heatmap, row=1, col=1)
    fig.add_trace(predicted_amplitude_heatmap, row=1, col=2)
    fig.add_trace(residual_amplitude_heatmap, row=1, col=3)
    fig.add_trace(original_phase_heatmap, row=2, col=1)
    fig.add_trace(predicted_phase_heatmap, row=2, col=2)
    fig.add_trace(residual_phase_heatmap, row=2, col=3)
    
    fig.update_layout(
        title_text=f"PSF reconstruction from model {model_name}",
        height=700,  # Set the height of the figure
        width=800    # Set the width of the figure
    )

    # Show the plot
    fig.show()

    return None


def plot_amplitude_phase_fully_connected_prediction_from_electric_field(
    model,
    ouput_flux,
    original_electric_field,
    log_scale=True
    ):
    """
    Function that plots the amplitude and phase, both original and predicted

    Input:
        model (keras.model): The model that will predict the electric field in the pupil plane
        output_flux (np.array): The input that the model will predict from
        original_complex_field (np.array): The original electric field in a flattened shape (1, realpartsize + imaginarypartsize)

    Returns:
        None
    """

    input_output_flux = np.array([ouput_flux])
    predicted_electric_field = model.predict(input_output_flux)[0]

    reshaped_predicted_electric_field = reshape_fc_electric_field_to_real_imaginary_matrix(predicted_electric_field)
    reshaped_original_electric_field = reshape_fc_electric_field_to_real_imaginary_matrix(original_electric_field)

    plot_amplitude_phase_from_electric_field(reshaped_original_electric_field,
                                             reshaped_predicted_electric_field,
                                             model.name,
                                             log_scale=log_scale)

    return None