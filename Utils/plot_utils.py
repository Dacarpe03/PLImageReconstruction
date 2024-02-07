import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    history
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
    plt.ylim(top=2, bottom=0)
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
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Original Amplitude", "Original Phase", "Reconstructed Amplitude", "Reconstructed Phase"))


    og_amplitude_heatmap = go.Heatmap(
                                z=original_amplitude,
                                colorbar_x=-0.2, 
                                colorbar_y=0.8,
                                colorbar=dict(len=0.5))

    og_phase_heatmap = go.Heatmap(
                            z=original_phase, 
                            colorbar_y = 0.8,
                            colorbar=dict(len=0.5))

    re_amplitude_heatmap = go.Heatmap(
                                z=predicted_amplitude, 
                                colorbar_x=-0.2, 
                                colorbar_y=0.2,
                                colorbar=dict(len=0.5))

    re_phase_heatmap = go.Heatmap(
                            z=predicted_phase,
                            colorbar_y=0.2,
                            colorbar=dict(len=0.5))

    fig.add_trace(og_amplitude_heatmap, row=1, col=1)
    fig.add_trace(og_phase_heatmap, row=1, col=2)
    fig.add_trace(re_amplitude_heatmap, row=2, col=1)
    fig.add_trace(re_phase_heatmap, row=2, col=2)

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

    original_amp_phase = np.swapaxes(original_amp_phase, 1, 2)
    original_amp_phase = np.swapaxes(original_amp_phase, 1, 3)
    
    amplitude_prediction = prediction[0][0]
    phase_prediction = prediction[0][1]

    original_amplitude = original_amp_phase[0][0]
    original_phaes = original_amp_phase[0][1]
    
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