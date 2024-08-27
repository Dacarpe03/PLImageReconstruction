import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

from data_utils import compute_amplitude_and_phase_from_electric_field, \
                       compute_intensity_from_electric_field, \
                       reshape_fc_electric_field_to_real_imaginary_matrix, \
                       compute_center_of_mass, \
                       compute_ratio, \
                       separate_zernike_distances

from psf_constants import PSF_TEMP_IMAGES


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
    model_name,
    top_y_lim=None,
    show_plot=True,
    save_image=True
    ):
    """
    Plots the history of the model in a graph

    Input:
        history (): The training history of the model
        model_name (string): The name of the model
        top_y_lim (float): The top y lim of the figure. If not specified it will be automatically set
        show_plot(bool): If True, show the plot
        save_image (bool): If True, save the image 
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
    if show_plot:
        plt.show()

    if save_image:
        img_path = f"{PSF_TEMP_IMAGES}/psf-{model_name}-1-evolution.png"
        plt.savefig(img_path)

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
        original_amp_phase (np.array): Original 3d array containing the amplitude and phase information in the focal plane

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
        model_name (string): The name of the model
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
        original_amp_phase (np.array): Original 3d array containing the amplitude and phase information in the pupil

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
    """
    Function to plot the generated and original amplitude and phase of a diffusion model

    Input:
        original_amp_phase (np.array): The numpy array with an original PSF from the dataset
        diffusion_output (np.array): The numpy array with a generated PSF from the difussion model
    """
    pred_amp = diffusion_output[0]
    pred_phase = diffusion_output[1]
    original_amp = original_amp_phase[0]
    original_phase = original_amp_phase[1]

    plot_amp_phase_prediction(pred_amp,
                              pred_phase,
                              original_amp,
                              original_phase,
                              "Diffusion model")


def plot_amplitude_phase_intensity(
    electric_field,
    log_scale=False,
    plot=True,
    save=False,
    title="",
    title_prefix="pid"
    ):

    """
    FUnction to plot the amplitude phase and intensity given a complex electric field

    Input:
        electric_field (np.array): A 2d complex matrix
        log_scale (bool): If true, use the log scale in the plot
        plot (bool): If True, show the plot
        save (bool): If True, save the plot in a .png
        title (string): The title of the plot
        title_prefix (string): The prefix of the .png
    """
    amplitude, phase = compute_amplitude_and_phase_from_electric_field(electric_field)
    intensity = amplitude**2

    if log_scale:
        amplitude = np.log10((amplitude/amplitude.max()))
        intensity = np.log10((intensity/intensity.max()))

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Phase", "Amplitude", "Intensity"))

    phase_heatmap = go.Heatmap(
                                            z=phase,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                len=0.3,
                                                x=0.10
                                            ))

    amplitude_heatmap = go.Heatmap(
                                            z=amplitude,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                len=0.3,
                                                x=0.5
                                    ))

    intenstity_heatmap = go.Heatmap(
                                            z=intensity,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                len=0.3,
                                                x=0.9
                                        ))

    cross = go.Scatter(
        x=[len(amplitude)/2],
        y=[len(amplitude)/2],
        mode='markers',
        marker=dict(size=10, color='red', symbol='cross'),
        showlegend=False
        )

    fig.add_trace(phase_heatmap, row=1, col=1)
    fig.add_trace(cross, row=1, col=1)
    fig.add_trace(amplitude_heatmap, row=1, col=2)
    fig.add_trace(cross, row=1, col=2)
    fig.add_trace(intenstity_heatmap, row=1, col=3)
    fig.add_trace(cross, row=1, col=3)

    fig.update_layout(
        title_text=title,
        height=500,  # Set the height of the figure
        width=1100    # Set the width of the figure
    )

    if plot:
        fig.show()
    if save:
        new_title = f"{title_prefix}-{title.replace(' ', '').lower()}"
        fig.write_image(f"{new_title}.png")

    return None


def plot_amplitude_phase_from_electric_field(
    original_electric_field,
    predicted_electric_field,
    model_name,
    log_scale=True,
    save_image=False,
    validation=False,
    train=False,
    show_plot=True):
    """
    Fuction that from an electric field represented by a matrix of complex numbers, computes amplitude, phase and intensity and plots them in heatmap
    
    Input:
        original_complex_field (np.array): A numpy array containing the original electric field complex numbers
        predicted_complex_field (np.array): A numpy array containing the original electric field complex numbers
        model_name (string): The name of the model that predicted the output
        log_scale (bool): If True, use logarithmic scale in the plot
        save_image (bool): If True, save the plot in a .png
        validation (bool): True to indicate that it is a validation datapoint, it will change the title of the .png and plot
        train (bool): True to indicate that it is a train datapoint, it will change the title of the .png and plot
        show_plot(bool): If True, show the plot


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
                                        colorscale='twilight',
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
                                        colorscale='twilight',
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
                                colorscale='twilight',
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
    
    cross = go.Scatter(
        x=[len(original_amplitudes)/2],
        y=[len(original_amplitudes)/2],
        mode='markers',
        marker=dict(size=10, color='red', symbol='cross'),
        showlegend=False
        )

    fig.add_trace(cross, row=1, col=1)
    fig.add_trace(cross, row=1, col=2)
    fig.add_trace(cross, row=1, col=3)
    fig.add_trace(cross, row=2, col=1)
    fig.add_trace(cross, row=2, col=2)
    fig.add_trace(cross, row=2, col=3)

    fig.update_layout(
        title_text=f"PSF reconstruction from model {model_name}",
        height=700,  # Set the height of the figure
        width=800    # Set the width of the figure
    )

    if show_plot:
        # Show the plot
        fig.show()

    if save_image:
        if validation:
            img_path = f"{PSF_TEMP_IMAGES}/psf-{model_name}-1-validation.png"
        if train:
            img_path = f"{PSF_TEMP_IMAGES}/psf-{model_name}-1-train.png"

        fig.write_image(img_path)

    return None


def plot_intensity_from_electric_field(
    original_intensity,
    predicted_intensity,
    model_name,
    log_scale=True,
    save_image=False,
    validation=False,
    train=False,
    show_plot=True):
    """
    Fuction that from an electric field intensity and its prediction represented by a matrix plots them in heatmap
    
    Input:
        original_intensity (np.array): A numpy array containing the original electric field complex numbers
        predicted_intensity (np.array): A numpy array containing the original electric field complex numbers
        model_name (string): The name of the model that predicted the output
        log_scale (bool): If True, use logarithmic scale in the plot
        save_image (bool): If True, save the plot in a .png
        validation (bool): True to indicate that it is a validation datapoint, it will change the title of the .png and plot
        train (bool): True to indicate that it is a train datapoint, it will change the title of the .png and plot
        show_plot(bool): If True, show the plot


    Returns:
        None

    """

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Original Intensity", "Predicted Intensity", "Intensity residual"))

    if log_scale:
        original_intensity = np.log10((original_intensity/original_intensity.max()))
        predicted_intensity = np.log10((predicted_intensity/predicted_intensity.max()))
        
    original_intensity_heatmap = go.Heatmap(
                                            z=original_intensity,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.14,
                                                y=-0.4,
                                                len=0.3,
                                                thickness=15
                                            ))

    predicted_intensity_heatmap = go.Heatmap(
                                            z=predicted_intensity,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.5,
                                                y=-0.4,
                                                len=0.3,
                                                thickness=15
                                    ))

    residual_intensity_heatmap = go.Heatmap(
                                            z=original_intensity - predicted_intensity,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.86,
                                                y=-0.4,
                                                len=0.3,
                                                thickness=15
                                        ))

    fig.add_trace(original_intensity_heatmap, row=1, col=1)
    fig.add_trace(predicted_intensity_heatmap, row=1, col=2)
    fig.add_trace(residual_intensity_heatmap, row=1, col=3)
    
    cross = go.Scatter(
        x=[len(original_intensity)/2],
        y=[len(original_intensity)/2],
        mode='markers',
        marker=dict(size=10, color='red', symbol='cross'),
        showlegend=False
        )

    fig.add_trace(cross, row=1, col=1)
    fig.add_trace(cross, row=1, col=2)
    fig.add_trace(cross, row=1, col=3)
    
    fig.update_layout(
        title_text=f"PSF reconstruction from model {model_name}",
        height=350,  # Set the height of the figure
        width=800    # Set the width of the figure
    )

    if show_plot:
        # Show the plot
        fig.show()

    if save_image:
        if validation:
            img_path = f"{PSF_TEMP_IMAGES}/psf-{model_name}-1-validation.png"
        if train:
            img_path = f"{PSF_TEMP_IMAGES}/psf-{model_name}-1-train.png"

        fig.write_image(img_path)

    return None


def plot_amplitude_phase_fully_connected_prediction_from_electric_field(
    model,
    ouput_flux,
    original_electric_field,
    log_scale=True,
    save_image=True,
    validation=False,
    train=False,
    cropped=False,
    show_plot=True
    ):
    """
    Function that plots the amplitude and phase, both original and predicted

    Input:
        model (keras.model): The model that will predict the electric field in the pupil plane
        output_flux (np.array): The input that the model will predict from
        original_complex_field (np.array): The original electric field in a flattened shape (1, realpartsize + imaginarypartsize)
        log_scale (bool): If True, use logarithmic scale in the plot
        save_image (bool): If True, save the plot in a .png
        validation (bool): True to indicate that it is a validation datapoint, it will change the title of the .png and plot
        train (bool): True to indicate that it is a train datapoint, it will change the title of the .png and plot
        cropped (bool): True to indicate that it is a cropped datapoint, it will change the title of the .png and plot
        show_plot(bool): If True, show the plot
    Returns:
        None
    """

    input_output_flux = np.array([ouput_flux])
    predicted_electric_field = model.predict(input_output_flux)[0]

    if cropped:
        reshaped_predicted_electric_field = reshape_fc_electric_field_to_real_imaginary_matrix(predicted_electric_field, 
        og_shape_depth = 2,
        og_shape_rows = 64,
        og_shape_cols = 64)
        reshaped_original_electric_field = reshape_fc_electric_field_to_real_imaginary_matrix(original_electric_field,
        og_shape_depth = 2,
        og_shape_rows = 64,
        og_shape_cols = 64)
    else:
        reshaped_predicted_electric_field = reshape_fc_electric_field_to_real_imaginary_matrix(predicted_electric_field)
        reshaped_original_electric_field = reshape_fc_electric_field_to_real_imaginary_matrix(original_electric_field)

    plot_amplitude_phase_from_electric_field(reshaped_original_electric_field,
                                             reshaped_predicted_electric_field,
                                             model.name,
                                             log_scale=log_scale,
                                             save_image=save_image,
                                             validation=validation,
                                             train=train,
                                             show_plot=show_plot)

    return None


def plot_intensity_fully_connected_prediction_from_electric_field(
    model,
    ouput_flux,
    original_intensity,
    log_scale=True,
    save_image=True,
    validation=False,
    train=False,
    cropped=False,
    show_plot=True
    ):
    """
    Function that plots the intensity, both original and predicted, from a PSF given a model and an output fluxs

    Input:
        model (keras.model): The model that will predict the electric field in the pupil plane
        output_flux (np.array): The input that the model will predict from
        original_complex_field (np.array): The original electric field in a flattened shape (1, realpartsize + imaginarypartsize)
        log_scale (bool): If True, use logarithmic scale in the plot
        save_image (bool): If True, save the plot in a .png
        validation (bool): True to indicate that it is a validation datapoint, it will change the title of the .png and plot
        train (bool): True to indicate that it is a train datapoint, it will change the title of the .png and plot
        cropped (bool): True to indicate that it is a cropped datapoint, it will change the title of the .png and plot
        show_plot(bool): If True, show the plot
    Returns:
        None
    """

    input_output_flux = np.array([ouput_flux])
    predicted_intensity = model.predict(input_output_flux)[0]

    if cropped:
        reshaped_predicted_intensity = predicted_intensity.reshape(64, 64)
        reshaped_original_intensity = original_intensity.reshape(64, 64)
    else:
        reshaped_predicted_intensity = predicted_intensity.reshape(128, 128)
        reshaped_original_intensity = original_intensity.reshape(128, 128)

    plot_intensity_from_electric_field(reshaped_original_intensity,
                                        reshaped_predicted_intensity,
                                        model.name,
                                        log_scale=log_scale,
                                        save_image=save_image,
                                        validation=validation,
                                        train=train,
                                        show_plot=show_plot)

    return None


def plot_19_mode_pl_flux(flux):
    """
    Plots the output flux of the PL, measured only in one wavelength

    Input:
        flux (np.array): The output flux of the PL to plot
    """
    fig = px.bar(y=flux, x=np.arange(1, len(flux)+1))
    fig.update_xaxes(title_text='Fiber', tickvals=np.arange(len(flux)+1))
    fig.update_yaxes(title_text='Output flux')
    fig.show()



def create_scatter_with_center_of_mass(
    x_coords,
    y_coords, 
    name='Untitled'):
    """
    Function that creates a scatter plot given x and y coordinates of a set of points and also its center of mass
    
    Input:
        x_coords (np.array): The x coordinates of the points  
        y_coords (np.array): The y coordinates of the points
        name (string): The name of the scatter plot

    Returns:
        scatter (go.Scatter): The scatter plot of the points
        x_mass_line (go.Scatter): A scatter plot with a vertical line in the x coordinate of the center of mass of the points
        y_mass_line (go.Scatter): A scatter plot with a horizontal line in the y coordinate of the center of mass of the points
    """
    scatter = go.Scatter(
        x=x_coords, 
        y=y_coords, 
        mode='markers', 
        showlegend=False,
        marker_color='blue',
        name=name)

    center_x, center_y = compute_center_of_mass(x_coords, y_coords)

    x_mass_line = go.Scatter(x=[center_x, center_x],
                             y=[np.min(y_coords), np.max(y_coords)],
                             mode='lines',
                                 showlegend=False,
                                 marker_color='coral')

    y_mass_line = go.Scatter(x=[np.min(x_coords), np.max(x_coords)],
                             y=[center_y, center_y],
                             mode='lines',
                             showlegend=False,
                             marker_color='coral')

    corr = np.corrcoef(x_coords, y_coords)[0, 1]

    name+=f"<br>Correlation: {round(corr, 3)}"
    return [scatter, x_mass_line, y_mass_line, name]


def create_histogram_with_center_of_mass(x_coords, y_coords, name='Untitled'):
    """
    Function that creates a 2d histogram of poinst plot given x and y coordinates of a set of points and also its center of mass
    
    Input:
        x_coords (np.array): The x coordinates of the points  
        y_coords (np.array): The y coordinates of the points
        name (string): The name of the scatter plot

    Returns:
        scatter (go.Histogram2d): The histogram plot of the points
        x_mass_line (go.Scatter): A scatter plot with a vertical line in the x coordinate of the center of mass of the points
        y_mass_line (go.Scatter): A scatter plot with a horizontal line in the y coordinate of the center of mass of the points
    """
    scatter = go.Histogram2d(
    x=x_coords,
    y=y_coords,
    nbinsx=150,  # Number of bins in x-direction
    nbinsy=150,  # Number of bins in y-direction
    colorscale='Viridis',
    name=name,
    showscale=False
    )

    center_x, center_y = compute_center_of_mass(x_coords, y_coords)

    x_mass_line = go.Scatter(x=[center_x, center_x],
                             y=[np.min(y_coords), np.max(y_coords)],
                             mode='lines',
                                 showlegend=False,
                                 marker_color='coral')

    y_mass_line = go.Scatter(x=[np.min(x_coords), np.max(x_coords)],
                             y=[center_y, center_y],
                             mode='lines',
                             showlegend=False,
                             marker_color='coral')

    corr = np.corrcoef(x_coords, y_coords)[0, 1]

    name+=f"<br>Correlation: {round(corr, 3)}"
    return [scatter, x_mass_line, y_mass_line, name]



def plot_euclidean_distances(
    pl_flux_distances,
    og_complex_field_distances,
    cropped_complex_field_distances,
    predicted_complex_field_distances,
    predicted_cropped_complex_field_distances,
    suffix=None
    ):
    """
    Function that plots figures that compare euclidean distances ratios. PL FLUX, COMPLEX FIELDS, CROPPED COMPLEX FIELDS, PREDICTED COMPLEX FIELDS, PREDICTED CROPPED COMPLEX FIELD DISTANCES

    Input:
        pl_flux_distances (np.array): The array of distances between pl fluxes pairs
        og_complex_field_distances (np.array): The array of distances between original PSFs intensities pairs
        cropped_complex_field_distances (np.array): The array of distances between cropped PSFs intensities pairs
        predicted_complex_field_distances (np.array): The array of distances between predicted PSFs intensities pairs
        predicted_cropped_complex_field_distances (np.array): The array of distances between cropped predicted PSFs intensities pairs
        suffix (string): The suffix indicating the number of modes of the dataset
    """
    og_corr = np.corrcoef(pl_flux_distances, og_complex_field_distances)[0, 1]
    cr_corr = np.corrcoef(pl_flux_distances, cropped_complex_field_distances)[0, 1]
    pr_corr = np.corrcoef(pl_flux_distances, predicted_complex_field_distances)[0, 1]
    pr_cr_corr = np.corrcoef(pl_flux_distances, predicted_cropped_complex_field_distances)[0, 1]

    fig = make_subplots(
        rows=2, 
        cols=2, 
        subplot_titles=(
            f"PL vs Original PSF<br>Correlation: {round(og_corr, 2)}", 
            f"PL vs Cropped PSF<br>Correlation: {round(cr_corr, 2)}", 
            f"PL vs Predicted PSF<br>Correlation: {round(pr_corr, 2)}", 
            f"PL vs Predicted Cropped PSF<br>Correlation: {round(pr_cr_corr, 2)}"))

    og_scatter, og_mass_x, og_mass_y = create_scatter_with_center_of_mass(pl_flux_distances, 
                                                                          og_complex_field_distances)

    cropped_scatter, cropped_mass_x, cropped_mass_y = create_scatter_with_center_of_mass(pl_flux_distances, 
                                                                                         cropped_complex_field_distances)

    predicted_scatter, predicted_mass_x, predicted_mass_y = create_scatter_with_center_of_mass(pl_flux_distances, 
                                                                                               predicted_complex_field_distances)

    predicted_cropped_scatter, predicted_cr_mass_x, predicted_cr_mass_y = create_scatter_with_center_of_mass(pl_flux_distances, 
                                                                                                             predicted_cropped_complex_field_distances)

    fig.add_trace(og_scatter, row=1, col=1)
    fig.add_trace(og_mass_x, row=1, col=1)
    fig.add_trace(og_mass_y, row=1, col=1)

    fig.add_trace(cropped_scatter, row=1, col=2)
    fig.add_trace(cropped_mass_x, row=1, col=2)
    fig.add_trace(cropped_mass_y, row=1, col=2)


    fig.add_trace(predicted_scatter, row=2, col=1)
    fig.add_trace(predicted_mass_x, row=2, col=1)
    fig.add_trace(predicted_mass_y, row=2, col=1)


    fig.add_trace(predicted_cropped_scatter, row=2, col=2)
    fig.add_trace(predicted_cr_mass_x, row=2, col=2)
    fig.add_trace(predicted_cr_mass_y, row=2, col=2)


    title = "Euclidean distances"
    if suffix is not None:
        title += f"in train subset {suffix}"
    fig.update_layout(
        title_text=title,
        height=700,  # Set the height of the figure
        width=1000    # Set the width of the figure
    )

    fig.update_xaxes(title_text='PL Fluxes euclidean distance')
    fig.update_yaxes(range=[0,120], title_text='PSF Intensity euclidean distance')

    fig.update_traces(
        marker=dict(size=1)
        )
    #fig.show()
    fig.write_image(f"{title}.png")

    return None


def plot_pl42_euclidean_distances(
    pl_flux_distances,
    lp_coeffs_distances,
    og_complex_field_distances,
    suffix=None,
    ):
    """
    Function that plots figures that compare euclidean distances ratios. PL FLUX, COMPLEX FIELDS, CROPPED COMPLEX FIELDS, PREDICTED COMPLEX FIELDS, PREDICTED CROPPED COMPLEX FIELD DISTANCES

    Input:
        pl_flux_distances (np.array): The array of distances between pl fluxes pairs
        og_complex_field_distances (np.array): The array of distances between original PSFs intensities pairs
        cropped_complex_field_distances (np.array): The array of distances between cropped PSFs intensities pairs
        predicted_complex_field_distances (np.array): The array of distances between predicted PSFs intensities pairs
        predicted_cropped_complex_field_distances (np.array): The array of distances between cropped predicted PSFs intensities pairs
        suffix (string): The suffix indicating the number of modes of the dataset
    """
    pl_flux_distances = pl_flux_distances.flatten()
    lp_coeffs_distances = lp_coeffs_distances.flatten()
    og_complex_field_distances = og_complex_field_distances.flatten()
    fl_og_corr = np.corrcoef(pl_flux_distances, og_complex_field_distances)[0, 1]
    pl_og_corr = np.corrcoef(lp_coeffs_distances, og_complex_field_distances)[0, 1]
    fl_pl = np.corrcoef(pl_flux_distances, lp_coeffs_distances)[0, 1]

    fig = make_subplots(
        rows=3, 
        cols=1, 
        subplot_titles=(
            f"PL flux vs PSF<br>Correlation: {round(fl_og_corr, 2)}", 
            f"LP coeffs vs PSF<br>Correlation: {round(pl_og_corr, 2)}", 
            f"PL flux vs LP coeffs<br>Correlation: {round(fl_pl, 2)}")
    )

    fl_og_scatter = create_scatter_with_center_of_mass(pl_flux_distances, 
                                                                                   og_complex_field_distances)

    pl_og_scatter = create_scatter_with_center_of_mass(lp_coeffs_distances, 
                                                                                   og_complex_field_distances)

    fl_lp_scatter = create_scatter_with_center_of_mass(pl_flux_distances, 
                                                                                   lp_coeffs_distances)

    fig.add_trace(fl_og_scatter[0], row=1, col=1)
    fig.add_trace(fl_og_scatter[1], row=1, col=1)
    fig.add_trace(fl_og_scatter[2], row=1, col=1)

    fig.add_trace(pl_og_scatter[0], row=2, col=1)
    fig.add_trace(pl_og_scatter[1], row=2, col=1)
    fig.add_trace(pl_og_scatter[2], row=2, col=1)


    fig.add_trace(fl_lp_scatter[0], row=3, col=1)
    fig.add_trace(fl_lp_scatter[1], row=3, col=1)
    fig.add_trace(fl_lp_scatter[2], row=3, col=1)


    title = "Euclidean distances for a 42 mode PL"
    if suffix is not None:
        title += f" ({suffix})"
    fig.update_layout(
        title_text=title,
        height=700,  # Set the height of the figure
        width=500    # Set the width of the figure
    )

    fig.update_traces(
        marker=dict(size=1)
        )
    #fig.show()
    fig.write_image(f"{title}.png")

    return None



def create_scatters_for_zernike_dataset(
        dataset,
        n_modes,
    ):
    """
    Function that creates a list of scatter plots with the euclidean distances

    Input:
        dataset (np.array): An array containing the distances between pairs of points (has everythin, zernike, lp, pl, psfs)
        n_modes (int): The number of zernike modes with which the dataset have been created.

    Returns:
        A list with all the scatter plots.
    """

    fluxes, lp_modes, zernike_coeffs, og_psf, pr_psf, og_cr_psf, pr_cr_psf = separate_zernike_distances(dataset)
    
    fl_to_psf_scatter = create_scatter_with_center_of_mass(fluxes, og_psf, name=f"PL flux vs {n_modes} terms Zernike PSF")
    lp_to_psf_scatter = create_scatter_with_center_of_mass(lp_modes, og_psf, name=f"LP modes vs {n_modes} terms Zernike PSF")

    fl_to_pr_psf_scatter = create_scatter_with_center_of_mass(fluxes, pr_psf, name=f"PL flux vs {n_modes} terms Predicted Zernike PSF")
    lp_to_pr_psf_scatter = create_scatter_with_center_of_mass(lp_modes, pr_psf, name=f"LP modes vs {n_modes} terms Predicted Zernike PSF")

    fl_to_cr_psf_scatter = create_scatter_with_center_of_mass(fluxes, og_cr_psf, name=f"PL flux vs {n_modes} terms Cropped Zernike PSF")
    lp_to_cr_psf_scatter = create_scatter_with_center_of_mass(lp_modes, og_cr_psf, name=f"LP modes vs {n_modes} terms Cropped Zernike PSF")

    fl_to_pr_cr_psf_scatter = create_scatter_with_center_of_mass(fluxes, pr_cr_psf, name=f"PL flux vs {n_modes} terms Predicted Cropped Zernike PSF")
    lp_to_pr_cr_psf_scatter = create_scatter_with_center_of_mass(lp_modes, pr_cr_psf, name=f"LP modes vs {n_modes} terms Predicted Cropped Zernike PSF")

    fl_to_lp_scatter = create_scatter_with_center_of_mass(fluxes, lp_modes, name=f"PL flux vs LP modes coefficients")

    return [fl_to_psf_scatter, lp_to_psf_scatter, fl_to_pr_psf_scatter, lp_to_pr_psf_scatter, fl_to_cr_psf_scatter, lp_to_cr_psf_scatter, fl_to_pr_cr_psf_scatter, lp_to_pr_cr_psf_scatter, fl_to_lp_scatter]


def plot_one_dataset_zernike_euclidean_distances(
    dataset,
    modes,
    suffix=None,
    show=False,
    save_image=True
    ):
    
    """
    A function that plots a list of scatter plots from a euclidean distance dataset

     Input:
        dataset (np.array): An array containing the distances between pairs of points (has everythin, zernike, lp, pl, psfs)
        n_modes (int): The number of zernike modes with which the dataset have been created.
        suffix (string): The suffix of the .png file to save
        show (bool): If True, show the plot
        save_image (bool): If True, save the image in a .png
    """
    
    zernike_mode_graphs = create_scatters_for_zernike_dataset(dataset, modes)

    #m2_corr = np.corrcoef(m2_fluxes, m2_psf)[0, 1]
    #m5_corr = np.corrcoef(m5_fluxes, m5_psf)[0, 1]
    #m9_corr = np.corrcoef(m9_fluxes, m9_psf)[0, 1]
    #m14_corr = np.corrcoef(m14_fluxes, m14_psf)[0, 1]
    #m20_corr = np.corrcoef(m20_fluxes, m20_psf)[0, 1]

    subplot_titles = []
    zm_titles = []
    for graph_info in zernike_mode_graphs:
        subplot_titles.append(graph_info[-1])

    fig = make_subplots(
        rows=9, 
        cols=1, 
        subplot_titles=subplot_titles
    )

    col=1
    row=1
    for graph in zernike_mode_graphs:
        print(f"Row {row}")
        scatter = graph[0]
        mass_x = graph[1]
        mass_y = graph[2]
        fig.add_trace(scatter, row=row, col=col)
        fig.add_trace(mass_x, row=row, col=col)
        fig.add_trace(mass_y, row=row, col=col)
        row+=1

    fig.update_xaxes(title_text="PL intensities distance", col=1, row=1)
    fig.update_xaxes(title_text="LP coefficients distance", col=1, row=2)
    fig.update_xaxes(title_text="PL intensities distance", col=1, row=3)
    fig.update_xaxes(title_text="LP coefficients distance", col=1, row=4)
    fig.update_xaxes(title_text="PL intensities distance", col=1, row=5)
    fig.update_xaxes(title_text="LP coefficients distance", col=1, row=6)
    fig.update_xaxes(title_text="PL intensities distance", col=1, row=7)
    fig.update_xaxes(title_text="LP coefficients distance", col=1, row=8)
    fig.update_xaxes(title_text="PL intensities distance", col=1, row=9)

    fig.update_yaxes(title_text="PSF distance", col=1, row=1)
    fig.update_yaxes(title_text="PSF distance", col=1, row=2)
    fig.update_yaxes(title_text="PSF distance", col=1, row=3)
    fig.update_yaxes(title_text="PSF distance", col=1, row=4)
    fig.update_yaxes(title_text="PSF distance", col=1, row=5)
    fig.update_yaxes(title_text="PSF distance", col=1, row=6)
    fig.update_yaxes(title_text="PSF distance", col=1, row=7)
    fig.update_yaxes(title_text="PSF distance", col=1, row=8)
    fig.update_yaxes(title_text="LP coefficients distance", col=1, row=9)

    title = f"{modes} modes"
    if suffix is not None:
        title += f"in train subset {suffix}"
    fig.update_layout(
        title_text=title,
        height=1900,  # Set the height of the figure
        width=400    # Set the width of the figure
    )

    #fig.update_xaxes(title_text='PL Fluxes euclidean distance')
    #fig.update_yaxes(title_text='PSF Intensity euclidean distance')

    fig.update_traces(
        marker=dict(size=1)
        )

    if show:
        fig.show()

    if save_image:
        print("Saving image")   
        fig.write_image(f"{title}.jpg")

    return None


def plot_psf_vs_pl_lp_zc_euclidean_distances(
    psf_distances,
    pl_distances,
    lp_distances,
    zc_distances,
    modes,
    psf_type,
    suffix=None,
    show=False,
    save_image=True
    ):
    """
    Function that plots psf distancs vs coefficients and pl

    Input:
        psf_distances (np.array): An array with euclidean distances from psf pairs
        pl_distances (np.array): An array with euclidean distances from pl output pairs
        lp_distances (np.array): An array with euclidean distances from lp coefficient pairs
        zc_distances (np.array): An array with euclidean distances from zernik coefficient pairs
        modes (int): The number of zernike modes
        psf_type (string): Either original croped or predicted

        suffix (string): The suffix of the .png file to save
        show (bool): If True, show the plot
        save_image (bool): If True, save the image in a .png
    """
    
    fl_to_psf_scatter = create_scatter_with_center_of_mass(pl_distances, psf_distances, name=f"PL flux vs PSF")
    lp_to_psf_scatter = create_scatter_with_center_of_mass(lp_distances, psf_distances, name=f"LP coefficients vs PSF")
    zm_to_psf_scatter = create_scatter_with_center_of_mass(zc_distances, psf_distances, name=f"Zernike coefficients vs PSF")

    zernike_mode_graphs = [fl_to_psf_scatter,
                           lp_to_psf_scatter,
                           zm_to_psf_scatter]

    subplot_titles = []
    zm_titles = []
    for graph_info in zernike_mode_graphs:
        subplot_titles.append(graph_info[-1])

    fig = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=subplot_titles
    )

    col=1
    row=1
    for graph in zernike_mode_graphs:
        print(f"Row {row}")
        scatter = graph[0]
        mass_x = graph[1]
        mass_y = graph[2]
        fig.add_trace(scatter, row=row, col=col)
        fig.add_trace(mass_x, row=row, col=col)
        fig.add_trace(mass_y, row=row, col=col)
        col+=1

    fig.update_xaxes(title_text="PL intensities distance", row=1, col=1)
    fig.update_xaxes(title_text="LP coefficients distance", row=1, col=2)
    fig.update_xaxes(title_text="Zernike coefficients distance", row=1, col=3)

    fig.update_yaxes(title_text="PSF intensity distance", row=1, col=1)
    fig.update_yaxes(title_text="PSF intensity distance", row=1, col=2)
    fig.update_yaxes(title_text="PSF intensity distance", row=1, col=3)

    title = f"{modes} Zernike modes {psf_type} PSF"

    if suffix is not None:
        title += f"in train subset {suffix}"
    fig.update_layout(
        title_text=title,
        height=400,  # Set the height of the figure
        width=1200 ,   # Set the width of the figure
        title={
            'font': {
                'size': 24  # Increase the font size

            },
            'xanchor':'left'
        }
    )

    #fig.update_xaxes(title_text='PL Fluxes euclidean distance')
    #fig.update_yaxes(title_text='PSF Intensity euclidean distance')

    fig.update_traces(
        marker=dict(size=1)
        )

    if show:
        fig.show()

    if save_image:
        print("Saving image")
        file_title=f"pid-{modes}m{psf_type}psfdistances"
        fig.write_image(f"{file_title}.png")

    return None


def plot_psf_vs_pl_lp_zc_euclidean_distances_histogram(
    psf_distances,
    pl_distances,
    lp_distances,
    zc_distances,
    modes,
    psf_type,
    suffix=None,
    show=False,
    save_image=True
    ):
    """
    Function that plots psf distancs vs coefficients and pl in a 2d histogram

    Input:
        psf_distances (np.array): An array with euclidean distances from psf pairs
        pl_distances (np.array): An array with euclidean distances from pl output pairs
        lp_distances (np.array): An array with euclidean distances from lp coefficient pairs
        zc_distances (np.array): An array with euclidean distances from zernik coefficient pairs
        modes (int): The number of zernike modes
        psf_type (string): Either original croped or predicted
        
        suffix (string): The suffix of the .png file to save
        show (bool): If True, show the plot
        save_image (bool): If True, save the image in a .png
    """
    
    fl_to_psf_scatter = create_histogram_with_center_of_mass(pl_distances, psf_distances, name=f"PL flux vs PSF")
    lp_to_psf_scatter = create_histogram_with_center_of_mass(lp_distances, psf_distances, name=f"LP coefficients vs PSF")
    zm_to_psf_scatter = create_histogram_with_center_of_mass(zc_distances, psf_distances, name=f"Zernike coefficients vs PSF")

    zernike_mode_graphs = [fl_to_psf_scatter,
                           lp_to_psf_scatter,
                           zm_to_psf_scatter]

    subplot_titles = []
    zm_titles = []
    for graph_info in zernike_mode_graphs:
        subplot_titles.append(graph_info[-1])

    fig = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=subplot_titles
    )

    col=1
    row=1
    for graph in zernike_mode_graphs:
        print(f"Row {row}")
        scatter = graph[0]
        mass_x = graph[1]
        mass_y = graph[2]
        fig.add_trace(scatter, row=row, col=col)
        fig.add_trace(mass_x, row=row, col=col)
        fig.add_trace(mass_y, row=row, col=col)
        col+=1

    fig.update_xaxes(title_text="PL intensities distance", row=1, col=1)
    fig.update_xaxes(title_text="LP coefficients distance", row=1, col=2)
    fig.update_xaxes(title_text="Zernike coefficients distance", row=1, col=3)

    fig.update_yaxes(title_text="PSF intensity distance", row=1, col=1)
    fig.update_yaxes(title_text="PSF intensity distance", row=1, col=2)
    fig.update_yaxes(title_text="PSF intensity distance", row=1, col=3)

    title = f"{modes} Zernike modes {psf_type} PSF"

    if suffix is not None:
        title += f"in train subset {suffix}"
    fig.update_layout(
        title_text=title,
        height=400,  # Set the height of the figure
        width=1200 ,   # Set the width of the figure
        title={
            'font': {
                'size': 24  # Increase the font size

            },
            'xanchor':'left'
        }
    )

    #fig.update_xaxes(title_text='PL Fluxes euclidean distance')
    #fig.update_yaxes(title_text='PSF Intensity euclidean distance')

    #fig.update_traces(
    #    marker=dict(size=1)
    #    )

    if show:
        fig.show()

    if save_image:
        print("Saving image")
        file_title=f"pid-{modes}m{psf_type}psfdistances"
        fig.write_image(f"{file_title}.png")

    return None


def plot_pl_lp_zc_euclidean_distances(
    pl_distances,
    lp_distances,
    zc_distances,
    modes,
    suffix=None,
    show=False,
    save_image=True
    ):
    """
    Function that plots euclidean distances between coefficients and pl in a scatter plot

    Input:
        pl_distances (np.array): An array with euclidean distances from pl output pairs
        lp_distances (np.array): An array with euclidean distances from lp coefficient pairs
        zc_distances (np.array): An array with euclidean distances from zernike coefficient pairs
        modes (int): The number of zernike modes
        suffix (string): The suffix of the .png file to save
        show (bool): If True, show the plot
        save_image (bool): If True, save the image in a .png
    """
    fl_to_lp_scatter = create_scatter_with_center_of_mass(pl_distances, lp_distances, name=f"PL flux vs LP coefficients")
    fl_to_zc_scatter = create_scatter_with_center_of_mass(pl_distances, zc_distances, name=f"PL flux vs Zernike coefficients")
    pl_to_zc_scatter = create_scatter_with_center_of_mass(lp_distances, zc_distances, name=f"LP coefficients vs Zernike coefficients")

    zernike_mode_graphs = [fl_to_lp_scatter,
                           fl_to_zc_scatter,
                           pl_to_zc_scatter]

    subplot_titles = []
    zm_titles = []
    for graph_info in zernike_mode_graphs:
        subplot_titles.append(graph_info[-1])

    fig = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=subplot_titles
    )

    col=1
    row=1
    for graph in zernike_mode_graphs:
        print(f"Row {row}")
        scatter = graph[0]
        mass_x = graph[1]
        mass_y = graph[2]
        fig.add_trace(scatter, row=row, col=col)
        fig.add_trace(mass_x, row=row, col=col)
        fig.add_trace(mass_y, row=row, col=col)
        col+=1

    fig.update_xaxes(title_text="PL intensities distance", row=1, col=1)
    fig.update_xaxes(title_text="PL intensities distance", row=1, col=2)
    fig.update_xaxes(title_text="LP coefficients distance", row=1, col=3)

    fig.update_yaxes(title_text="LP coefficients distance", row=1, col=1)
    fig.update_yaxes(title_text="Zernike coefficients distance", row=1, col=2)
    fig.update_yaxes(title_text="Zernike coefficients distance", row=1, col=3)

    title = f"PL and coefficients relationship (from {modes} Zernike modes PSF)"

    if suffix is not None:
        title += f"in train subset {suffix}"
    fig.update_layout(
        title_text=title,
        height=400,  # Set the height of the figure
        width=1200 ,   # Set the width of the figure
        title={
            'font': {
                'size': 24  # Increase the font size

            },
            'xanchor':'left'
        }
    )

    #fig.update_xaxes(title_text='PL Fluxes euclidean distance')
    #fig.update_yaxes(title_text='PSF Intensity euclidean distance')

    fig.update_traces(
        marker=dict(size=1)
        )

    if show:
        fig.show()

    if save_image:
        print("Saving image")
        file_title=f"pid-{modes}mcoefficientsdistances"
        fig.write_image(f"{file_title}.png")

    return None


def plot_zernike_euclidean_distances(
    datasets,
    suffix=None,
    show=False,
    save_image=True
    ):
    
    """
    Function that plots all euclidean distances from all datasets

    Input:
        distances (np.array): An array with euclidean distances from dataset point pairs        
        suffix (string): The suffix of the .png file to save
        show (bool): If True, show the plot
        save_image (bool): If True, save the image in a .png
    """
    modes = [2, 5, 9, 14, 20]
    i=0
    all_graphs = []
    for dataset in datasets:
        print(f"{i} modes")
        zernike_mode_graphs = create_scatters_for_zernike_dataset(dataset, modes[i])
        all_graphs.append(zernike_mode_graphs)
        i+=1

    #m2_corr = np.corrcoef(m2_fluxes, m2_psf)[0, 1]
    #m5_corr = np.corrcoef(m5_fluxes, m5_psf)[0, 1]
    #m9_corr = np.corrcoef(m9_fluxes, m9_psf)[0, 1]
    #m14_corr = np.corrcoef(m14_fluxes, m14_psf)[0, 1]
    #m20_corr = np.corrcoef(m20_fluxes, m20_psf)[0, 1]

    subplot_titles = []
    for zernike_mode_graphs in all_graphs:
        zm_titles = []
        for graph_info in zernike_mode_graphs:
            zm_titles.append(graph_info[-1])
        subplot_titles.append(zm_titles)

    subplot_titles = list(zip(*subplot_titles))
    subplot_titles = [item for sublist in subplot_titles for item in sublist]
    fig = make_subplots(
        rows=8, 
        cols=5, 
        subplot_titles=subplot_titles
    )

    col=1
    for zernike_mode_graphs in all_graphs:
        print(f"Col {col}")
        row=1
        for graph in zernike_mode_graphs:
            print(f"Row {row}")
            scatter = graph[0]
            mass_x = graph[1]
            mass_y = graph[2]
            fig.add_trace(scatter, row=row, col=col)
            fig.add_trace(mass_x, row=row, col=col)
            fig.add_trace(mass_y, row=row, col=col)
            row+=1

        col+=1

    title = "Zernike Euclidean distances"
    if suffix is not None:
        title += f"in train subset {suffix}"
    fig.update_layout(
        title_text=title,
        height=1500,  # Set the height of the figure
        width=2000    # Set the width of the figure
    )

    #fig.update_xaxes(title_text='PL Fluxes euclidean distance')
    #fig.update_yaxes(title_text='PSF Intensity euclidean distance')

    fig.update_traces(
        marker=dict(size=1)
        )

    if show:
        fig.show()

    if save_image:
        print("Saving image")
        fig.write_image(f"{title}.jpg")

    return None


def create_boxplot(data, name=""):
    """
    Creates a boxplot of euclidean distances

    Input:
        data (np.array): A set of distances from pairs of points (either psf, zernike, lp or pl)
        name (string): The name of the plot
    Returns:
        boxplot. (go.Box): The boxplot object
    """
    data_mean = np.mean(data)
    data_std = np.std(data)
    text = f"<br>Ratio mean: {round(data_mean, 2)}<br>Ratio std: {round(data_std, 2)}"
    name += f"\n{text}"
    boxplot = go.Box(y=data, name=name, boxpoints=False, showlegend=False)
    return boxplot


def plot_boxplot_euclidean_distances(
    pl_flux_distances,
    og_complex_field_distances,
    cropped_complex_field_distances,
    predicted_complex_field_distances,
    predicted_cropped_complex_field_distances,
    suffix=None
    ):
    """
    Creates and saves a series of boxplots of euclidean distances from a dataset

    Input:
        pl_flux_distances (np.array): A set of distances from pl fluxes pairs
        og_complex_field_distances (np.array): A set of distances from psf pairs
        cropped_complex_field_distances (np.array): A set of distances from cropped psf pairs
        predicted_complex_field_distances (np.array): A set of distances from predicted psf pairs
        predicted_cropped_complex_field_distances (np.array): A set of distances from predicted cropped psf pairs
        suffix (string): A suffix indicating the dataset name
    """
    fig = go.Figure()

    og_plf_ratio = compute_ratio(og_complex_field_distances, pl_flux_distances)
    cropped_og_plf_ratio = compute_ratio(cropped_complex_field_distances, pl_flux_distances)
    predicted_plf_ratio = compute_ratio(predicted_complex_field_distances, pl_flux_distances)
    predicted_cropped_plf_ratio = compute_ratio(predicted_cropped_complex_field_distances, pl_flux_distances)

    og_boxplot = create_boxplot(og_plf_ratio, name="Original PSF - PL")
    cropped_boxplot = create_boxplot(cropped_og_plf_ratio, name="Cropped PSF - PL")
    predicted_boxplot = create_boxplot(predicted_plf_ratio, name="Predicted PSF - PL")
    predicted_cropped_boxplot = create_boxplot(predicted_cropped_plf_ratio, name="Predicted Cropped PSF - PL")


    fig.add_trace(og_boxplot)
    fig.add_trace(cropped_boxplot)
    fig.add_trace(predicted_boxplot)
    fig.add_trace(predicted_cropped_boxplot)


    title = "Euclidean distance ratios"
    if suffix is not None:
        title += f"in train subset {suffix}"

    fig.update_layout(
        title_text=title,
        height=700,  # Set the height of the figure
        width=1000    # Set the width of the figure
    )

    fig.update_traces(
        marker=dict(size=1)
        )

    fig.update_yaxes(title_text='Ratio')

    #fig.show()
    fig.write_image(f"{title}.png")

    return None


def plot_boxplot_zernike_euclidean_distances(
    m2_fluxes,
    m2_psf,
    m5_fluxes,
    m5_psf,
    m9_fluxes,
    m9_psf,
    m14_fluxes,
    m14_psf,
    m20_fluxes,
    m20_psf,
    suffix=None
    ):
    """
    Plots and saves boxplots for all datasets:

    Input:
        m2_fluxes (np.array): Array containing disntaces between pl fluxes from 2 zernike modes dataset pairs of point
        m2_psf (np.array): Array containing disntaces between psfs from 2 zernike modes dataset pairs of point
        m5_fluxes (np.array): Array containing disntaces between pl fluxes from 5 zernike modes dataset pairs of point
        m5_psf (np.array): Array containing disntaces between psfs from 5 zernike modes dataset pairs of point
        m9_fluxes (np.array): Array containing disntaces between pl fluxes from 9 zernike modes dataset pairs of point
        m9_psf (np.array): Array containing disntaces between psfs from 9 zernike modes dataset pairs of point
        m14_fluxes (np.array): Array containing disntaces between pl fluxes from 14 zernike modes dataset pairs of point
        m14_psf (np.array): Array containing disntaces between psfs from 14 zernike modes dataset pairs of point
        m20_fluxes (np.array): Array containing disntaces between pl fluxes from 20 zernike modes dataset pairs of point
        m20_psf (np.array): Array containing disntaces between psfs from 20 zernike modes dataset pairs of point
        suffix (string): THe suffix indicating the name of the dataset       
    """
    fig = go.Figure()

    m2_ratio = compute_ratio(m2_psf, m2_fluxes)

    m5_ratio = compute_ratio(m5_psf, m5_fluxes)

    m9_ratio = compute_ratio(m9_psf, m9_fluxes)

    m14_ratio = compute_ratio(m14_psf, m14_fluxes)

    m20_ratio = compute_ratio(m20_psf, m20_fluxes)

    m2_boxplot = create_boxplot(m2_ratio, name="2 Mode Zernike PSF - PL")
    m5_boxplot = create_boxplot(m5_ratio, name="5 Mode Zernike PSF - PL")
    m9_boxplot = create_boxplot(m9_ratio, name="9 Mode Zernike PSF - PL")
    m14_boxplot = create_boxplot(m14_ratio, name="14 Mode Zernike PSF - PL")
    m20_boxplot = create_boxplot(m20_ratio, name="20 Mode Zernike PSF - PL")


    fig.add_trace(m2_boxplot)
    fig.add_trace(m5_boxplot)
    fig.add_trace(m9_boxplot)
    fig.add_trace(m14_boxplot)
    fig.add_trace(m20_boxplot)


    title = "Euclidean distance ratios"
    if suffix is not None:
        title += f"in train subset {suffix}"

    fig.update_layout(
        title_text=title,
        height=700,  # Set the height of the figure
        width=1000    # Set the width of the figure
    )

    fig.update_traces(
        marker=dict(size=1)
        )

    fig.update_yaxes(title_text='Ratio')

    #fig.show()
    fig.write_image(f"{title}.png")

    return None


def plot_clusters_from_labels(
    dataset_coordinates,
    labels,
    title,
    x_title,
    y_title,
    dataset_name,
    cluster_type,
    axis_range=[-2.3, 2.3]):
    """
    Function that plots and saves 2d clusters:

    Input:
        dataset_coordinates (np.array): The coordinates of the points
        labels (np.array): The labels of each of the points
        title (string): THe title of the plot
        x_title (string): The name of the x axis
        y_title (string): The name of the y axis
        dataset_name (string): The name of the dataset
        cluser_type (string): The name of the clustering algorithm
        axis_range (list): THe range of the axis
    """

    df = pd.DataFrame(dataset_coordinates, columns=[x_title, y_title])
    df['label'] = labels

    fig = px.scatter(df, x=x_title, y=y_title, color=df['label'].astype(str), title=title)

    fig.update_layout(
        autosize=False,
        legend_title_text='Labels',
        width=600,
        height=600,
        xaxis=dict(scaleanchor='y', scaleratio=1, range=axis_range),
        yaxis=dict(scaleanchor='x', scaleratio=1, range=axis_range)
    )

    fig.show()  
    fig.write_image(f'mdid-{dataset_name}{cluster_type}clusters.png')


def plot_grid_clusters(
    data,
    data_labels,
    labels_list,
    title,
    xtitle,
    ytitle,
    xtickval_jumps,
    dataset_name,
    cluster_type,
    cluster_line_width=2,
    samples_per_cluster=10,
    y_tick_jump_size=1,
    width=500,
    height=800
    ):
    """
    Function that plots and saves samples from the clusters clusters:

    Input:
        data (np.array): The datapoints
        dataset_labels (np.array): The labels of the points
        labels_list (list): A list with all labels
        title (string): THe title of the plot
        x_title (string): The name of the x axis
        y_title (string): The name of the y axis
        dataset_name (string): The name of the dataset
        cluser_type (string): The name of the clustering algorithm
        cluster_line_width (float): The width of the line that encloses the cluster samples
        samples_per_cluster (int): The number of examples per cluster
        y_tick_jump_size (int): The size of the jump between ticks in the y axis
        heigth (float): The height of the plot
        width (float): The width of the plot
    """
    samples = []
    
    middles = []
    ticktexts = []
    yboxes=[0]
    
    label_count = 0
    for label_type in labels_list:
        subsamples = data[data_labels==label_type]
        if len(subsamples) > 0:
            finish = min(len(subsamples), samples_per_cluster)
            subsamples = subsamples[0:finish]
            samples.append(subsamples)

            yboxes.append(yboxes[-1]+finish)
            middles.append(finish/2)

            if label_count % y_tick_jump_size == 0:
                ticktexts.append(label_type)
            label_count+=1

    samples = np.concatenate(samples)
    heatmap = go.Heatmap(
        z=samples,
        colorscale='Viridis'
    )

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=xtitle
        ),
        yaxis=dict(
            title=ytitle
        ),
        width=width,
        height=height
    )

    fig = go.Figure(data=[heatmap], layout=layout)

    for i in range(0, len(labels_list)):
        fig.add_shape(
            type="rect",
            x0=-0.5, y0=yboxes[i]-0.5, x1=len(data[0])-0.5, y1=yboxes[i+1]-0.5,
            line=dict(color="red", width=cluster_line_width)
        )

    tickvals = []
    tick = len(samples)
    for middle in middles:
        tickval = tick - middle
        tick = tick - middle*2
        if label_count % y_tick_jump_size == 0:
            tickvals.append(tickval)
        label_count += 1

    ticktexts.reverse()
    fig.update_yaxes(
        tickvals=tickvals,
        ticktext=ticktexts
    )
    
    fig.update_xaxes(
        tickvals=np.arange(0, len(data[0]), xtickval_jumps),
    )

    fig.update_layout(
        margin=dict(t=100, l=100)
    )

    fig.show()
    fig.write_image(f'mdid-{dataset_name}{cluster_type}gridclusters.png')


def plot_kneighbours(data, neighbours):
    """
    A functiont that plots the number of points that have a numer of neighbours in a distance vs the distance needed for that
    Input:
        ddata (np.array): Datapoints
        neighbours: The number of neighbours
    """
    nbrs = NearestNeighbors(n_neighbors=neighbours).fit(data)
    distances, indices = nbrs.kneighbors(data)

    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.ylabel(f"{neighbours}-NN Distance ")
    plt.xlabel("Points sorted by distance to nearest neighbours")
    plt.title(f"{neighbours}-NN Distance Graph")
    plt.show()


def get_number_of_clusters(labels):
    """
    Function that prints the number of clusters that are not noise
    """
    print("Number of clusters:", np.max(labels)+1)
    mask = labels != -1
    points_that_are_not_noise = np.sum(mask)
    print("Numbers that are not noise:", points_that_are_not_noise)
    
    return labels

def plot_cluster_labels_count(labels,
                              type_of_clustering,
                              dataset_name,
                              xtick_jump_size=1,
                              title=None):
    """
    Function that plots and saves a barplot with the number of points per cluster
    Input:
        labels (np.array): The labels of the points
        type_of_clustering (string): The name of the clusterng algorithm
        dataset_name (string): THe name of the dataset
        title (string): The title of the plot.
    """
    non_noise_labels = labels[labels != -1]
    counter = Counter(non_noise_labels)
    most_common = counter.most_common()[0]
    least_common = counter.most_common()[:-2:-1][0]
    print(f"The most repeated label is {most_common[0]} with {most_common[1]} occurrences.")
    print(f"The least repeated label is {least_common[0]} with {least_common[1]} occurrence.")

    keys = list(counter.keys())
    counts = list(counter.values())
    
    integers = list(counter.keys())
    
    print("Cluster density mean:", np.mean(counts))
    print("Cluster density variance:", np.std(counts))
    plt.bar(keys, counts)
    plt.xticks(range(1, len(integers), xtick_jump_size), rotation=90)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title(f'Label frequency in the {dataset_name} {type_of_clustering} clustering')

    plt.savefig(f"mdid-{dataset_name.lower().replace(' ', '')}{type_of_clustering}density.png")
    plt.show()


def plot_nmi_evolutions_over_clusters(
    number_of_clusters,
    nmi_zernike_psf,
    nmi_zernike_lp,
    nmi_zernike_flux,
    nmi_psf_lp,
    nmi_psf_flux,
    nmi_lp_flux,
    n_modes):
    """
    Plots and saves the evolution of mutual information over number of clusters:

    Input:
        number_of_clusters (list): A list with the number of clusters used
        nmi_zernike_psf (list): The list of mutual information score evolution between zernike and psf
        nmi_zernike_lp (list): The list of mutual information score evolution between zernike and lp
        nmi_zernike_flux (list): The list of mutual information score evolution between zernike and flux
        nmi_psf_lp (list): The list of mutual information score evolution between psf and lp
        nmi_psf_flux (list): The list of mutual information score evolution between psf and flux
        nmi_lp_flux (list): The list of mutual information score evolution between lp and flux
        n_modes (int): The number of modes of the dataset
    """
    
    fig = make_subplots(rows=3, 
                        cols=2,
                        subplot_titles=["Zernike coefficients vs PSF AMI",
                                        "Zernike coefficients vs LP coefficients AMI",
                                        "Zernike coefficients vs PL output fluxes AMI",
                                        "PSF vs LP coefficients AMI",
                                        "PSF vs PL output fluxes AMI",
                                        "LP coefficients vs PL output fluxes AMI"])

    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_psf, mode='lines+markers', name='Zernike coefficients vs PSF AMI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_lp, mode='lines+markers', name='Zernike coefficients vs LP coefficients AMI'), row=1, col=2)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_flux, mode='lines+markers', name='Zernike coefficients vs PL output fluxes AMI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_psf_lp, mode='lines+markers', name='PSF vs LP coefficients AMI'), row=2, col=2)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_psf_flux, mode='lines+markers', name='PSF vs PL output fluxes AMI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_lp_flux, mode='lines+markers', name='LP coefficients vs PL output fluxes AMI'), row=3, col=2)

    # Update layout
    fig.update_layout(height=840,
                      width=1200,
                      title_text=f"AMI evolution over different number of clusters for {n_modes} zernike modes")
    fig.update_yaxes(title_text='AMI',
                     range=[0, 1])
    fig.update_xaxes(title_text='Number of clusters',
                     tickvals=number_of_clusters,
                     type="log")
    fig.show()

    fig.write_image(f'nmia-nmievolutionover{n_modes}.png')


def plot_ami_evolutions_over_clusters_with_complex(
    number_of_clusters,
    nmi_zernike_psf,
    nmi_zernike_complex_psf,
    nmi_zernike_lp,
    nmi_zernike_flux,
    nmi_zernike_complex_flux,
    nmi_psf_lp,
    nmi_complex_psf_lp,
    nmi_psf_flux,
    nmi_complex_psf_complex_flux,
    nmi_lp_flux,
    nmi_lp_complex_flux,
    n_samples,
    best_case=False,
    save=False):
    """
    Plots and saves the evolution of mutual information over number of clusters:

    Input:
        number_of_clusters (list): A list with the number of clusters used
        nmi_zernike_psf (list): The list of mutual information score evolution between zernike and psf 
        nmi_zernike_complex_psf (list): The list of mutual information score evolution between zernike and complex field psf
        nmi_zernike_lp (list): The list of mutual information score evolution between zernike and lp
        nmi_zernike_flux (list): The list of mutual information score evolution between zernike and flux
        nmi_psf_lp (list): The list of mutual information score evolution between psf and lp
        nmi_complex_psf_lp (list): The list of mutual information score evolution between complex field psf and lp
        nmi_psf_flux (list): The list of mutual information score evolution between psf and flux
        nmi_complex_psf_complex_flux (list): The list of mutual information score evolution between complex field psf and fluxes complex amplitudes
        nmi_lp_flux (list): The list of mutual information score evolution between lp and flux
        nmi_lp_complex_flux (list): The list of mutual information score evolution between lp and flux complex amplitudes
        n_samples (int): The number of samples of the dataset
        best_case (bool): If true, print dotted lines on all of the plots with the pl complex amplitud and lp coeffs AMI
        save (bool): If true, save the plot in a .png
    """
    
    fig = make_subplots(rows=6, 
                        cols=2,
                        subplot_titles=["Zernike coefficients vs PSF AMI",
                                        "Zernike coefficients vs complex PSF AMI",
                                        "Zernike coefficients vs LP coefficients AMI",
                                        "",
                                        "Zernike coefficients vs PL output fluxes AMI",
                                        "Zernike coefficients vs complex PL output fluxes AMI",
                                        "PSF vs LP coefficients AMI",
                                        "Complex PSF vs LP coefficients AMI",
                                        "PSF vs PL fluxes AMI",
                                        "Complex PSF vs Complex PL output fluxes AMI",
                                        "LP coefficients vs PL output fluxes AMI",
                                        "LP coefficients vs Complex PL output fluxes AMI"])

    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_psf, mode='lines+markers', name='Zernike coefficients vs PSF AMI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_complex_psf, mode='lines+markers', name='Zernike coefficients vs complex PSF AMI'), row=1, col=2)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_lp, mode='lines+markers', name='Zernike coefficients vs LP coefficients AMI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_flux, mode='lines+markers', name='Zernike coefficients vs PL output fluxes AMI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_complex_flux, mode='lines+markers', name='Zernike coefficients vs complex PL output fluxes AMI'), row=3, col=2)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_psf_lp, mode='lines+markers', name='PSF vs LP coefficients AMI'), row=4, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_complex_psf_lp, mode='lines+markers', name='Complex PSF vs LP coefficients AMI'), row=4, col=2)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_psf_flux, mode='lines+markers', name='PSF vs PL output fluxes AMI'), row=5, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_complex_psf_complex_flux, mode='lines+markers', name='complex PSF vs complex PL output fluxes AMI'), row=5, col=2)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_lp_flux, mode='lines+markers', name='LP coefficients vs PL output fluxes AMI'), row=6, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_lp_complex_flux, mode='lines+markers', name='LP coefficients vs complex PL output fluxes AMI'), row=6, col=2)

    if best_case:

        showlegend=True
        for row in range(1, 7):
            for col in range(1,3):
                if row == 2 and col ==2:
                    print()
                else:
                    fig.add_trace(go.Scatter(x=number_of_clusters, 
                                             y=nmi_lp_complex_flux, line = dict(color='royalblue', width=1, dash='dash'), 
                                             name='Best case', showlegend=showlegend), row=row, col=col)
                showlegend=False
    
    # Update layout
    fig.update_layout(height=1680,
                      width=1200,
                      title_text=f"AMI evolution over different number of clusters for 9 zernike modes and {n_samples}")
    fig.update_yaxes(title_text='AMI',
                     range=[0, 1])
    fig.update_xaxes(title_text='Number of clusters',
                     tickvals=number_of_clusters,
                     type="log")
    fig.show()
    if save:
        fig.write_image(f'ld-amievolutionover{n_samples}.png')


def plot_nmi_evolutions_over_clusters_no_lp(
    number_of_clusters,
    nmi_zernike_psf,
    nmi_zernike_flux,
    nmi_psf_flux,
    n_modes):
    """
    Plots and saves the evolution of mutual information over number of clusters excluding lp:

    Input:
        number_of_clusters (list): A list with the number of clusters used
        nmi_zernike_psf (list): The list of mutual information score evolution between zernike and psf 
        nmi_zernike_flux (list): The list of mutual information score evolution between zernike and flux
        nmi_psf_flux (list): The list of mutual information score evolution between psf and flux
        n_modes (int): The number of modes of the dataset
    """
    
    fig = make_subplots(rows=1, 
                        cols=3,
                        subplot_titles=["Zernike coefficients vs PSF AMI",
                                        "Zernike coefficients vs PL output fluxes AMI",
                                        "PSF vs PL output fluxes AMI"])

    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_psf, mode='lines+markers', name='Zernike coefficients vs PSF AMI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_zernike_flux, mode='lines+markers', name='Zernike coefficients vs PL output fluxes AMI'), row=1, col=2)
    fig.add_trace(go.Scatter(x=number_of_clusters, y=nmi_psf_flux, mode='lines+markers', name='PSF vs PL output fluxes AMI'), row=1, col=3)
    
    # Update layout
    fig.update_layout(height=270,
                      width=1800,
                      title_text=f"AMI evolution over different number of clusters for {n_modes} zernike modes")
    fig.update_yaxes(title_text='AMI',
                     range=[0, 1])
    fig.update_xaxes(title_text='Number of clusters',
                     tickvals=number_of_clusters,
                     type="log")
    fig.show()

    fig.write_image(f'nmia-nmievolutionoverbig{n_modes}.png')


def plot_nmi_evolution_over_modes(
    number_of_clusters,
    nmi_evolutions,
    title):
    """
    Plots and saves the evolution of mutual information over number of clusters excluding lp:

    Input:
        number_of_clusters (list): A list with the number of clusters used
        nmi_evolutions (list): The list of mutual information score evolution between zernike and psf 
        n_modes (int): The number of modes of the dataset
    """
    fig = go.Figure()
    modes = [2, 5, 9, 14, 20, 27, 35, 44]
    for nmi_evo, n_mode in zip(nmi_evolutions, modes):
        fig.add_trace(go.Scatter(x=number_of_clusters, 
                                 y=nmi_evo, mode='lines+markers', 
                                 name=f'{n_mode} modes'))
    
    # Update layout
    fig.update_layout(height=600,
                      width=800,
                      title_text=title)
    fig.update_yaxes(title_text='AMI',
                     range=[0, 1])
    fig.update_xaxes(title_text='Number of clusters',
                     tickvals=number_of_clusters,
                     type="log")
    fig.show()
    fig.write_image(f'nmia-{title.strip().lower().replace(" ", "")}.png')


def plot_ami_evolution_over_dataset_sizes(
    number_of_clusters,
    ami_evolutions,
    title):
    """
    Plots and saves the evolution of mutual information over dataset sizes excluding lp:

    Input:
        number_of_clusters (list): A list with the number of clusters used. REMOVE FROM THE FUNCTION NOT USED.
        ami_evolutions (list): The list of mutual information score evolutions
        title (string): The title of the plot
    """
    fig = go.Figure()
    modes = [500, 1000, 2000, 5000, 10000, 20000]
    max_ami = []
    for ami_evo in ami_evolutions:
        max_ami.append(max(ami_evo))

    fig.add_trace(go.Scatter(x=modes, 
                             y=max_ami, mode='lines+markers'))
    
    # Update layout
    fig.update_layout(height=600,
                      width=800,
                      title_text=title)
    fig.update_yaxes(title_text='AMI',
                     range=[0, 1])
    fig.update_xaxes(title_text='Datasetsize',
                     tickvals=modes,
                     type="log")
    fig.show()
    fig.write_image(f'ld-{title.strip().lower().replace(" ", "")}.png')