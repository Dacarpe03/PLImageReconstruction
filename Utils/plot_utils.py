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


def plot_amplitude_phase_intensity(
    electric_field,
    log_scale=False,
    plot=True,
    save=False,
    title="",
    title_prefix="pid"
    ):
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
    Fuction that from an electric field represented by a matrix of complex numbers, computes amplitude, phase and intensity and plots them in heatmap
    
    Input:
        complex_field (np.array): A numpy array containing the electric field complex numbers

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
    Function that plots the amplitude and phase, both original and predicted

    Input:
        model (keras.model): The model that will predict the electric field in the pupil plane
        output_flux (np.array): The input that the model will predict from
        original_complex_field (np.array): The original electric field in a flattened shape (1, realpartsize + imaginarypartsize)

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
    """
    fig = px.bar(y=flux, x=np.arange(1, len(flux)+1))
    fig.update_xaxes(title_text='Fiber', tickvals=np.arange(len(flux)+1))
    fig.update_yaxes(title_text='Output flux')
    fig.show()



def create_scatter_with_center_of_mass(x_coords, y_coords, name='Untitled'):
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



def plot_euclidean_distances(
    pl_flux_distances,
    og_complex_field_distances,
    cropped_complex_field_distances,
    predicted_complex_field_distances,
    predicted_cropped_complex_field_distances,
    suffix=None
    ):
    
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


def plot_pl_lp_zc_euclidean_distances(
    pl_distances,
    lp_distances,
    zc_distances,
    modes,
    suffix=None,
    show=False,
    save_image=True
    ):

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
    cluster_type
    ):

    samples = []
    
    middles = []
    ticktexts = []
    yboxes=[0]
    
    for label_type in labels_list:
        subsamples = data[data_labels==label_type]
        if len(subsamples) > 0:
            finish = min(len(subsamples), 10)
            subsamples = subsamples[0:finish]
            samples.append(subsamples)

            yboxes.append(yboxes[-1]+finish)
            middles.append(finish/2)
            ticktexts.append(label_type)

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
        width=500,
        height=800
    )

    fig = go.Figure(data=[heatmap], layout=layout)

    for i in range(0, len(ticktexts)):
        fig.add_shape(
            type="rect",
            x0=-0.5, y0=yboxes[i]-0.5, x1=len(data[0])-0.5, y1=yboxes[i+1]-0.5,
            line=dict(color="red", width=2)
        )

    tickvals = []
    tick = len(samples)
    for middle in middles:
        tickval = tick - middle
        tick = tick - middle*2
        tickvals.append(tickval)

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
    nbrs = NearestNeighbors(n_neighbors=neighbours).fit(data)
    distances, indices = nbrs.kneighbors(data)

    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.ylabel(f"{neighbours}-NN Distance ")
    plt.xlabel("Points sorted by distance to nearest neighbours")
    plt.title(f"{neighbours}-NN Distance Graph")
    plt.show()


def get_number_of_clusters(labels):
    #hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=neighbours)
    #labels = hdbscan_clusterer.fit_predict(data)
    print("Number of clusters:", np.max(labels)+1)
    mask = labels != -1
    points_that_are_not_noise = np.sum(mask)
    print("Numbers that are not noise:", points_that_are_not_noise)
    
    return labels

def plot_cluster_labels_count(labels,
                              type_of_clustering,
                              dataset_name):

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
    plt.xticks(range(1, len(integers), 2), rotation=90)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title(f'Label frequency in the {dataset_name} {type_of_clustering} clustering')
    plt.savefig(f'mdid-{dataset_name}{type_of_clustering}density.png')
    plt.show()