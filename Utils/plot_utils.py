import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_utils import compute_amplitude_and_phase_from_electric_field, \
                       reshape_fc_electric_field_to_real_imaginary_matrix, \
                       compute_center_of_mass, \
                       compute_ratio

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
    top_y_lim=0.5,
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
    log_scale=False
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
                                                x=0.14,
                                                y=0.47,
                                                len=0.3,
                                                thickness=15
                                            ))

    amplitude_heatmap = go.Heatmap(
                                            z=amplitude,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.5,
                                                y=0.47,
                                                len=0.3,
                                                thickness=15
                                    ))

    intenstity_heatmap = go.Heatmap(
                                            z=intensity,
                                            colorscale='viridis',
                                            colorbar=dict(
                                                orientation='h',
                                                x=0.86,
                                                y=0.47,
                                                len=0.3,
                                                thickness=15
                                        ))

    fig.add_trace(phase_heatmap, row=1, col=1)
    fig.add_trace(amplitude_heatmap, row=1, col=2)
    fig.add_trace(intenstity_heatmap, row=1, col=3)

    fig.show()

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


def plot_19_mode_pl_flux(flux):
    """
    Plots the output flux of the PL, measured only in one wavelength
    """
    fig = px.bar(y=flux, x=np.arange(1, len(flux)+1))
    fig.update_xaxes(title_text='Fiber', tickvals=np.arange(len(flux)+1))
    fig.update_yaxes(title_text='Output flux')
    fig.show()



def create_scatter_with_center_of_mass(x_coords, y_coords):
    scatter = go.Scatter(
        x=x_coords, 
        y=y_coords, 
        mode='markers', 
        showlegend=False,
        marker_color='blue')

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

    return scatter, x_mass_line, y_mass_line


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


def plot_zernike_euclidean_distances(
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
    
    m2_corr = np.corrcoef(m2_fluxes, m2_psf)[0, 1]
    m5_corr = np.corrcoef(m5_fluxes, m5_psf)[0, 1]
    m9_corr = np.corrcoef(m9_fluxes, m9_psf)[0, 1]
    m14_corr = np.corrcoef(m14_fluxes, m14_psf)[0, 1]
    m20_corr = np.corrcoef(m20_fluxes, m20_psf)[0, 1]

    fig = make_subplots(
        rows=3, 
        cols=2, 
        subplot_titles=(
            f"PL vs 2 Mode Zernike PSF<br>Correlation: {round(m2_corr, 2)}",
            f"PL vs 5 Mode Zernike PSF<br>Correlation: {round(m5_corr, 2)}",
            f"PL vs 9 Mode Zernike PSF<br>Correlation: {round(m9_corr, 2)}",
            f"PL vs 14 Mode Zernike PSF<br>Correlation: {round(m14_corr, 2)}",
            f"PL vs 20 Mode Zernike PSF<br>Correlation: {round(m20_corr, 2)}"))

    m2_scatter, m2_mass_x, m2_mass_y = create_scatter_with_center_of_mass(m2_fluxes, 
                                                                          m2_psf)

    m5_scatter, m5_mass_x, m5_mass_y = create_scatter_with_center_of_mass(m5_fluxes, 
                                                                          m5_psf)

    m9_scatter, m9_mass_x, m9_mass_y = create_scatter_with_center_of_mass(m9_fluxes, 
                                                                          m9_psf)

    m14_scatter, m14_mass_x, m14_mass_y = create_scatter_with_center_of_mass(m14_fluxes, 
                                                                             m14_psf)

    m20_scatter, m20_mass_x, m20_mass_y = create_scatter_with_center_of_mass(m20_fluxes, 
                                                                             m20_psf)

    fig.add_trace(m2_scatter, row=1, col=1)
    fig.add_trace(m2_mass_x, row=1, col=1)
    fig.add_trace(m2_mass_y, row=1, col=1)

    fig.add_trace(m5_scatter, row=1, col=2)
    fig.add_trace(m5_mass_x, row=1, col=2)
    fig.add_trace(m5_mass_y, row=1, col=2)


    fig.add_trace(m9_scatter, row=2, col=1)
    fig.add_trace(m9_mass_x, row=2, col=1)
    fig.add_trace(m9_mass_y, row=2, col=1)


    fig.add_trace(m14_scatter, row=2, col=2)
    fig.add_trace(m14_mass_x, row=2, col=2)
    fig.add_trace(m14_mass_y, row=2, col=2)

    fig.add_trace(m20_scatter, row=3, col=1)
    fig.add_trace(m20_mass_x, row=3, col=1)
    fig.add_trace(m20_mass_y, row=3, col=1)


    title = "Zernike Euclidean distances"
    if suffix is not None:
        title += f"in train subset {suffix}"
    fig.update_layout(
        title_text=title,
        height=1050,  # Set the height of the figure
        width=1000    # Set the width of the figure
    )

    fig.update_xaxes(title_text='PL Fluxes euclidean distance')
    fig.update_yaxes(title_text='PSF Intensity euclidean distance')

    fig.update_traces(
        marker=dict(size=1)
        )
    #fig.show()
    fig.write_image(f"{title}.png")

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