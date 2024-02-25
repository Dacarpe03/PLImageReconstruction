import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pickle import dump

from constants import NUMPY_SUFFIX, \
					  SUBFILE_SAMPLES

from psf_constants import PSF_DATA_PATH, \
					  	  LANTERN_FIBER_FILENAME

from plot_utils import plot_map

import os

from hcipy.atmosphere import *
from hcipy import *

import matplotlib.pyplot as plt

from lantern_fiber_utils import LanternFiber
from skimage.transform import resize, rescale


def load_numpy_data(
	data_filepath,
	n_points=None
	):
	"""
	Function that retrieves numpy data given a path

	Input:
		data_filepath (string): The data file path
		n_points (int): The number of points to load

	Returns:
		data (np.array): An array containing the loaded data
	"""
	# If a number of points to sample has been specified, then get the first n_points
	if n_points is not None:
		data = np.load(data_filepath)[0:n_points]
	# Else load the whole file
	else:
		data = np.load(data_filepath)

	return data


def flatten_data(
	data
	):
	"""
	Function that flattens an array of 2d arrays into an array of 1d arrays

	Input:
		data (np.array): The 3d array to flatten

	Returns:
		data (np.array): The 2d array with flattened matrixes into vectors
	"""

	# Obtain the number of data points
	n_points = data.shape[0]
	# Compute the new length of the flattened data point
	flattened_length = np.prod(data.shape[1:])
	# Flatten all the points from the original array
	flattened_data = data.reshape((n_points, flattened_length))

	return flattened_data


def normalize_data(
	data,
	min_max=None,
	):
	"""
	This function scales the data so that it has mean=0 an standard deviation=1

	Input:
		data (np.array): The array to normalize
		min_max (tuple): The range of the values

	Returns:
		normalized_data (np.array): The normalized data
		scaler (sklearn.preprocessing.StandardScaler): The scaler in case we need to unnormalize the data
	"""

	# Reshape the data into a 1d array
	flattened_data = np.expand_dims(data.flatten(), 1)
	# Create a StandardScaler object with mean=0 and std=1
	if min_max is None:
		scaler = StandardScaler(with_mean=True, 
								with_std=True)
	else:
		scaler = MinMaxScaler(feature_range=min_max)

	# Fit the scaler on the data and transform it (normalize)
	flattened_normalized_data = scaler.fit_transform(flattened_data)

	# Reshape the data in the original shape
	normalized_data = flattened_normalized_data.reshape(data.shape)

	return normalized_data, scaler


def split_data(
	data_array,
	val_ratio
	):
	"""
	Function to split a data array into train and validation sets
	Input:
		data_array (np.array): The array to be splitted
		val_ratio (float): The ratio of size between the original array and the validation array
	
	Returns:
		train_array (np.array): The array containing the training set
		val_array (np.array): The array containing the validation set
	"""

	# Compute the length of the train data
	data_lenght = len(data_array)
	train_length = int((1-val_ratio)*data_lenght)

	# Split the dataset into train and validation set with the computed lenght
	train_array = data_array[0:train_length]
	val_array = data_array[train_length:]

	return train_array, val_array


def split_fluxes(
		fluxes_array,
		train_size,
		validation_size,
		test_size
	):
	"""
	Function to split the flux data into train, validation and test sets

	Input:
		fluxes_array (np.array): The fluxes array to be splitted
		train_size (int): The number of data points in the training set
		validation_size (int): The number of data points in the validation set
		test_size (int): The number of data points in the test set
	
	Returns:
		train_fluxes (np.array): The array containing the training fluxes
		validation_fluxes (np.array): The array containing the validation fluxes
		test_fluxes (np.array): The array containing the test fluxes

	"""
	train_fluxes = fluxes_array[0: train_size]
	validation_fluxes = fluxes_array[train_size: train_size + validation_size]
	test_fluxes = fluxes_array[train_size + validation_size: train_size + validation_size + test_size]

	return train_fluxes, validation_fluxes, test_fluxes


def shuffle_arrays(
	array_list
	):
	"""
	Function that applies the same shuffle to a list of arrays

	Input:
		array_list (list): A list with numpy arrays to be shuffled

	Returns:
		shuffled_array_list (list): A list with the numpy arrays shuffled
	"""

	# Compute the number of data points in the array
	array_length = len(array_list[0])
	# Create a new arrangement of indices
	shuffled_indices = np.random.permutation(array_length)

	# Shuffle all the arrays with the new arrangement
	shuffled_array_list = []
	for array in array_list:
		shuffled_array_list.append(array[shuffled_indices])

	# print(array_list[0][0][0], shuffled_array_list[0][0][0])
	return shuffled_array_list


def fuse_amplitude_and_phase(
	amplitudes_array,
	phases_array,
	axis=1
	):
	"""
	Function that stacks the amplitudes and phase arrays into one
	Input:
		amplitudes_array (np.array): An array of shape (n, 96, 96) containing the amplitude maps
		phases_array (np.array): An array of shape (n, 96, 96) containing the phase maps

	Returns:
		amp_phase_array (np.array): An array of shape (n, 2, 96, 96) containing the amplitude maps
	"""
	# Stack both arrays
	amp_phase_array = np.stack([amplitudes_array, phases_array],
								axis=axis)

	return amp_phase_array


def trim_data(
	data_array
	):
	"""
	This function asumes that the important information of the imaged is in the circle inscribed in the 2d matrix, 
	so every cell outside of it turns its value to 0

	Input:
		data_array (np.array): A 3d array containing the original 2d images to trim

	Returns:
		data_array (np.array): A 3d array containing the trimmed 2d images
	"""

	circle_diameter = data_array[0].shape[0]
	circle_radius = circle_diameter/2

	# Create a list of x and y coordinates of the image pixels
	x, y = np.meshgrid(np.arange(circle_diameter), np.arange(circle_diameter))

	# Create a 2d array with the distance of each pixel to the center of the circle
	distance_from_center = np.sqrt((x - circle_radius)**2 + (y - circle_radius)**2)

	# Now create a mask defining which pixels of the matrix are inside the circle
	outside_circle_mask = distance_from_center > circle_radius

	# Apply the mask to the original data, turning the value of the pixels outside the circle to 0
	for i in range(len(data_array)):
		data_array[i][outside_circle_mask] = 0

	return data_array


def add_row_padding(
	data_array,
	top_rows=0,
	bottom_rows=0
	):
	"""
	Function to add a row of 0s to each data point in an array

	Input:
		data_array (np.array): A 3d array containing the 2d data points to pad.
		top_rows (int): The number of zero rows to add to each data point on top of the image
		bottom_rows (int): The number of zero rows to add to each data point at bottom of the image

	Returns:
		padded_data_array (np.array): The 3d array with the 2d padded data points
	"""
	zeros_shape = (data_array[0].shape[1])
	new_data_array = np.zeros((data_array.shape[0], data_array.shape[1] + top_rows, data_array.shape[2]))

	for i in range(len(data_array)):
		new_zero_row = np.zeros((top_rows,zeros_shape))
		new_data_array[i] = np.append(new_zero_row,
							 	  	  data_array[i],
						          	  axis=0)


	final_data_array = np.zeros((new_data_array.shape[0], 
								 new_data_array.shape[1] + bottom_rows, 
								 new_data_array.shape[2]))

	for i in range(len(new_data_array)):
		new_zero_row = np.zeros((bottom_rows,zeros_shape))
		final_data_array[i] = np.append(new_data_array[i],
							 	  	    new_zero_row,
						          	    axis=0)
	
	return final_data_array


def process_amp_phase_data(
	n_points=None,
	trim_amplitude=False,
	trim_phase=False,
	normalize_flux=False,
	normalize_amplitude=False,
	normalize_phase=False,
	shuffle=False,
	flatten_fluxes=False,
	split=False,
	val_ratio=0.1,
	flux_top_padding=0,
	flux_bottom_padding=0,
	amp_phase_top_padding=0,
	amp_phase_bottom_padding=0,
	):
	"""
	Function that retrieves numpy data given a path and processes it

	Input:
		flux_data_filepath (string): The feature data file path
		amplitude_data_filepath (string): The amplitude data file path
		phase_data_filepath (string): The phase data file path
		n_points (int): The number of points to load
		normalize_flux (bool): Indicates wheter or not apply normalization to the flux data
		normalize_amplitude (bool): Indicates wheter or not apply normalization to the amplitude data
		normalize_phase (bool): Indicates whether or not apply normalization to the phase data
		shuffle (bool): Indicates whether or not to shuffle the data (all together: fluxes, amplitudes and phases)
		flatten_fluxes (bool): Indicates whether or not flatten the flux array
		split (bool): Indicates whether or not split the datasets into train and validation sets
		val_ratio (float): The ratio of size between the original array and the validation array
		
	Returns:
		If split is True:
			train_fluxes_array (np.array): The array containing the fluxes training set
			val_fluxes_array (np.array): The array containing the fluxes validation set
			train_amp_phase_array (np.array): The array containing the amplitude+phase training set
			val_amp_phase_array (np.array): The array containing the amplitude+phase validation set
			scalers (list): A list of scalers of each of the normalized arrays (can be empty)

		If split is False:
			fluxes_array (np.array): The array containing the fluxes
			amp_phase_array (np.array): The array containing the amplitudes and phase merged
			scalers (list): A list of scalers of each of the normalized arrays (can be empty)
	"""

	# LOAD DATA
	# If a number of points to sample has been specified, then get the first n_points

	flux_data_filepath = f"{FLUXES_FOLDER}/{FLUXES_FILE}"
	fluxes_array = np.load(flux_data_filepath)

	my_amps = []
	my_phases = []
	for i in range(8):
		amp_name = f"{SLM_FOLDER}0{i}/{AMPLITUDE_FILE}"
		amp_array = np.load(amp_name)
		my_amps.append(amp_array)

		phase_name = f"{SLM_FOLDER}0{i}/{PHASE_FILE}"
		phase_array = np.load(phase_name)
		my_phases.append(phase_array)

	amplitudes_array = np.concatenate(my_amps, axis=0)
	phases_array = np.concatenate(my_phases, axis=0)

	if n_points is not None:
		fluxes_array = fluxes_array[0:n_points]
		amplitudes_array = amplitudes_array[0:n_points]
		phases_array = phases_array[0:n_points]


	# TRIM DATA
	if trim_amplitude:
		amplitudes_array = trim_data(amplitudes_array)
	if trim_phase:
		phases_array = trim_data(phases_array)

	# PADDING DATA
	if flux_top_padding > 0 or flux_bottom_padding > 0:
		fluxes_array = add_row_padding(fluxes_array,
									   flux_top_padding,
									   flux_bottom_padding)

	if amp_phase_top_padding > 0 or amp_phase_bottom_padding > 0:
		amplitudes_array = add_row_padding(amplitudes_array,
									   	   amp_phase_top_padding,
									   	   amp_phase_bottom_padding)

		phases_array = add_row_padding(phases_array,
									   amp_phase_top_padding,
									   amp_phase_bottom_padding)

	# NORMALIZE_DATA
	scalers = []
	if normalize_flux:
		fluxes_array, fluxes_scaler = normalize_data(fluxes_array)
		scalers.append(fluxes_scaler)

	if normalize_amplitude:
		amplitudes_array, amplitudes_scaler = normalize_data(amplitudes_array)
		scalers.append(amplitudes_scaler)

	if normalize_phase:
		phases_array, phases_scaler = normalize_data(phases_array)
		scalers.append(phases_scaler)

	# SHUFFLE DATA
	if shuffle:
		fluxes_array, amplitudes_array, phases_array = shuffle_arrays([fluxes_array,
                                                               		   amplitudes_array,
                                                               		   phases_array])

	# FLATTEN FLUXES
	if flatten_fluxes:
		fluxes_array = flatten_data(fluxes_array)

	# MERGE PHASE AND AMPLITUDE
	amp_phase_array = fuse_amplitude_and_phase(amplitudes_array,
                      					       phases_array)

	# SPLIT DATA
	if split_data:
		train_fluxes_array, val_fluxes_array = split_data(fluxes_array,
                                                                    val_ratio)

		train_amp_phase_array, val_amp_phase_array = split_data(amp_phase_array,
                                                        		val_ratio)

		return train_fluxes_array, \
			   val_fluxes_array, \
			   train_amp_phase_array, \
			   val_amp_phase_array, \
			   scalers


	return fluxes_array, \
		   amp_phase_array, \
		   scalers



def load_validation_data(
	trim_amplitude=False,
	trim_phase=False,
	normalize_flux=False,
	normalize_amplitude=False,
	normalize_phase=False,
	shuffle=False,
	flatten_fluxes=False,
	flux_top_padding=0,
	flux_bottom_padding=0,
	amp_phase_top_padding=0,
	amp_phase_bottom_padding=0,
	):
	"""
	Function that retrieves numpy data given a path and processes it

	Input:
		normalize_flux (bool): Indicates wheter or not apply normalization to the flux data
		normalize_amplitude (bool): Indicates wheter or not apply normalization to the amplitude data
		normalize_phase (bool): Indicates whether or not apply normalization to the phase data
		shuffle (bool): Indicates whether or not to shuffle the data (all together: fluxes, amplitudes and phases)
		flatten_fluxes (bool): Indicates whether or not flatten the flux array
		
	Returns:
		If split is True:
			train_fluxes_array (np.array): The array containing the fluxes training set
			val_fluxes_array (np.array): The array containing the fluxes validation set
			train_amp_phase_array (np.array): The array containing the amplitude+phase training set
			val_amp_phase_array (np.array): The array containing the amplitude+phase validation set
			scalers (list): A list of scalers of each of the normalized arrays (can be empty)

		If split is False:
			fluxes_array (np.array): The array containing the fluxes
			amp_phase_array (np.array): The array containing the amplitudes and phase merged
			scalers (list): A list of scalers of each of the normalized arrays (can be empty)
	"""

	# LOAD DATA
	# If a number of points to sample has been specified, then get the first n_points

	flux_data_filepath = f"{FLUXES_FOLDER}/{FLUXES_FILE}"
	fluxes_array = np.load(flux_data_filepath)[70000:80000]

	amp_data_filepath = f"{SLM_FOLDER}07/{AMPLITUDE_FILE}"
	phase_data_filepath = f"{SLM_FOLDER}07/{PHASE_FILE}"

	amplitudes_array = np.load(amp_data_filepath)
	phases_array = np.load(phase_data_filepath)


	# TRIM DATA
	if trim_amplitude:
		amplitudes_array = trim_data(amplitudes_array)
	if trim_phase:
		phases_array = trim_data(phases_array)

	# PADDING DATA
	if flux_top_padding > 0 or flux_bottom_padding > 0:
		fluxes_array = add_row_padding(fluxes_array,
									   flux_top_padding,
									   flux_bottom_padding)

	if amp_phase_top_padding > 0 or amp_phase_bottom_padding > 0:
		amplitudes_array = add_row_padding(amplitudes_array,
									   	   amp_phase_top_padding,
									   	   amp_phase_bottom_padding)

		phases_array = add_row_padding(phases_array,
									   amp_phase_top_padding,
									   amp_phase_bottom_padding)

	# NORMALIZE_DATA
	scalers = []
	if normalize_flux:
		fluxes_array, fluxes_scaler = normalize_data(fluxes_array)
		scalers.append(fluxes_scaler)

	if normalize_amplitude:
		amplitudes_array, amplitudes_scaler = normalize_data(amplitudes_array)
		scalers.append(amplitudes_scaler)

	if normalize_phase:
		phases_array, phases_scaler = normalize_data(phases_array)
		scalers.append(phases_scaler)

	# SHUFFLE DATA
	if shuffle:
		fluxes_array, amplitudes_array, phases_array = shuffle_arrays([fluxes_array,
                                                               		   amplitudes_array,
                                                               		   phases_array])

	# FLATTEN FLUXES
	if flatten_fluxes:
		fluxes_array = flatten_data(fluxes_array)

	# MERGE PHASE AND AMPLITUDE
	amp_phase_array = fuse_amplitude_and_phase(amplitudes_array,
                      					       phases_array)

	return fluxes_array, \
		   amp_phase_array, \
		   scalers


def save_numpy_array(
	array,
	filepath,
	single_precision=True
	):

	"""
	Function that stores a numpy array to a file in float32 precision

	Input:
		array (np.array): The numpy array to store in a file
		filepath (string): The path to the file where the array will be saved

	Returns:
		None
	"""
	if os.path.isfile(filepath) or os.path.exists(filepath):
		print(f"ERROR: {filepath} already exists.")
		return None

	if single_precision:
		# Change the type of the array
		single_precision_array = np.float32(array)

		# Save the array in a numpy file
		np.save(filepath, single_precision_array)

	else:
		np.save(filepath, array)


	return None


def save_scaler(
	scaler,
	filepath
	):
	"""
	Saves a scaler

	Input:
		scaler (sklearn.preprocessing.StandardScaler): A scaler used to normalize a dataset
		filepath (string): The path to the scaler

	Returns:
		None
	"""

	# Save the scaler in a pickle file
	dump(scaler, open(filepath, 'wb'))


def train_generator(
	features_path,
	labels_path,
	batch_size,
	do_shuffle=False,
	n_samples=80000
	):
	"""
	This is the train data generator, loads batches dynamically to train with bigger sizes of data

	Input:
		features_path (string): The path to the feature files, in this case it will be the FLUX_PATH_PREFIX
		labels_path (string): The path to the label files, in this case it will be the AMP_PHASE_PATH_PREFIX
		batch_size (int): The size of the arrays
		do_shuffle (bool): If True, then shuffle the data
	"""
	
	while True:
		start_index = 0
		end_index = 0 + batch_size
		current_file = 0

		current_fluxes_array, current_amp_phase_array = load_subfile_for_train_generator(features_path,
																						 labels_path,
																						 current_file,
																						 do_shuffle=do_shuffle)
		# Go through the subfiles
		while end_index < n_samples:
			# Compute the indexes in the subfiles
			batch_start = start_index%10000
			batch_end = end_index%10000

			# If we need to load another file then do it 
			if batch_start > batch_end:
				# Load the last part of the current file
				first_partial_fluxes = current_fluxes_array[batch_start:]
				first_partial_amp_phase = current_amp_phase_array[batch_start:]

				current_file += 1
				current_fluxes_array, current_amp_phase_array = load_subfile_for_train_generator(features_path,
																						 		 labels_path,
																						 		 current_file,
																						 		 do_shuffle=do_shuffle)

				second_partial_fluxes = current_fluxes_array[:batch_end]
				second_partial_amp_phase = current_amp_phase_array[:batch_end]

				fluxes_batch = np.concatenate([first_partial_fluxes, second_partial_fluxes], axis=0)
				amp_phase_batch = np.concatenate([first_partial_amp_phase, second_partial_amp_phase], axis=0)

			else:
				fluxes_batch = current_fluxes_array[batch_start:batch_end]
				amp_phase_batch = current_amp_phase_array[batch_start:batch_end]


			start_index += batch_size
			end_index += batch_size

			yield fluxes_batch, amp_phase_batch


def load_subfile_for_train_generator(feature_path_prefix,
									 labels_path_prefix,
									 subfile_number,
									 do_shuffle=False):
	"""
	Loads the numpy arrays of a subfile

	Input:
		features_path (string): The path to the feature files, in this case it will be the FLUX_PATH_PREFIX
		labels_path (string): The path to the label files, in this case it will be the AMP_PHASE_PATH_PREFIX
		subfile_number (string): The identifier of the current subfile that we are reading
		do_shuffle (bool): If True, then shuffle the data

	Returns:
		current_fluxes_array (np.array): The array containing the fluxes
		current_amp_phase_array (np.array): The array containing the amplitude and phases
	"""

	# Create the file names
	current_features_filename = f"{feature_path_prefix}0{subfile_number}{NUMPY_SUFFIX}"
	current_labels_filename = f"{labels_path_prefix}0{subfile_number}{NUMPY_SUFFIX}"

	# Load the new arrays
	current_fluxes_array = np.load(current_features_filename)
	current_amp_phase_array = np.load(current_labels_filename)

	# Shuffle if needed
	if do_shuffle:
		shuf_current_fluxex_array, shuf_current_amp_phase_array = shuffle_arrays([current_fluxes_array, current_amp_phase_array])
		return shuf_current_fluxex_array, shuf_current_amp_phase_array

		
	return current_fluxes_array, current_amp_phase_array


### PSF RELATED
def generate_psf_complex_fields(
	filepath,
	telescope_diameter=1,
	wavelength=1e-6,
	pupil_grid_size=256,
	focal_q=8,
	num_airy=8,
	fried_parameter=0.2,
	outer_scale=20,
	velocity=10,
	n_samples=SUBFILE_SAMPLES,
	plot=False
	):
	"""
	This function generates wavefronts and propagates in through the atmosphere and an aperture to obtain aberrated PSFs that will be stored in the indicated file.
	
	Input:
		filepath (string): The path of file to store the psf
		teslecope_diameter (float): The diameter of the aperture
		wavelength (float): The wavelength of the light
		pupil_grid_size (int): The pixels per row (or columns as it is a square) of the grid
		focal_q (int)
		num_airy=16
	"""

	D_tel = 0.5
	wavelength = 1e-6

	pupil_grid = make_pupil_grid(256, D_tel)
	focal_grid = make_focal_grid(q=8, num_airy=8, spatial_resolution=wavelength/D_tel)
	propagator = FraunhoferPropagator(pupil_grid, focal_grid)

	aperture = make_circular_aperture(D_tel)(pupil_grid)

	fried_parameter = 0.2 # meter
	outer_scale = 20 # meter
	velocity = 10 # meter/sec

	Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
	atmosphere = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

	wf = Wavefront(aperture, wavelength)

	propagated_wavefronts = propagate_wavefronts(n_samples,
												 wf,
												 propagator,
												 atmosphere,
												 plot=plot)

	save_wavefronts_complex_fields(propagated_wavefronts,
								   filepath)
	return None


def propagate_wavefronts(
	n_samples,
	wavefront,
	propagator,
	atmosphere,
	plot=False
	):
	"""
	This function propagates a wavefront through an atmosphere layer several times and results the wavefront propagated until the focal plane of a propagator
	"""
	wavefronts = []

	for t in range(n_samples):
		atmosphere.reset()
		propagated_wavefront = propagator(atmosphere(wavefront))
		wavefronts.append(propagated_wavefront)

		if plot:
			original_psf = propagator(wavefront)
			plt.clf()
			plt.subplot(1,4,1)
			imshow_field(propagated_wavefront.phase, vmin=-6)
			plt.colorbar()
			plt.subplot(1,4,2)
			imshow_field(np.log10(propagated_wavefront.amplitude/propagated_wavefront.amplitude.max()), vmin=-6)
			plt.colorbar()
			plt.subplot(1,4,3)
			imshow_field(np.log10(propagated_wavefront.intensity/ propagated_wavefront.intensity.max()), vmin=-6)

			plt.subplot(1,4,4)
			imshow_field(np.log10(original_psf.intensity/ original_psf.intensity.max()), vmin=-6)
			plt.colorbar()
			plt.draw()

	return wavefronts


def save_wavefronts_complex_fields(
	propagated_wavefronts,
	filepath
	):
	
	n_fields = len(propagated_wavefronts)
	# Compute the rows (or columns as the wf is a square grid)
	square_side = int(np.sqrt(propagated_wavefronts[0].amplitude.shape))
	complex_fields = np.zeros((n_fields, square_side, square_side), dtype='complex')

	for i in range(n_fields):
		real_part = np.array([propagated_wavefronts[i].real])
		imaginary = np.array([propagated_wavefronts[i].imag])
		comp_amp_phase = real_part + imaginary*1j
		comp_amp_phase = comp_amp_phase.reshape((square_side, square_side))
		complex_fields[i] = comp_amp_phase

	save_numpy_array(complex_fields, filepath, single_precision=False)


def compute_output_fluxes_from_complex_field(
	complex_fields_file_path,
	output_fluxes_file_path,
	plot=False,
	verbose=False
	):
	
	# Create the lantern fiber
	n_core = 1.44
	n_cladding = 1.4345
	wavelength = 1.5 # microns
	core_radius = 32.8/2 # microns

	# Scale parameters
	max_r = 2 # Maximum radius to calculate mode field, where r=1 is the core diameter
	npix = 200 # Half-width of mode field calculation in pixels
	show_plots = False

	# Input fields
	inp_pix_scale = 4 # input pixels / fiber-field pixels

	lantern_fiber = LanternFiber(n_core, 
					 			 n_cladding,
					 			 core_radius,
					 			 wavelength)
	lantern_fiber.find_fiber_modes()
	lantern_fiber.make_fiber_modes(npix=npix, show_plots=show_plots, max_r=max_r)
	modes_to_measure = np.arange(lantern_fiber.nmodes)

	input_complex_fields = np.load(complex_fields_file_path)
	n_fields = input_complex_fields.shape[0]
	transfer_matrix = load_transfer_matrix()

	output_fluxes = np.zeros((input_complex_fields.shape[0], len(modes_to_measure)))

	for k in range(n_fields):
		original_field = input_complex_fields[k,:,:]
		resized_field_real = rescale(original_field.real, inp_pix_scale)
		resized_field_imag = rescale(original_field.imag, inp_pix_scale)
		resized_field = resized_field_real + resized_field_imag*1j

		input_field = resized_field
		cnt = input_field.shape[1]//2
		input_field = input_field[cnt-lantern_fiber.npix:cnt+lantern_fiber.npix, cnt-lantern_fiber.npix:cnt+lantern_fiber.npix]

		lantern_fiber.input_field = input_field
		lantern_fiber.plot_injection_field(lantern_fiber.input_field, show_colorbar=False, logI=True, vmin=-3, fignum=50)

		coupling, mode_coupling, mode_coupling_complex = lantern_fiber.calc_injection_multi(
			mode_field_numbers=modes_to_measure,
			verbose=verbose, 
			show_plots=plot, 
			fignum=11,
			complex=True,
			ylim=0.3,
			return_abspower=True)

		# Now get the complex amplitudes of the PL outputs:
		pl_outputs = transfer_matrix @ mode_coupling_complex

		# In real life, we just measure the intensities of the outputs:
		pl_output_fluxes = np.abs(pl_outputs)**2
		output_fluxes[k] = pl_output_fluxes

		if plot:
			# Plot input mode coefficients and output fluxes
			xlabels = np.arange(lantern_fiber.nmodes)
			plt.figure(1)
			plt.clf()
			plt.subplot(311)
			plt.bar(xlabels, np.abs(mode_coupling_complex))
			plt.title('Input mode amplitudes')
			plt.subplot(312)
			plt.bar(xlabels, np.angle(mode_coupling_complex))
			plt.title('Input mode phases')
			plt.subplot(313)
			plt.bar(xlabels, pl_output_fluxes)
			plt.title('Output fluxes')
			plt.tight_layout()

	save_numpy_array(output_fluxes, output_fluxes_file_path)


def load_transfer_matrix(
	lanter_fiber_directory=PSF_DATA_PATH,
	lantern_fiber_filename=LANTERN_FIBER_FILENAME):

	lantern_fiber = LanternFiber(datadir=PSF_DATA_PATH, nmodes=19, nwgs=19)
	lantern_fiber.load_savedvalues(LANTERN_FIBER_FILENAME)
	lantern_fiber.make_transfer_matrix_mm2sm(show_plots=True)

	transfer_matrix = lantern_fiber.Cmat # This is the complex transfer matrix
	return transfer_matrix

