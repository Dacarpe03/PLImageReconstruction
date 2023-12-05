import numpy as np
from sklearn.preprocessing import StandardScaler


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
	data
	):
	"""
	This function scales the data so that it has mean=0 an standard deviation=1

	Input:
		data (np.array): The array to normalize

	Returns:
		normalized_data (np.array): The normalized data
		scaler (sklearn.preprocessing.StandardScaler): The scaler in case we need to unnormalize the data
	"""

	array_dimensions = len(data.shape)
	# Reshape the data into a 1d array
	flattened_data = data.reshape(-1, array_dimensions)

    # Create a StandardScaler object with mean=0 and std=1
	scaler = StandardScaler(with_mean=True, 
							with_std=True)

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

	return shuffled_array_list


def fuse_amplitude_and_phase(
	amplitudes_array,
	phases_array
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
								axis=1)

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
	flux_data_filepath,
	amplitude_data_filepath,
	phase_data_filepath,
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
	if n_points is not None:
		fluxes_array = np.load(flux_data_filepath)[0:n_points]
		amplitudes_array = np.load(amplitude_data_filepath)[0:n_points]
		phases_array = np.load(phase_data_filepath)[0:n_points]

	# Else load the whole file
	else:
		fluxes_array = np.load(flux_data_filepath)
		amplitudes_array = np.load(amplitude_data_filepath)
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



