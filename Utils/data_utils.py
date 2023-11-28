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

