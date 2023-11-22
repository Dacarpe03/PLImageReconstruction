import numpy as np


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

	# Obtain the number of data points and their dimensions
	n_points, x_length, y_length = data.shape
	# Compute the new length of the flattened data point
	flat_length = x_length * y_length
	# Flatten all the points from the original array
	flattened_data = data.reshape(n_points, flat_length)

	return flattened_data