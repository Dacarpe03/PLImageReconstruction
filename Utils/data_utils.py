import numpy as np
import pandas as pd


def load_numpy_data(data_filepath,
			  		n_points=None):
	"""
	Function that retrieves numpy data given a path

	Input:
		data_filepath (string): The data file path
		n_points (int): The number of points to load

	Returns:
		data (np.array): An array containing the loaded data
	"""

	if n_points is not None:
		data = np.load(data_filepath)[0:n_points]
	else:
		data = np.load(data_filepath)

	return data
