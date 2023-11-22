import plotly.express as px


def plot_map(whatever_map):
	"""
	Plots an amplitude, phase of flux map

	Input:
		whatever_map (np.array): A 2D array containing the map

	Returns:
		None
	"""
	
	fig = px.imshow(whatever_map)
	fig.show()
	return None