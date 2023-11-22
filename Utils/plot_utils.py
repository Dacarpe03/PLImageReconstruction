import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd


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
	fig = px.imshow(whatever_map)
	# Show the figure
	fig.show()
	return None


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
    plt.ylim(top=0.5, bottom=0)
    # Show the plot
    plt.show()

    return None