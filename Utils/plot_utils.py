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

	fig = px.imshow(whatever_map)
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
    results = pd.DataFrame(history.history)
    results.plot(figsize=(8,5))
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.ylim(top=0.5, bottom=0)
    
    plt.show()

    return None