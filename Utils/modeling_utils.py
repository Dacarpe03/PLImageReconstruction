import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation


def create_linear_architecture_for_amplitude_reconstruction(
	input_shape,
	output_size,
	hidden_layer_sizes,
	regularizer,
	hidden_activation,
	output_activation,
	use_batch_normalization=True,
	name="SurfaceReconstructor"
	):
	"""
	Defines de architecture of the neural network

	Input:
		input_shape (tuple): The shape a data point in the features dataset
		output_shape (tuple): The shape of a data point in the labels dataset
		hidden_layer_sizes (list): A list of integers
		regularizer (keras.regularizers): A regularizer for the hidden layers (e.g. L1, see keras documentation for more)
		hidden_activation (string): The name of the activation function of the hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		output_activation (string): The name of the activation function of the output layers (e.g 'linear', see keras documentation for more)
		use_batch_normalization (bool): If True, then add batch normalization to the hidder layers
		name (string): The name of the model

	Returns:
		model (keras.Sequential): A keras neural network model with the architecture specified
	"""
	# As the output is an image, we will show the fucking
	output_size = output_shape[0] * output_shape[1]
	# Create a sequential model
	model = Sequential(
		name=name
		)

	# Create input layer
	model.add(
		InputLayer(
			input_shape=input_shape,
			batch_size=None)
			)

	# Create the hidden layers of the neural network
	for neurons in hidden_layer_sizes:

		# Define layer
		model.add(
			Dense(
				neurons,
				kernel_regularizer=regularizer,
				kernel_initializer=keras.initializers.HeNormal(seed=None),
				use_bias=False
				)
			)

		# Add normalization
		if use_batch_normalization:
			model.add(
				BatchNormalization()
				)

		# Define the activation function
		model.add(
			Activation(
				hidden_activation
				)
			)

	# Add output layer
	model.add(
		Dense(
			output_size,
			activation=output_activation
			)
		)

	# Reshape the linear neurons into the reconstructed image
	model.add(
		Reshape(
			output_shape
			)
		)

	return model