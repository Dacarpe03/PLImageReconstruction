import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Reshape


def create_linear_architecture_for_amplitude_reconstruction(
	input_shape,
	output_shape,
	hidden_layer_sizes,
	regularizer,
	hidden_activation,
	output_activation,
	use_batch_normalization=True,
	name="AmplitudeReconstructor"
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
	
	# As the output is an image, we need to create a final layer with as many neurons as pixels
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


def create_fully_connected_architecture_for_amplitude_and_phase_reconstruction(
	input_shape,
	output_shape,
	hidden_layer_sizes,
	regularizer,
	hidden_activation,
	output_activation,
	use_batch_normalization=True,
	name="AmplitudeReconstructor"
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
	
	# As the output is an image, we need to create a final layer with as many neurons as pixels
	output_size = np.prod(output_shape)

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
				# kernel_regularizer=regularizer,
				# kernel_initializer=keras.initializers.HeNormal(seed=None),
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


def compile_linear_model_for_amplitude_reconstruction(
	model,
	loss_function,
	optimizer,
	metric
	):
	"""
	Tells the model how to train

	Input:
		model(keras.Sequential): The sequential model to compile
		loss_function (keras.losses): The loss function used to update the weights of the neurons (eg. MeanSquaredError())
		optimizer (keras.optimizers): The optimizer used to update the weights of the neurons (eg. Adam)
		metric (keras.metrics): The metrics to monitor during the training (eg. MeanSquaredError())

	Returns:
		None
	"""
	model.compile(
		loss=loss_function,
		optimizer=optimizer,
		metrics=[metric]
		)

	return None


def train_linear_model_for_amplitude_reconstruction(
	model,
	train_features,
	train_labels,
	val_features,
	val_labels,
	batch_size,
	epochs,
	callbacks):

	"""
	Fits the model to the train instances of the data.

	Input:
		model (keras.Sequential): The sequential model to train
		train_features (np.array): An np.array containing np.array with the train features
		train_labels (np.array): An np.array containing np.array with the train features
		batch_size(int): The batch size of training samples used before each weight update
		epochs (int): The number of times the training goes through the training data
		val_features (np.array): An np.array containing np.array with the train features
		val_labels (np.array): An np.array containing np.array with the train features 
		callbacks (list): A list of keras callbacks used during the training.

	Returns:
		history (): The training history of the model
	"""
	history = model.fit(train_features,
						train_labels,
						batch_size=batch_size,
						epochs=epochs,
						validation_data=(val_features, val_labels),
						callbacks=callbacks,
						verbose=1)

	return history