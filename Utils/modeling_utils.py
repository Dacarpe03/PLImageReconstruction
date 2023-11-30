import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Reshape, Dropout



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
	name="AmplitudeReconstructor",
	use_dropout=False,
	dropout_rate=0.1
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
				kernel_regularizer=regularizer,
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

		if use_dropout:
			model.add(
				Dropout(
					0.1
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


def create_convolutional_architecture_for_amplitude_and_phase_reconstruction(
	input_shape,
	output_shape,
	convolutional_layer_sizes,
	convolutinal_layer_kernels,
	fully_connected_hidden_layer_sizes,
	regularizer,
	convolutional_activation,
	fully_connected_hidden_activation,
	output_activation,
	use_batch_normalization=True,
	name="ConvolutionalAmplitudePhaseReconstructor"
	):

	"""
	Defines de architecture of the convolutional neural network

	Input:
		input_shape (tuple): The shape a data point in the features dataset
		output_shape (tuple): The shape of a data point in the labels dataset
		convolutional_layer_sizes (list): A list of integers containing the number of filter per convolutional layer
		convolutinal_layer_kernels (list): A list of integers containing the size of the kernel per convolutional layer
		fully_connected_hidden_layer_sizes (list): A list of integers
		regularizer (keras.regularizers): A regularizer for the hidden layers (e.g. L1, see keras documentation for more)
		fully_connected_hidden_activation (string): The name of the activation function of the hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		convolutional_activation (string): The name of the activation function of the convolutional hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		output_activation (string): The name of the activation function of the output layers (e.g 'linear', see keras documentation for more)
		use_batch_normalization (bool): If True, then add batch normalization to the hidder layers
		name (string): The name of the model

	Returns:
		model (keras.Sequential): A keras neural network model with the architecture specified
	"""

	model = Sequential(
				name=name
			)
	
	input_shape = input_shape + (1, )
	output_size = np.prod(output_shape)


	model.add(
		Conv2D(
			convolutional_layer_sizes[0],
			convolutinal_layer_kernels[0],
			activation=convolutional_activation,
			input_shape=input_shape
		)
	)

	model.add(
		MaxPooling2D(
			pool_size=(2,2)
		)
	)

	for i in range(1, len(convolutional_layer_sizes)):
		model.add(
				Conv2D(
					convolutional_layer_sizes[i],
					convolutinal_layer_kernels[i],
					activation=convolutional_activation
				)
		)

		model.add(
				MaxPooling2D(
					pool_size=(2,2)
				)
		)

	model.add(
			Flatten()
	)
	
	for neurons in fully_connected_hidden_layer_sizes:

		# Define layer
		model.add(
			Dense(
				neurons,
				# kernel_regularizer=regularizer,
				# kernel_initializer=keras.initializers.HeNormal(seed=None),
				kernel_regularizer=regularizer,
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
				fully_connected_hidden_activation
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

	model.summary()
	return model


def compile_model(
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


def train_model(
	model,
	train_features,
	train_labels,
	val_features,
	val_labels,
	epochs,
	batch_size,
	callbacks):

	"""
	Fits the model to the train instances of the data.

	Input:
		model (keras.Sequential): The sequential model to train
		train_features (np.array): An np.array containing np.array with the train features
		train_labels (np.array): An np.array containing np.array with the train features
		val_features (np.array): An np.array containing np.array with the train features
		val_labels (np.array): An np.array containing np.array with the train features 
		epochs (int): The number of times the training goes through the training data
		batch_size(int): The batch size of training samples used before each weight update
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




