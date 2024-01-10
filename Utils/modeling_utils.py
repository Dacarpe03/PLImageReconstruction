import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential

from keras.layers import InputLayer, \
						 Conv2D, \
						 Conv1D, \
						 MaxPooling2D, \
						 Flatten, \
						 Dense, \
						 BatchNormalization, \
						 Activation, \
						 Reshape, \
						 Dropout, \
						 UpSampling2D

from data_utils import train_generator

from constants import MODELS_FOLDER_PATH, \
					  KERAS_SUFFIX, \
					  MODELS_DESCRIPTION_FILE_PATH


def create_fully_connected_architecture_for_amplitude_and_phase_reconstruction(
	input_shape,
	output_shape,
	hidden_layer_sizes,
	regularizer,
	hidden_activation,
	output_activation,
	use_batch_normalization=True,
	name="FCAmplitudePhaseReconstructor",
	use_dropout=False,
	dropout_rate=0.1
	):
	"""
	Instantiates the architecture of a fully connected neural network for amplitude and phase reconstruction

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
		
	model.summary()
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
	Instantiates the architecture of a convolutional neural network for amplitude and phase reconstruction

	Input:
		input_shape (tuple): The shape a data point in the features dataset
		output_shape (tuple): The shape of a data point in the labels dataset
		convolutional_layer_sizes (list): A list of integers containing the number of filters per convolutional layer
		convolutinal_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
		fully_connected_hidden_layer_sizes (list): A list of integers
		regularizer (keras.regularizers): A regularizer for the hidden layers (e.g. L1, see keras documentation for more)
		convolutional_activation (string): The name of the activation function of the convolutional hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		fully_connected_hidden_activation (string): The name of the activation function of the hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		output_activation (string): The name of the activation function of the output layers (e.g 'linear', see keras documentation for more)
		use_batch_normalization (bool): If True, then add batch normalization to the hidder layers
		name (string): The name of the model

	Returns:
		model (keras.Sequential): A keras neural network model with the architecture specified
	"""

	model = Sequential(
				name=name
			)
	
	input_shape = input_shape
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

		if i%2 == 1:
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


def create_autoencoder_for_flux(
	input_shape,
	convolutional_layer_sizes,
	convolutional_layer_kernels,
	convolutional_activation,
	output_activation,
	name="AutoEncoder",
	padding="same",
	use_batch_normalization=True
	):
	"""
	Instantiates the architecture of a convolutional neural network for amplitude and phase reconstruction

	Input:
		input_shape (tuple): The shape a data point in the features dataset
		convolutional_layer_sizes (list): A list of integers containing the number of filters per convolutional layer
		convolutinal_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
		convolutional_activation (string): The name of the activation function of the convolutional hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		output_activation (string): The name of the activation function of the output layers (e.g 'linear', see keras documentation for more)
		name (string): The name of the model
		padding (string): The padding used in convolutional layers
		use_batch_normalization (bool): If True, then add batch normalization to the hidder layers

	Returns:
		model (keras.Sequential): A keras neural network model with the architecture specified
	"""

	# Add a single dimension to the array for the max pooling to be possible
	input_shape = input_shape
	model = Sequential(
				name=name
			)


	# INPUT
	model.add(
		Conv2D(
			convolutional_layer_sizes[0],
			convolutional_layer_kernels[0],
			activation=convolutional_activation,
			input_shape=input_shape,
			padding=padding
		)
	)

	model.add(
			Conv2D(
				convolutional_layer_sizes[0],
				convolutional_layer_kernels[0],
				activation=convolutional_activation,
				padding=padding
			)
	)

	model.add(
			MaxPooling2D(
				pool_size=(2,2)
			)
	)

	# ENCODER
	for i in range(1, len(convolutional_layer_sizes)-1):
		for j in range(2):
			model.add(
					Conv2D(
						convolutional_layer_sizes[i],
						convolutional_layer_kernels[i],
						activation=convolutional_activation,
						padding=padding

					)
			)

		if use_batch_normalization:
			model.add(
				BatchNormalization()
			)

		model.add(
				MaxPooling2D(
					pool_size=(2,2)
				)
		)

	for j in range(2):
		if j==1:
			layer_name = "bottleneck"
		else:
			layer_name = "pre_bottleneck"
		model.add(
				Conv2D(
					convolutional_layer_sizes[-1],
					convolutional_layer_kernels[-1],
					activation=convolutional_activation,
					padding=padding,
					name=layer_name
				)
		)
	
	# DECODER
	convolutional_layer_sizes.reverse()
	convolutional_layer_kernels.reverse()

	for i in range(1, len(convolutional_layer_sizes)):
		model.add(
				UpSampling2D(
					size=(2,2)
				)
		)
		for j in range(2):

			model.add(
					Conv2D(
						convolutional_layer_sizes[i],
						convolutional_layer_kernels[i],
						activation=convolutional_activation,
						padding=padding
					)
			)

		if use_batch_normalization:
			model.add(
				BatchNormalization()
			)

	# OUTPUT
	model.add(
			Conv2D(
				1,
				(3,3), 
				activation=output_activation,
				padding=padding
			)
		)

	model.summary()
	return model


def create_convolutional_architecture_with_encoder_for_amplitude_phase_reconstruction(
	autoencoder,
	convolutional_layer_sizes,
	convolutional_layer_kernels,
	convolutional_activation,
	output_activation,
	model_name,
	padding='same',
	use_batch_normalization=True
	):
	"""
	This function creates a convolutional nn with a freezed encoder input (representing the flux) to reconstruct the amplitude and phase map
	
	Input:
		autoencoder (keras.models): An autoencoder model to decouple and join to a new model
		convolutional_layer_sizes (list): A list of integers containing the number of filter per convolutional layer
		convolutional_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
		convolutional_activation (string): The name of the activation function of the convolutional hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		output_activation (string): The name of the activation function of the output layers (e.g 'linear', see keras documentation for more)
		model_name (string): The name of the model
		padding (string): The padding used in convolutional layer
		use_batch_normalization (bool): If True, then add batch normalization to the hidder layers
	"""

	# Extract the encoder from the autoencoder
	encoder = keras.models.Model(autoencoder.input,
								 autoencoder.get_layer('bottleneck').output, 
								 name=model_name)

	# Freeze the encoder neurons
	for layer in encoder.layers:
		layer.trainable = False
		if (layer.name == 'bottleneck'):
			break

	conv_input = encoder.output
	conv_layers = UpSampling2D(size=(1,2))(conv_input)

	for i in range(len(convolutional_layer_sizes)):
		for j in range(2):
			conv_layers = Conv2D(
								convolutional_layer_sizes[i],
								convolutional_layer_kernels[i],
								activation=convolutional_activation,
								padding=padding
							)(conv_layers)

		if use_batch_normalization:
			conv_layers = BatchNormalization()(conv_layers)

		conv_layers = UpSampling2D(
						size=(2,2)
						)(conv_layers)

	# OUTPUT_LAYER
	conv_layers = Conv2D(
					2,
					(3,3), 
					activation=output_activation,
					padding=padding
					)(conv_layers)

	conv_model = keras.Model(inputs=encoder.input, outputs=conv_layers)
	conv_model.summary()
	return conv_model


def create_convolutional_architecture_for_psf_reconstruction(
	fc_layer_sizes,
	fc_activation,
	convolutional_layer_sizes,
	convolutional_layer_kernels,
	convolutional_activation,
	output_activation,
	regularizer,
	model_name,
	padding='same',
	use_batch_normalization=True,
	use_dropout=False
	):
	"""
	This function creates a convolutional nn with a freezed encoder input (representing the flux) to reconstruct the amplitude and phase map
	
	Input:
		fc_layer_sizes (list): A list of integers containing the number of neurons per layer in the fully connected part
		convolutional_layer_sizes (list): A list of integers containing the number of filter per convolutional layer
		convolutional_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
		convolutional_activation (string): The name of the activation function of the convolutional hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		output_activation (string): The name of the activation function of the output layers (e.g 'linear', see keras documentation for more)
		model_name (string): The name of the model
		padding (string): The padding used in convolutional layer
		use_batch_normalization (bool): If True, then add batch normalization to the hidder layers
	"""

	# Add a single dimension to the array for the max pooling to be possible
	input_shape = (19, 1)
	model = Sequential(
				name=model_name
			)

	# Create input layer
	model.add(
		InputLayer(
			input_shape=input_shape,
			batch_size=None)
			)

	# Create the hidden layers of the neural network
	for neurons in fc_layer_sizes:

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
				fc_activation
				)
			)

		if use_dropout:
			model.add(
				Dropout(
					0.1
				)
			)

	for i in range(len(convolutional_layer_sizes)):

		model.add(
				Conv1D(
					convolutional_layer_sizes[i],
					convolutional_layer_kernels[i],
					activation=convolutional_activation,
					padding=padding
				)					
			)

		if use_batch_normalization:
			model.add(
				BatchNormalization()
			)


	# OUTPUT
	model.add(
			Conv1D(
				2,
				3, 
				activation=output_activation,
				padding=padding
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


def train_model_with_generator(model,
						 	   fluxes_path,
						 	   amplitudes_path,
						 	   validation_fluxes,
						 	   validation_amplitudes,
						 	   epochs,
						 	   batch_size,
						 	   callbacks,
						 	   n_samples=70000
						 	   ):
	"""
	Fits the model to the train instances of the data.

	Input:
		model (keras.Sequential): The sequential model to train
		train_generator (function): A train generator
		validation_fluxes (np.array): An np.array containing np.array with the train features
		validation_amplitudes (np.array): An np.array containing np.array with the train features 
		epochs (int): The number of times the training goes through the training data
		batch_size(int): The batch size of training samples used before each weight update
		callbacks (list): A list of keras callbacks used during the training.

	Returns:
		history (): The training history of the model
	"""

	train_gen = train_generator(fluxes_path,
								amplitudes_path,
								batch_size,
								do_shuffle=False,
								n_samples=n_samples)

	steps_per_epoch = math.ceil(n_samples/batch_size)
	validation_steps = math.ceil(len(validation_amplitudes)/batch_size)

	history = model.fit_generator(train_gen, 
								  steps_per_epoch=steps_per_epoch,
								  validation_data=(validation_fluxes, validation_amplitudes),
								  epochs=epochs,
								  callbacks=callbacks,
								  verbose=1)


	return history


def store_model(
	model,
	model_name,			
	description):
	"""
	Stores the model in the DATA_FOLDER with the name with a description in the neural network descriptions file

	Input:
		model (keras.models): The model to save in the models folder
		model_name (string): The name of the model
		description (string): The description of the model 

	Returns:
		None
	"""

	# To not overwrite other model, look for the number of models with the same configuration to add a version number at the end of the file
	model_files = os.listdir(MODELS_FOLDER_PATH)
	version = sum(1 for file in model_files if file.startswith(model_name))+1

	# Save its description
	with open(MODELS_DESCRIPTION_FILE_PATH, 'a') as f:
		f.write(f"===={model_name}_{version}====\n")
		f.write(description)
		f.write("\n\n")


	# Create the model path
	model_file_path = f"{MODELS_FOLDER_PATH}/{model_name}_{version}{KERAS_SUFFIX}"

	# Save the model
	model.save(model_file_path)

	return None


def load_model(
	model_name):
	"""
	Loads a model given its name

	Input:
		model_name (string): The name of the model to load

	Returns:
		model (keras.models): The loaded model
	"""
	model_path = f"{MODELS_FOLDER_PATH}/{model_name}{KERAS_SUFFIX}"
	model = keras.models.load_model(model_path)
	return model


def evaluate_model(
	model,
	features,
	labels
	):
	"""
	Evaluates a model and prints its mean squared error
	
	Input:
		model (keras.models): The model to evaluate
		features (np.array): The array with the features to predict
		labels (np.array): The labels of the features to evaluate

	Returns:
		None
	"""
	results = model.evaluate(features, labels)
	print("MSE:", results[1])
	return results
