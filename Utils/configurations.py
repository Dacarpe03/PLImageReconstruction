from abc import ABC, abstractmethod
from typing import Type

from keras.losses import MeanSquaredError as LossesMeanSquaredError
from keras.optimizers import Adam
from keras.metrics import MeanSquaredError as MetricsMeanSquaredError
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import L1, L2


from constants import FC_INPUT_SHAPE, \
					  FC_OUTPUT_SHAPE, \
					  CNN_INPUT_SHAPE, \
					  CNN_OUTPUT_SHAPE, \
					  AUTOENCODER_INPUT_SHAPE

class ConfigurationElement(ABC):
	@abstractmethod
	def unpack_hyperparameters(self):
		"""
		This method returns all the hyperparameters of the configuration element
		"""
		pass


class FullyConnectedArchitecture(ConfigurationElement):
	"""This class contains the information of a fully connected model architecture"""
	def __init__(
		self, 
		input_shape,
		output_shape,
		hidden_layer_sizes,
		regularizer,
		hidden_activation,
		output_activation,
		use_batch_normalization,
		model_name,
		use_dropout=False,
		dropout_rate=0.1
		):
		super(FullyConnectedArchitecture, self).__init__()
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.hidden_layer_sizes = hidden_layer_sizes
		self.regularizer = regularizer
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		self.use_batch_normalization = use_batch_normalization
		self.model_name = model_name
		self.use_dropout = use_dropout
		self.dropout_rate = dropout_rate


	def unpack_hyperparameters(self):
		"""
		This method returns all the hyperparameters of the architecture configuration

		Input:
			None
		
		Returns:
			input_shape (tuple): The shape of the neural network input
			output_shape (tuple): The shape of the neural network output
			hidden_layer_shizes (list): The list with sizes of the hidden fully connected linear layers
			regularizer (keras.regularizers): The regularizer for the hidden layers
			hidden_activation (string): The name of the activation function in the hidden layers
			output_activation (string): The name of the activation in the output layer
			use_batch_normalization (bool): True if use batch normalization between hidden layers
			model_name (string): The name of the model
			use_dropout (bool): True if using dropout
			dropout_rate (float): The dropout rate of the hidden layers during training
		"""

		return self.input_shape, \
			   self.output_shape, \
			   self.hidden_layer_sizes, \
			   self.regularizer, \
			   self.hidden_activation, \
			   self.output_activation, \
			   self.use_batch_normalization, \
			   self.model_name, \
			   self.use_dropout, \
			   self.dropout_rate


class ConvolutionalArchitecture(ConfigurationElement):
	"""This class contains the information of a convolutional model architecture"""
	def __init__(
		self, 
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
		model_name="ConvolutionalAmplitudePhaseReconstructor"
		):
		super(ConvolutionalArchitecture, self).__init__()
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.convolutional_layer_sizes = convolutional_layer_sizes
		self.convolutinal_layer_kernels = convolutinal_layer_kernels
		self.fully_connected_hidden_layer_sizes = fully_connected_hidden_layer_sizes
		self.regularizer = regularizer
		self.convolutional_activation = convolutional_activation
		self.fully_connected_hidden_activation = fully_connected_hidden_activation
		self.output_activation = output_activation
		self.use_batch_normalization = use_batch_normalization
		self.model_name = model_name


	def unpack_hyperparameters(self):
		"""
		This method returns all the hyperparameters of the architecture configuration

		Input:
			None
		
		Returns:
			input_shape (tuple): The shape of the neural network input
			output_shape (tuple): The shape of the neural network output
			convolutional_layer_sizes (list): A list of integers containing the number of filters per convolutional layer
			convolutinal_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
			fully_connected_hidden_layer_sizes (list): The list with sizes of the hidden fully connected linear layers
			regularizer (keras.regularizers): The regularizer for the hidden layers
			convolutional_activation (string): The name of the activation function in the convolutional layers
			fully_connected_hidden_activation (string): The name of the activation function in the fully connected layers
			output_activation (string): The name of the activation in the output layer
			use_batch_normalization (bool): True if use batch normalization between hidden layers
			model_name (string): The name of the model
		"""

		return self.input_shape, \
			   self.output_shape, \
			   self.convolutional_layer_sizes, \
			   self.convolutinal_layer_kernels, \
			   self.fully_connected_hidden_layer_sizes, \
			   self.regularizer, \
			   self.convolutional_activation, \
			   self.fully_connected_hidden_activation, \
			   self.output_activation, \
			   self.use_batch_normalization, \
			   self.model_name


class AutoEncoderArchitecture(ConfigurationElement):
	"""This class contains the information of the architecture of an autoencoder model"""
	def __init__(
		self,
		input_shape,
		convolutional_layer_sizes,
		convolutinal_layer_kernels,
		convolutional_activation,
		output_activation,
		model_name="FluxAutoencoder",
		padding="same",
		use_batch_normalization=True
		):

		super(AutoEncoderArchitecture, self).__init__()
		self.input_shape = input_shape
		self.convolutional_layer_sizes = convolutional_layer_sizes
		self.convolutinal_layer_kernels = convolutinal_layer_kernels
		self.convolutional_activation = convolutional_activation
		self.output_activation = output_activation
		self.model_name = model_name
		self.padding = padding
		self.use_batch_normalization = use_batch_normalization


	def unpack_hyperparameters(self):
		"""
		This method returns all the hyperparameters of the architecture configuration

		Input:
			None
		
		Returns:
			input_shape (tuple): The shape of the neural network input
			convolutional_layer_sizes (list): A list of integers containing the number of filters per convolutional layer
			convolutinal_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
			convolutional_activation (string): The name of the activation function in the convolutional layers		
			output_activation (string): The name of the activation in the output layer
			use_batch_normalization (bool): True if use batch normalization between hidden layers
			model_name (string): The name of the model
		"""

		return self.input_shape, \
			   self.convolutional_layer_sizes, \
			   self.convolutinal_layer_kernels, \
			   self.convolutional_activation, \
			   self.output_activation, \
			   self.model_name, \
			   self.padding, \
			   self.use_batch_normalization


class EncoderConvolutionalArchitecture(ConfigurationElement):
	"""This class contains the information of a encoder + convolution model architecture"""
	def __init__(
		self,
		convolutional_layer_sizes,
		convolutinal_layer_kernels,
		convolutional_activation,
		output_activation,
		model_name="EncoderConvolutionalArchitecture70000",
		padding="same",
		use_batch_normalization=True
		):

		super(EncoderConvolutionalArchitecture, self).__init__()
		self.convolutional_layer_sizes = convolutional_layer_sizes
		self.convolutinal_layer_kernels = convolutinal_layer_kernels
		self.convolutional_activation = convolutional_activation
		self.output_activation = output_activation
		self.model_name = model_name
		self.padding = padding
		self.use_batch_normalization = use_batch_normalization


	def unpack_hyperparameters(self):
		"""
		This method returns all the hyperparameters of the architecture configuration

		Input:
			None
		
		Returns:
			convolutional_layer_sizes (list): A list of integers containing the number of filters per convolutional layer
			convolutinal_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
			convolutional_activation (string): The name of the activation function in the convolutional layers		
			output_activation (string): The name of the activation in the output layer
			use_batch_normalization (bool): True if use batch normalization between hidden layers
			model_name (string): The name of the model
		"""

		return self.convolutional_layer_sizes, \
			   self.convolutinal_layer_kernels, \
			   self.convolutional_activation, \
			   self.output_activation, \
			   self.model_name, \
			   self.padding, \
			   self.use_batch_normalization


class PSFConvolutionalArchitecture(ConfigurationElement):
	"""This class contains the information of a encoder + convolution model architecture"""
	def __init__(
		self,
		fc_layer_sizes,
		fc_activation,
		convolutional_layer_sizes,
		convolutinal_layer_kernels,
		convolutional_activation,
		output_activation,
		regularizer,
		model_name="PSFConvolutionalArchitecture",
		padding="same",
		use_batch_normalization=True,
		use_dropout=False
		):

		super(PSFConvolutionalArchitecture, self).__init__()
		self.fc_layer_sizes = fc_layer_sizes
		self.fc_activation = fc_activation
		self.convolutional_layer_sizes = convolutional_layer_sizes
		self.convolutinal_layer_kernels = convolutinal_layer_kernels
		self.convolutional_activation = convolutional_activation
		self.output_activation = output_activation
		self.regularizer = regularizer
		self.model_name = model_name
		self.padding = padding
		self.use_batch_normalization = use_batch_normalization
		self.use_dropout = use_dropout


	def unpack_hyperparameters(self):
		"""
		This method returns all the hyperparameters of the architecture configuration

		Input:
			None
		
		Returns:
			convolutional_layer_sizes (list): A list of integers containing the number of filters per convolutional layer
			convolutinal_layer_kernels (list): A list of tuples containing the size of the kernel per convolutional layer
			convolutional_activation (string): The name of the activation function in the convolutional layers		
			output_activation (string): The name of the activation in the output layer
			use_batch_normalization (bool): True if use batch normalization between hidden layers
			model_name (string): The name of the model
		"""

		return self.fc_layer_sizes, \
			   self.fc_activation, \
			   self.convolutional_layer_sizes, \
			   self.convolutinal_layer_kernels, \
			   self.convolutional_activation, \
			   self.output_activation, \
			   self.regularizer, \
			   self.model_name, \
			   self.padding, \
			   self.use_batch_normalization


class CompilationConfiguration(ConfigurationElement):
	"""
	This class contains the model compilation hyperparameters
	"""

	def __init__(
		self, 
		loss_function,
		optimizer,
		metric
		):
		super(CompilationConfiguration, self).__init__()
		self.loss_function = loss_function
		self.optimizer = optimizer
		self.metric = metric


	def unpack_hyperparameters(
		self
		):
		"""
		This method returns all the hyperparameters of the architecture configuration

		Input:
			None
		
		Returns:
			loss_function (keras.losses): A loss function to backpropagate
			optimizer (keras.optimizers): An optimizer to tune the neural network
			learning_rate (float): The learning rate used by the optimizer
			metric (keras.metrics): The metric to monitor to execute the callbacks
		"""

		return self.loss_function, \
			   self.optimizer, \
			   self.metric


class TrainingConfiguration(ConfigurationElement):
	"""
	This class contains the training hyperparameters
	"""
	def __init__(
		self, 
		epochs,
		batch_size,
		callbacks
		):
		super(TrainingConfiguration, self).__init__()
		self.epochs = epochs
		self.batch_size = batch_size
		self.callbacks = callbacks


	def unpack_hyperparameters(
		self
		):
		"""
		This method returns all the hyperparameters of the architecture configuration

		Input:
			None
		
		Returns:
			epochs (int): The number of epochs to train the model
			batch_size (batch_size): The size of the batch inside the epochs
			callbacks (list): A list of callbacks to use during the training
		"""

		return self.epochs, \
			   self.batch_size, \
			   self.callbacks



class Configuration(ABC):
	"""
	This class encapsulates the hyperparameter configuration for architecture, compilation and training of a model
	"""

	def __init__(
		self,
		architecture_hyperparams: ConfigurationElement,
		compilation_hyperparams: ConfigurationElement,
		traning_hyperparams: ConfigurationElement,
		description: str
		):

		self.architecture_hyperparams = architecture_hyperparams
		self.compilation_hyperparams = compilation_hyperparams
		self.training_hyperparams = traning_hyperparams
		self.description = description


	def unpack_architecture_hyperparameters(self):
		return self.architecture_hyperparams.unpack_hyperparameters()

	def unpack_compilation_hyperparameters(self):
		return self.compilation_hyperparams.unpack_hyperparameters()

	def unpack_training_hyperparameters(self):
		return self.training_hyperparams.unpack_hyperparameters()

	def get_description(self):
		return self.description


################################################################################## FULLY CONNECTED
def SimpleFCModel(
	):
	"""
	Function that creates the model configuration for the first working model (a fully connected one)
	"""

	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (1)
	hidden_layer_sizes = [128, 128, 128, 128, 256, 256, 512, 2000, 4000]
	regularizer = None
	hidden_activation = 'rel'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = False
	dropout_rate = 0.2
	model_name = "Flux-Sum"

	architecture_hyperparams = FullyConnectedArchitecture(
									input_shape, 
                                    output_shape, 
                                    hidden_layer_sizes, 
                                    regularizer,
                                    hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name,
                                    use_dropout=use_dropout,
                                    dropout_rate=dropout_rate
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Fully Connected
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Hidden layers: {hidden_layer_sizes}
		-Regularizer: None
		-Hidden Layers Activation: {hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
		-Dropout: {use_dropout}, {dropout_rate}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 200
	batch_size = 64
	
	reduce_lr = ReduceLROnPlateau(
					'val_mean_squared_error', 
					factor=0.1, 
					patience=10, 
					verbose=1)
	early_stop = EarlyStopping(
					'val_mean_squared_error',
					patience=25, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 10 x0.1
			-Early Stop: MSE 25
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def PSFSimpleFCModel(
	):
	"""
	Function that creates the model configuration for the first working model (a fully connected one)
	"""

	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (2*128*128)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = False
	dropout_rate = 0.2
	model_name = "PSFRecontructorSuperBigFC10000"

	architecture_hyperparams = FullyConnectedArchitecture(
									input_shape, 
                                    output_shape, 
                                    hidden_layer_sizes, 
                                    regularizer,
                                    hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name,
                                    use_dropout=use_dropout,
                                    dropout_rate=dropout_rate
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Fully Connected
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Hidden layers: {hidden_layer_sizes}
		-Regularizer: None
		-Hidden Layers Activation: {hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
		-Dropout: {use_dropout}, {dropout_rate}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 200
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=20, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=50, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 20 x0.1
			-Early Stop: MSE 50
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration



def SimpleFCWithBN(
	):
	"""
	Function that creates the model configuration for the first working model adding batch normalization
	"""

	# Define architecture hyperparmeters
	input_shape = FC_INPUT_SHAPE
	output_shape = FC_OUTPUT_SHAPE
	hidden_layer_sizes = [2048, 512, 2048, 4096]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = True
	model_name = "NewConv10000"

	architecture_hyperparams = FullyConnectedArchitecture(
									input_shape, 
                                    output_shape, 
                                    hidden_layer_sizes, 
                                    regularizer,
                                    hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Fully Connected
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Hidden layers: {hidden_layer_sizes}
		-Regularizer: None
		-Hidden Layers Activation: {hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.0001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 1000
	batch_size = 128
	
	reduce_lr = ReduceLROnPlateau(
					'val_mean_squared_error', 
					factor=0.1, 
					patience=15, 
					verbose=1)
	early_stop = EarlyStopping(
					'val_mean_squared_error',
					patience=50, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 15 x0.1
			-Early Stop: MSE 50
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def FullyConnectedDropoutAndBN(
	):
	"""
	Function that creates the model configuration for a fully connected model with dropout and batch normalization
	"""

	# Define architecture hyperparmeters
	input_shape = FC_INPUT_SHAPE
	output_shape = FC_OUTPUT_SHAPE
	hidden_layer_sizes = [256, 256, 128, 128, 64, 64, 512, 512, 1024]
	regularizer = L1(0.0001)
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = True
	model_name = "FullyConnectedDropoutAndBN"
	use_dropout = True
	dropout_rate = 0.2

	architecture_hyperparams = FullyConnectedArchitecture(
									input_shape, 
                                    output_shape, 
                                    hidden_layer_sizes, 
                                    regularizer,
                                    hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name,
                                    use_dropout,
                                    dropout_rate
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Fully Connected
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Hidden layers: {hidden_layer_sizes}
		-Regularizer: L1 0.0001
		-Hidden Layers Activation: {hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
		-Dropout: {use_dropout} {dropout_rate}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.0001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 500
	batch_size = 128
	
	reduce_lr = ReduceLROnPlateau(
					'val_mean_squared_error', 
					factor=0.1, 
					patience=15, 
					verbose=1)
	early_stop = EarlyStopping(
					'val_mean_squared_error',
					patience=50, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 15 x0.1
			-Early Stop: MSE 50
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def FCDropoutOnly(
	):
	"""
	Function that creates the model configuration for a fully connected archictecture with dropout as the only regularizing tool
	"""

	# Define architecture hyperparmeters
	input_shape = FC_INPUT_SHAPE
	output_shape = FC_OUTPUT_SHAPE
	hidden_layer_sizes = [1024, 2048, 2048, 2048]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	model_name = "FCDropoutOnly"
	use_dropout = False
	dropout_rate = 0.1

	architecture_hyperparams = FullyConnectedArchitecture(
									input_shape, 
                                    output_shape, 
                                    hidden_layer_sizes, 
                                    regularizer,
                                    hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name,
                                    use_dropout,
                                    dropout_rate
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Fully Connected
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Hidden layers: {hidden_layer_sizes}
		-Regularizer: None
		-Hidden Layers Activation: {hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
		-Dropout: {use_dropout} {dropout_rate}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.01
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 200
	batch_size = 64
	
	reduce_lr = ReduceLROnPlateau(
					'val_mean_squared_error', 
					factor=0.1, 
					patience=15, 
					verbose=1)
	early_stop = EarlyStopping(
					'val_mean_squared_error',
					patience=50, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-Early Stop: MSE 50
			-ReduceLROnPlateau: MSE 15 x0.1
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def FCDropoutL1(
	):
	"""
	Function that creates the model configuration for a fully connected model with dropout and L1 regularization
	"""

	# Define architecture hyperparmeters
	input_shape = FC_INPUT_SHAPE
	output_shape = FC_OUTPUT_SHAPE
	hidden_layer_sizes = [1024, 2048, 2048, 2048]
	regularizer = L1(0.0001)
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	model_name = "FCDropoutL1"
	use_dropout = True
	dropout_rate = 0.1

	architecture_hyperparams = FullyConnectedArchitecture(
									input_shape, 
                                    output_shape, 
                                    hidden_layer_sizes, 
                                    regularizer,
                                    hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name,
                                    use_dropout,
                                    dropout_rate
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Fully Connected
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Hidden layers: {hidden_layer_sizes}
		-Regularizer: L1 (0.05)
		-Hidden Layers Activation: {hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
		-Dropout: {use_dropout} {dropout_rate}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 300
	batch_size = 16
	
	reduce_lr = ReduceLROnPlateau(
					'val_mean_squared_error', 
					factor=0.1, 
					patience=30, 
					verbose=1)
	early_stop = EarlyStopping(
					'val_mean_squared_error',
					patience=100, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 30 x0.1
			-Early Stop: MSE 100
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def FCDropoutL2(
	):
	"""
	Function that creates the model configuration for a fully connected model with dropout and L2 regularization
	"""

	# Define architecture hyperparmeters
	input_shape = FC_INPUT_SHAPE
	output_shape = FC_OUTPUT_SHAPE
	hidden_layer_sizes = [1024, 2048, 2048, 2048]
	regularizer = L2(0.0001)
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	model_name = "FCDropoutL2"
	use_dropout = True
	dropout_rate = 0.1

	architecture_hyperparams = FullyConnectedArchitecture(
									input_shape, 
                                    output_shape, 
                                    hidden_layer_sizes, 
                                    regularizer,
                                    hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name,
                                    use_dropout,
                                    dropout_rate
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Fully Connected
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Hidden layers: {hidden_layer_sizes}
		-Regularizer: L2 (0.0001)
		-Hidden Layers Activation: {hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
		-Dropout: {use_dropout} {dropout_rate}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.0001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 200
	batch_size = 64
	
	reduce_lr = ReduceLROnPlateau(
					'val_mean_squared_error', 
					factor=0.1, 
					patience=30, 
					verbose=1)
	early_stop = EarlyStopping(
					'val_mean_squared_error',
					patience=100, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 30 x0.1
			-Early Stop: MSE 100
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


########################################################################################### CONVOLUTIONAL MODELS
def SimpleConvolutional(
	):
	"""
	Function that creates the model configuration for the first convolutional model
	"""

	# Define architecture hyperparmeters
	input_shape = CNN_INPUT_SHAPE
	output_shape = CNN_OUTPUT_SHAPE
	convolutional_layer_sizes = [128, 256, 512]	
	convolutinal_layer_kernels = [(3,3), (3,3), (3,3)]
	fully_connected_hidden_layer_sizes = [4096, 2048, 2048, 1024, 1024, 1024]
	regularizer = None
	convolutional_activation = 'relu'
	fully_connected_hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = True
	model_name="NewConv80000"

	architecture_hyperparams = ConvolutionalArchitecture(
									input_shape, 
                                    output_shape,
                                    convolutional_layer_sizes,
                                    convolutinal_layer_kernels,
                                    fully_connected_hidden_layer_sizes, 
                                    regularizer,
                                    convolutional_activation,
                                    fully_connected_hidden_activation,
                                    output_activation,
                                    use_batch_normalization,
                                    model_name
                                    )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Convolutional
		-Input shape: {input_shape}
		-Output shape: {output_shape}
		-Convolutional Layers: {convolutional_layer_sizes}
		-Convolutonal Kernels: {convolutinal_layer_kernels}
		-Fully Connected Hidden layers: {fully_connected_hidden_layer_sizes}
		-Regularizer: None
		-Convolutional Activation: {convolutional_activation}
		-Hidden Layers Activation: {fully_connected_hidden_activation}
		-Output Layer Activation: {output_activation}
		-Batch Normalization: {use_batch_normalization}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 200
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'val_mean_squared_error', 
					factor=0.1, 
					patience=15, 
					verbose=1)
	early_stop = EarlyStopping(
					'val_mean_squared_error',
					patience=50, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 15 x0.1
			-Early Stop: MSE 50
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def AutoEncoderConfiguration(
	):
	"""
	Function that creates the model configuration for a flux autoencoder
	"""

	# Define architecture hyperparmeters
		
		
	input_shape = AUTOENCODER_INPUT_SHAPE
	convolutional_layer_sizes = [512, 128, 64, 8]
	convolutinal_layer_kernels = [(3,3), (3,3), (3,3), (3,3)]
	convolutional_activation = 'relu'
	output_activation = 'linear'
	model_name="NewFluxAutoencoder80000"
	padding="same"
	use_batch_normalization=True

	architecture_hyperparams = AutoEncoderArchitecture(
									input_shape,
									convolutional_layer_sizes,
									convolutinal_layer_kernels,
									convolutional_activation,
									output_activation,
									model_name=model_name,
									padding=padding,
									use_batch_normalization=use_batch_normalization
                                )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Autoencoder
		-Input shape: {input_shape}
		-Convolutional Layers: {convolutional_layer_sizes} (Inverse in the decoder)
		-Convolutonal Kernels: {convolutinal_layer_kernels} (Inverse in the decoder)
		-Convolutional Activation: {convolutional_activation}
		-Output Layer Activation: {output_activation}
		-Padding: {padding}
		-Use Batch Normalization: {use_batch_normalization}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 75
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=5, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=15, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 8 x0.1
			-Early Stop: MSE 15
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def EncoderConvolutionalConfiguration(
	):
	"""
	Function that creates the model configuration for a model with a frozen encoder and a convolutional decoder for amplitude and phase
	"""

	# Define architecture hyperparmeters
		
		
	convolutional_layer_sizes = [1024, 512, 256, 256]
	convolutinal_layer_kernels = [(3,3), (3,3), (3,3), (3,3)]
	convolutional_activation = 'relu'
	output_activation = 'linear'
	model_name="NewEncConv80000"

	architecture_hyperparams = EncoderConvolutionalArchitecture(
									convolutional_layer_sizes,
									convolutinal_layer_kernels,
									convolutional_activation,
									output_activation,
									model_name=model_name
                                )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-Encoder + Convolutional
		-Convolutional Layers: {convolutional_layer_sizes}
		-Convolutonal Kernels: {convolutinal_layer_kernels}
		-Convolutional Activation: {convolutional_activation}
		-Output Layer Activation: {output_activation}
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.0001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 100
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=5, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=15, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 8 x0.1
			-Early Stop: MSE 15
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def PSFConvolutionalConfiguration():
	"""
	Function that creates the model configuration for a model with a frozen encoder and a convolutional decoder for amplitude and phase
	"""

	# Define architecture hyperparmeters
		
	fc_layer_sizes = [2048, 2048, 2048, 2048, 2048]
	fc_activation = 'elu'
	convolutional_layer_sizes = [512, 128, 32]
	convolutinal_layer_kernels = [3, 3, 3]
	convolutional_activation = 'relu'
	output_activation = 'linear'
	regularizer = None
	model_name = "PSFConvolutional"
	use_batch_normalization = True
	use_dropout = False

	architecture_hyperparams = PSFConvolutionalArchitecture(
									fc_layer_sizes,
									fc_activation,
									convolutional_layer_sizes,
									convolutinal_layer_kernels,
									convolutional_activation,
									output_activation,
									regularizer,
									model_name=model_name,
									use_batch_normalization=use_batch_normalization,
									use_dropout=use_dropout
                                )

	description = f"""
	=== {model_name} ===
	*ARCHITECTURE HYPERPARAMETERS:
		-PSF Convolutional
		-Fully connected Layers: {fc_layer_sizes}
		-Fully connected Activation: {fc_activation}
		-Convolutional Layers: {convolutional_layer_sizes}
		-Convolutonal Kernels: {convolutinal_layer_kernels}
		-Convolutional Activation: {convolutional_activation}
		-Output Layer Activation: {output_activation}
		-Regularizer: None
		-Batch Normalization: {use_batch_normalization}
		-Dropout: {use_dropout} 
	"""

	# Define compilation hyperparameters
	loss_function = LossesMeanSquaredError()
	learning_rate = 0.001
	optimizer = Adam(
		learning_rate=learning_rate,
		beta_1=0.9,
		beta_2=0.999
		)
	metric = MetricsMeanSquaredError()

	compilation_hyperparams = CompilationConfiguration(
								loss_function, 
								optimizer, 
								metric)

	description += f"""
	*COMPILATION HYPERPARAMETERS:
		-Optimizer: ADAM lr={learning_rate}, beta_1=0.9, beta_2=0.999
		-Loss Function: MSE
		-Metric: MSE
	"""

	# Define training hyperparameters
	epochs = 100
	batch_size = 16
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=5, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=15, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks)

	description += f"""
	* TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Batch size: {batch_size}
		-Callbacks:
			-ReduceLROnPlateau: MSE 8 x0.1
			-Early Stop: MSE 15
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration



