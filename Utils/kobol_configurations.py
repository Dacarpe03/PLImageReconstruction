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
		callbacks,
		n_samples
		):
		super(TrainingConfiguration, self).__init__()
		self.epochs = epochs
		self.batch_size = batch_size
		self.callbacks = callbacks
		self.n_samples = n_samples


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
			   self.callbacks, \
			   self.n_samples



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


def CroppedSimpleFC(
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (2*64*64)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = False
	dropout_rate = 0.2
	model_name = f"CroppedSimpleFC{n_samples}"

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
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=50, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def CroppedDR01FC(
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (2*64*64)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = True
	dropout_rate = 0.1
	model_name = f"CroppedDR01FC{n_samples}"

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
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=50, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


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
		callbacks,
		n_samples
		):
		super(TrainingConfiguration, self).__init__()
		self.epochs = epochs
		self.batch_size = batch_size
		self.callbacks = callbacks
		self.n_samples = n_samples


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
			   self.callbacks, \
			   self.n_samples



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


def CroppedSimpleFC(
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (2*64*64)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = False
	dropout_rate = 0.2
	model_name = f"CroppedSimpleFC{n_samples}"

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
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=50, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def CroppedDR02FC(
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (2*64*64)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = True
	dropout_rate = 0.2
	model_name = f"CroppedDR02FC{n_samples}"

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
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=50, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def CroppedBNFC(
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (2*64*64)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = True
	use_dropout = False
	dropout_rate = 0.2
	model_name = f"CroppedBNFC{n_samples}"

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
	batch_size = 32
	
	reduce_lr = ReduceLROnPlateau(
					'mean_squared_error', 
					factor=0.1, 
					patience=50, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def ZernikeCroppedFC(
		name="ZernikeCroppedFC",
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (2*64*64)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = False
	dropout_rate = 0.2
	model_name = f"{name}{n_samples}"

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
					patience=40, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def ZernikeCroppedIntensityFC(
		name="ZernikeCroppedIntensityFC",
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (64*64)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = False
	dropout_rate = 0.2
	model_name = f"{name}{n_samples}"

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
					patience=40, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def ZernikeFC(
		name="ZernikeFC",
		n_samples=10000
	):
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
	model_name = f"{name}{n_samples}"

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
					patience=40, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration


def ZernikeIntensityFC(
		name="ZernikeIntensityFC",
		n_samples=10000
	):
	# Define architecture hyperparmeters
	input_shape = 19
	output_shape = (128*128)
	hidden_layer_sizes = [1024, 1024, 1024, 1024, 1024, 1024]
	regularizer = None
	hidden_activation = 'relu'
	output_activation = 'linear'
	use_batch_normalization = False
	use_dropout = False
	dropout_rate = 0.2
	model_name = f"{name}{n_samples}"

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
					patience=40, 
					verbose=1)
	early_stop = EarlyStopping(
					'mean_squared_error',
					patience=70, 
					verbose=1)
	callbacks = [reduce_lr, early_stop]

	training_hyperparameters = TrainingConfiguration(
									epochs,
									batch_size,
									callbacks,
									n_samples)

	description += f"""
	*TRAINING HYPERPARAMETERS:
		-Epochs: {epochs}
		-Samples: {n_samples}
		-Batch size: {batch_size}
		-Callbacks: 
			-ReduceLROnPlateau: MSE 50 x0.1
			-Early Stop: MSE 70
	"""

	model_configuration = Configuration(
							architecture_hyperparams,
							compilation_hyperparams,
							training_hyperparameters,
							description
							)

	return model_configuration