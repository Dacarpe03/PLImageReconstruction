from abc import ABC, abstractmethod

class ConfigurationElement(ABC):
	@abstractmethod
	def unpack_hyperparameters(self):
		"""
		This method returns all the hyperparameters of the configuration element
		"""
		pass



class FullyConnectedArchitecture(ConfigurationElement):
	"""This class contains the information of the architecture of a model"""
	def __init__(
		self, 
		input_shape,
		output_shape,
		hidden_layer_sizes,
		regularizer,
		hidden_activation,
		output_activation,
		use_batch_normalization,
		model_name
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
		"""

		return self.input_shape, \
			   self.output_shape, \
			   self.hidden_layer_sizes, \
			   self.regularizer, \
			   self.hidden_activation, \
			   self.output_activation, \
			   self.use_batch_normalization, \
			   self.model_name


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
			callbacks (list): A list of callbacks to use during the trainingx
		"""

		return self.epochs, \
			   self.batch_size, \
			   self.callbacks
