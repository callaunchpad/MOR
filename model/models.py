import numpy as np
import tensorflow as tf
from activations import Activations

def resolve_model(name):
	models = {
		"FeedForwardNeuralNetwork": FeedForwardNeuralNetwork,
		"BatchNormalizationNeuralNetwork": BatchNormalizationNeuralNetwork
	}
	return models[name]

class FeedForwardNeuralNetwork():

	def __init__(self, config):
		self.config = config
		self.activations = Activations()
		self.inputs = tf.placeholder(tf.float32, shape=(1, self.config['input_size']))
		params_size = (self.config['input_size'] * self.config['n_nodes_per_layer']) + self.config['n_nodes_per_layer'] + self.config['n_hidden_layers'] * (self.config['n_nodes_per_layer']**2 + self.config['n_nodes_per_layer']) + (self.config['n_nodes_per_layer'] * self.config['output_size']) + self.config['output_size']
		self.params = tf.placeholder(tf.float32)

	def model(self):
		"""
		Builds Tensorflow graph
		Returns:
			(tensor): Output Tensor for the graph
		"""
		start = 0
		weights = tf.reshape(self.params[start : self.config['input_size'] * self.config['n_nodes_per_layer']], [self.config['input_size'], self.config['n_nodes_per_layer']])
		start += self.config['input_size'] * self.config['n_nodes_per_layer']
		biases = tf.reshape(self.params[start : start + self.config['n_nodes_per_layer']], [self.config['n_nodes_per_layer']])
		start += self.config['n_nodes_per_layer']
		hidden_layer = self.activations.resolve_activation(self.config['hidden_layer_activation'])(tf.add(tf.matmul(self.inputs, weights), biases))

		for i in range(self.config['n_hidden_layers']):
			weights = tf.reshape(self.params[start : start + self.config['n_nodes_per_layer'] * self.config['n_nodes_per_layer']], [self.config['n_nodes_per_layer'], self.config['n_nodes_per_layer']])
			start += self.config['n_nodes_per_layer'] * self.config['n_nodes_per_layer']
			biases = tf.reshape(self.params[start : start + self.config['n_nodes_per_layer']], [self.config['n_nodes_per_layer']])
			start += self.config['n_nodes_per_layer']
			hidden_layer = self.activations.resolve_activation(self.config['hidden_layer_activation'])(tf.add(tf.matmul(hidden_layer, weights), biases))

		weights = tf.reshape(self.params[start : start + self.config['n_nodes_per_layer'] * self.config['output_size']], [self.config['n_nodes_per_layer'], self.config['output_size']])
		start += self.config['n_nodes_per_layer'] * self.config['output_size']
		biases = tf.reshape(self.params[start : start + self.config['output_size']], [self.config['output_size']])
		start += self.config['output_size']
		output_layer = tf.scalar_mul(self.config['output_scale'], self.activations.resolve_activation(self.config['output_activation'])(tf.add(tf.matmul(hidden_layer, weights), biases)))
		return output_layer

	def init_master_params(self):
		"""
		Computes initial random gaussian values for master weights and biases
		Returns:
			(float array): Random gaussian values for neural network weights and biases
		"""
		master_params = []
		weights = np.random.normal(0, 1, self.config['input_size'] * self.config['n_nodes_per_layer'])
		master_params += list(weights)
		biases = np.random.normal(0, 1, self.config['n_nodes_per_layer'])
		master_params += list(biases)

		for i in range(self.config['n_hidden_layers']):
			weights = np.random.normal(0, 1, self.config['n_nodes_per_layer'] * self.config['n_nodes_per_layer'])
			master_params += list(weights)
			biases = np.random.normal(0, 1, self.config['n_nodes_per_layer'])
			master_params += list(biases)

		weights = np.random.normal(0, 1, self.config['n_nodes_per_layer'] * self.config['output_size'])
		master_params += list(weights)
		biases = np.random.normal(0, 1, self.config['output_size'])
		master_params += list(biases)
		return master_params

	def feed_dict(self, inputs, params):
		"""
		Fills the feed_dict for the Tensorflow graph
		Returns:
			(dict): Feed_dict filled with given values for placeholders
		"""
		return {self.inputs: inputs, self.params: params}

class BatchNormalizationNeuralNetwork():
	def __init__(self, config):
		self.config = config
		self.activations = Activations()
		self.inputs = tf.placeholder(tf.float32, shape=(1, self.config['input_size']))
		params_size = (self.config['input_size'] * self.config['n_nodes_per_layer']) + self.config['n_nodes_per_layer'] + 2*self.config['input_size'] + self.config['n_hidden_layers'] * (self.config['n_nodes_per_layer']**2 + self.config['n_nodes_per_layer'] + 2*self.config['n_nodes_per_layer']) + (self.config['n_nodes_per_layer'] * self.config['output_size']) + self.config['output_size'] + 2*self.config['n_nodes_per_layer']
		self.params = tf.placeholder(tf.float32, shape=(params_size))
		self.learning_rate = self.config['learning_rate']
		self.keep_rate = self.config['keep_rate']

	def model(self):
		"""
		Builds Tensorflow graph
		Returns:
			(tensor): Output Tensor for the graph
		"""
		start = 0
		nodes_per_layer = self.config['n_nodes_per_layer']
		input_size = self.config['input_size']
		activation = self.config['hidden_layer_activation']
		num_hidden_layers = self.config['n_hidden_layers']
		output_size = self.config['output_size']
		output_scale = self.config['output_scale']
		output_activation = self.config['output_activation']

		weights = tf.reshape(self.params[start : input_size * nodes_per_layer], [input_size, nodes_per_layer])
		start += input_size * nodes_per_layer
		biases = tf.reshape(self.params[start : start + nodes_per_layer], [nodes_per_layer])
		start += nodes_per_layer

		mean, var = tf.nn.moments(weights, [1], keep_dims=True)
		stddev = tf.sqrt(var)
		weights = tf.div(tf.subtract(weights, mean), stddev)
		gamma = tf.reshape(self.params[start : start + input_size], [input_size, 1])
		start += input_size
		beta = tf.reshape(self.params[start : start + input_size], [input_size, 1])
		start += input_size
		weights = tf.add(tf.multiply(gamma, weights), beta)

		hidden_layer = self.activations.resolve_activation(activation)(tf.add(tf.matmul(self.inputs, weights), biases))

		for i in range(num_hidden_layers):
			weights = tf.reshape(self.params[start : start + nodes_per_layer * nodes_per_layer], [nodes_per_layer, nodes_per_layer])
			start += nodes_per_layer * nodes_per_layer
			biases = tf.reshape(self.params[start : start + nodes_per_layer], [nodes_per_layer])
			start += nodes_per_layer

			# Add batch normalization by subtracting the mean and dividing by the std dev
			mean, var = tf.nn.moments(weights, [1], keep_dims=True)
			stddev = tf.sqrt(var)
			weights = tf.div(tf.subtract(weights, mean), stddev)

			gamma = tf.reshape(self.params[start : start + nodes_per_layer], [nodes_per_layer, 1])
			start += nodes_per_layer
			beta = tf.reshape(self.params[start : start + nodes_per_layer], [nodes_per_layer, 1])
			start += nodes_per_layer
			weights = tf.add(tf.multiply(gamma, weights), beta)

			hidden_layer = self.activations.resolve_activation(activation)(tf.add(tf.matmul(hidden_layer, weights), biases))
			hidden_layer = tf.nn.dropout(hidden_layer, self.keep_rate)

		weights = tf.reshape(self.params[start : start + nodes_per_layer * output_size], [nodes_per_layer, output_size])
		start += nodes_per_layer * output_size
		biases = tf.reshape(self.params[start : start + output_size], [output_size])
		start += output_size

		mean, var = tf.nn.moments(weights, [1], keep_dims=True)
		stddev = tf.sqrt(var)
		weights = tf.div(tf.subtract(weights, mean), stddev)

		gamma = tf.reshape(self.params[start : start + nodes_per_layer], [nodes_per_layer, 1])
		start += nodes_per_layer
		beta = tf.reshape(self.params[start : start + nodes_per_layer], [nodes_per_layer, 1])
		start += nodes_per_layer
		weights = tf.add(tf.multiply(gamma, weights), beta)

		output_layer = tf.scalar_mul(output_scale, self.activations.resolve_activation(output_activation)(tf.add(tf.matmul(hidden_layer, weights), biases)))
		return output_layer

	def init_master_params(self):
		"""
		Computes initial random gaussian values for master weights and biases
		Returns:
			(float array): Random gaussian values for neural network weights and biases
		"""
		master_params = []
		weights = np.random.normal(0, 1, self.config['input_size'] * self.config['n_nodes_per_layer'])
		master_params += list(weights)
		biases = np.random.normal(0, 1, self.config['n_nodes_per_layer'])
		master_params += list(biases)
		master_params += [1 for i in range(self.config['input_size'])]
		master_params += [0 for i in range(self.config['input_size'])]

		for i in range(self.config['n_hidden_layers']):
			weights = np.random.normal(0, 1, self.config['n_nodes_per_layer'] * self.config['n_nodes_per_layer'])
			master_params += list(weights)
			biases = np.random.normal(0, 1, self.config['n_nodes_per_layer'])
			master_params += list(biases)
			master_params += [1 for i in range(self.config['n_nodes_per_layer'])]
			master_params += [0 for i in range(self.config['n_nodes_per_layer'])]

		weights = np.random.normal(0, 1, self.config['n_nodes_per_layer'] * self.config['output_size'])
		master_params += list(weights)
		biases = np.random.normal(0, 1, self.config['output_size'])
		master_params += list(biases)
		master_params += [1 for i in range(self.config['n_nodes_per_layer'])]
		master_params += [0 for i in range(self.config['n_nodes_per_layer'])]
		return master_params

	def feed_dict(self, inputs, params):
		"""
		Fills the feed_dict for the Tensorflow graph
		Returns:
			(dict): Feed_dict filled with given values for placeholders
		"""
		return {self.inputs: inputs, self.params: params}