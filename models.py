import numpy as np
import tensorflow as tf
from config import Config
from activations import Activations

def resolve_model(name):
	models = {
		"FeedForwardNeuralNetwork": FeedForwardNeuralNetwork
	}
	return models[name]()

class FeedForwardNeuralNetwork():

	def __init__(self):
		self.config = Config().config
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
		output_layer = self.activations.resolve_activation(self.config['output_activation'])(tf.add(tf.matmul(hidden_layer, weights), biases))
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
