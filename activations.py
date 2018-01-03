import tensorflow as tf

class Activations():

	def __init__(self):
		self.map = {
			"none": self.constant,
			"selu": self.selu,
			"softmax": self.softmax
		}

	def resolve_activation(self, name):
		return self.map [name]
	
	def selu(self, x):
		"""
		Implementation of SELU activation: https://arxiv.org/abs/1706.02515
		Credit: https://github.com/bioinf-jku/SNNs/blob/master/selu.py
		Args:
		    x (tensor): Layer of value to apply activation to
		Returns:
			(tensor): Scaled exponential linear unit applied to input tensor
		"""
		with tf.name_scope('elu') as scope:
			alpha = 1.6732632423543772848170429916717
			scale = 1.0507009873554804934193349852946
			return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

	def softmax(self, x):
		return tf.nn.softmax(x)

	def constant(self, x):
		return x
	