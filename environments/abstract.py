from abc import ABCMeta, abstractmethod

class Environment():
	__metaclass__ = ABCMeta

	def __init__(self):
		self.discrete = True

	@abstractmethod
	def reset(self):
		"""
		Reset current position to beginning.
		"""
		pass

	@abstractmethod
	def act(self, action, population, master):
		"""
		Move end effector to the given location
		"""
		pass

	@abstractmethod
	def inputs(self, timestep):
		"""
		Return the inputs for the neural network
		"""
		pass

	@abstractmethod
	def reward_params(self, moved):
		"""
		Return the parameters for the proposed reward function
		"""
		pass

	@abstractmethod
	def pre_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass

	@abstractmethod
	def post_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass