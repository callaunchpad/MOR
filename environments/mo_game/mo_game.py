import pygame

class MOGame(Environment):

	def __init__(self):
		self.discrete = False

	def reset(self):
		"""
		Reset current position to beginning.
		"""
		pass

	def act(self, action, population, params, master):
		"""
		Move end effector to the given location
		"""
		pass

	def inputs(self, timestep):
		"""
		Return the inputs for the neural network
		"""
		pass

	def reward_params(self, moved):
		"""
		Return the parameters for the proposed reward function
		"""
		pass

	def pre_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass

	def post_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass

	def reached_target(self):
		"""
		Check if the target goal was achieved
		"""
		pass