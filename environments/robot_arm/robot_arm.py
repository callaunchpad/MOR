import numpy as np
import logging
import os, errno
from datetime import datetime
from ..abstract import Environment
from maddux.environment import Environment
from maddux.objects import Ball
from maddux.robots import simple_human_arm

class RobotArm(Environment):

	def __init__(self, env, training_directory):
		self.discrete = False
		self.training_directory = training_directory
		self.env = env
		self.arm = self.env.robot
		self.current = self.arm.end_effector_position()
		self.ball = self.env.dynamic_objects[0]
		self.target = self.ball.position
		# self.recording_queue = []
		logging.info("Robot Arm: End effector starts at {}".format(self.current))
		logging.info("Target: Ball at {}".format(self.target))

	def reset(self):
		"""
		Reset current position to beginning.
		"""
		self.arm.reset()
		self.current = self.arm.end_effector_position()

	def act(self, location, population, master):
		"""
		Move end effector to the given location
		"""
		success = True
		past = self.current
		self.current = location[0]
		if population % 100 == 0 and master:
			try:
				self.arm.ikine(location[0])
				timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
				training_path = self.training_directory + "/paths/"
				try:
				    os.makedirs(training_path)
				except OSError as e:
				    if e.errno != errno.EEXIST:
				        raise
				record_path = training_path + "pop_" + population + ".npy"
				video_path = training_path + "pop_" + population + ".mp4"
				self.arm.save_path(record_path)
				self.env.animate(duration=5.0, save_path=video_path)
				# self.recording_queue.append(record_path)
			except ValueError as e:
				success = False
				logging.warn("Could not solve IK for position: {}". format(location[0]))
		logging.info("Current Position: {}".format(self.current))
		return success

	def inputs(self, t):
		"""
		Return the inputs for the neural network
		"""
		inputs = [self.current[0], self.current[1], self.current[2], self.target[0], self.target[1], self.target[2], t+1]
		return inputs

	def reward_params(self, success):
		"""
		Return the parameters for the proposed reward function
		"""
		params = [self.current, self.target, success]
		return params

	def pre_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass

	def post_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		# logging.debug("Recording Videos")
		# for path in self.recording_queue:
		# 	self.env.animate(duration=5.0, save_path=path)
		# logging.debug("Completed recording all videos")
		pass