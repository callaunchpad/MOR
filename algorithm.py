import sys
import inspect
import logging
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import Config
from models import resolve_model
from rewards import resolve_reward
from env import test_cases, resolve_env


class NES():
	"""
	Implementation of NES algorithm by OpenAI: https://arxiv.org/pdf/1703.03864.pdf
	"""

	def __init__(self, training_directory):
		self.config = Config().config
		self.training_directory = training_directory
		self.env = resolve_env(self.config['environment'])(test_cases[self.config['environment']][self.config['environment_index']])
		if not self.env.solution_exists():
			raise Exception("No solution exists for the given environment.")
		else:
			print("{} Solution exists for the given environment.\n".format('\x1b[6;30;42m' + 'Success' + '\x1b[0m'))
		self.model = resolve_model(self.config['model'])
		self.reward = resolve_reward(self.config['reward'])
		self.master_params = self.model.init_master_params()
		logging.info("\nReward:")
		logging.info(inspect.getsource(self.reward) + "\n")

	def run_simulation(self, sample_params, model):
		"""
		Black box interaction with environment using model as the action decider given environmental inputs.
		Args:
		    sample_params (tensor): Master parameters jittered with gaussian noise
		    model (tensor): The output layer for a tensorflow model
		Returns:
			reward (float): Fitness function evaluated on the completed trajectory
		"""
		with tf.Session() as sess:
			reward = 0
			moved = False
			for t in range(self.config['n_timesteps_per_trajectory']):
				inputs = np.array([self.env.current[0], self.env.current[1], self.env.target[0], self.env.target[1], self.env.is_wall(self.env.current, 0), self.env.is_wall(self.env.current, 1), self.env.is_wall(self.env.current, 2), self.env.is_wall(self.env.current, 3), t+1]).reshape((1, self.config['input_size']))
				action_dist = sess.run(model, self.model.feed_dict(inputs, sample_params))
				direction = np.argmax(action_dist)
				moved = self.env.move(direction)
				logging.info("Action: {}, Current Position: {}".format(self.env.direction_name(direction), self.env.current))
			reward += self.reward(self.env.current, self.env.target, moved)
			self.env.reset()
			return reward

	def update(self, noise_samples, rewards):
		"""
		Update function for the master parameters (weights and biases for neural network)
		Args:
		    noise_samples (float array): List of the noise samples for each individual in the population
		    rewards (float array): List of rewards for each individual in the population
		"""
		normalized_rewards = (rewards - np.mean(rewards))
		if np.std(rewards) != 0.0:
			normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
		self.master_params += (self.config['learning_rate'] / (self.config['n_individuals'] * self.config['noise_std_dev'])) * np.dot(noise_samples.T, normalized_rewards)

	def run(self):
		"""
		Run NES algorithm given parameters from config.
		"""
		model = self.model.model()
		n_reached_target = []
		population_rewards = []
		for p in range(self.config['n_populations']):
			logging.info("Population: {}\n{}".format(p+1, "="*30))
			noise_samples = np.random.randn(self.config['n_individuals'], len(self.master_params))
			rewards = np.zeros(self.config['n_individuals'])
			n_individual_target_reached = 0
			for i in range(self.config['n_individuals']):
				logging.info("Individual: {}".format(i+1))
				sample_params = self.master_params + noise_samples[i]
				rewards[i] = self.run_simulation(sample_params, model)
				n_individual_target_reached += rewards[i] == 1
				logging.info("Individual {} Reward: {}\n".format(i+1, rewards[i]))
			self.update(noise_samples, rewards)
			n_reached_target.append(n_individual_target_reached)
			population_rewards.append(sum(rewards)/len(rewards))
			self.plot_graphs([range(p+1), range(p+1)], [population_rewards, n_reached_target], ["Average Reward per population", "Number of times target reached per Population"], ["reward.png", "success.png"], ["line", "scatter"])
		logging.info("Reached Target {} Total Times".format(sum(n_reached_target)))

	def plot_graphs(self, x_axes, y_axes, titles, filenames, types):
		for i in range(len(x_axes)):
			plt.title(titles[i])
			if types[i] == "line":
				plt.plot(x_axes[i], y_axes[i])
			if types[i] == "scatter":
				plt.scatter(x_axes[i], y_axes[i])
				plt.plot(np.unique(x_axes[i]), np.poly1d(np.polyfit(x_axes[i], y_axes[i], 1))(np.unique(x_axes[i])))
			plt.savefig(self.training_directory + filenames[i])
			plt.clf()