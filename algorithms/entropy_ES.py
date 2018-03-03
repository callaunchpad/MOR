import sys
import inspect
import logging
import numpy as np
from scipy.stats import entropy
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.models import resolve_model
from model.rewards import resolve_reward
from environments.env import test_cases, resolve_env


class EntES():
    """
    MaxEnt reward function.
    """

    def __init__(self, training_directory, config):
        self.config = config
        self.training_directory = training_directory
        self.model_save_directory = self.training_directory + 'params/'
        self.env = resolve_env(self.config['environment'])(test_cases[self.config['environment']][self.config['environment_index']], self.training_directory, self.config)
        self.env.pre_processing()
        self.model = resolve_model(self.config['model'])(self.config)
        self.reward = resolve_reward(self.config['reward'])
        self.config['MOR_flag'] == "True"
        self.multiple_rewards = resolve_multiple_rewards(self.config(['multiple_rewards']))
        self.master_params = self.model.init_master_params(self.config['from_file'], self.config['params_file'])
        self.learning_rate = self.config['learning_rate']
        self.noise_std_dev = self.config['noise_std_dev']
        self.moving_success_rate = 0
        if (self.config['from_file']):
            logging.info("\nLoaded Master Params from:")
            logging.info(self.config['params_file'])
        logging.info("\nReward:")
        logging.info(inspect.getsource(self.reward) + "\n")

    def run_simulation(self, sample_params, model, population, counts, master=False):
        """
        Black box interaction with environment using model as the action decider given environmental inputs.
        Args:
            sample_params (tensor): Master parameters jittered with gaussian noise
            model (tensor): The output layer for a tensorflow model
        Returns:
            reward (float): Fitness function evaluated on the completed trajectory
        """
        this_counts = {}
        with tf.Session() as sess:
            reward = 0
            valid = False
            for t in range(self.config['n_timesteps_per_trajectory']):
                inputs = np.array(self.env.inputs(t)).reshape((1, self.config['input_size']))
                net_output = sess.run(model, self.model.feed_dict(inputs, sample_params))
                probs = np.exp(net_output[0]) / np.sum(np.exp(net_output[0]))
                if self.env.discrete:
                    action = np.argmax(probs)
                    # action = np.random.choice(np.arange(probs.shape[0]), p=probs)
                valid = self.env.act(action, population, sample_params, master)
                reward += entropy(probs)/self.config['n_timesteps_per_trajectory'] \
                    + 1/np.sqrt(self.lookup((self.env.current), counts))/self.config['n_timesteps_per_trajectory']
                this_counts[(self.env.current)] = self.lookup((self.env.current), this_counts, base=0) + 1
                # counts[(self.env.current)] = self.lookup((self.env.current), counts) + 1/self.config['n_individuals']
            reward += self.reward(self.env.reward_params(valid))
            success = self.env.reached_target()
            self.env.reset()
            return reward, success, this_counts

    def lookup(self, k, d, base=1):
        if k in d:
            return d[k]
        else:
            return base

    def update(self, noise_samples, rewards, n_individual_target_reached):
        """
        Update function for the master parameters (weights and biases for neural network)
        Args:
            noise_samples (float array): List of the noise samples for each individual in the population
            rewards (float array): List of rewards for each individual in the population
        """
        normalized_rewards = (rewards - np.mean(rewards))
        if np.std(rewards) != 0.0:
            normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)

        self.moving_success_rate = 1./np.e * float(n_individual_target_reached) / float(self.config['n_individuals']) \
            + (1. - 1./np.e) * self.moving_success_rate
        self.learning_rate = self.config['learning_rate'] * (1 - self.moving_success_rate)
        logging.info("Learning Rate: {}".format(self.learning_rate))
        logging.info("Noise Std Dev: {}".format(self.noise_std_dev))

        self.master_params += (self.learning_rate / (self.config['n_individuals'] * self.noise_std_dev)) * np.dot(noise_samples.T, normalized_rewards)

    def run(self):
        """
        Run NES algorithm given parameters from config.
        """
        model = self.model.model()
        n_reached_target = []
        population_rewards = []
        counts = {}
        for p in range(self.config['n_populations']):
            logging.info("Population: {}\n{}".format(p+1, "="*30))
            noise_samples = np.random.randn(self.config['n_individuals'], len(self.master_params))
            rewards = np.zeros(self.config['n_individuals'])
            n_individual_target_reached = 0
            self.run_simulation(self.master_params, model, p, counts, master=True) # Run master params for progress check, not used for training
            batch_counts = {}
            for i in range(self.config['n_individuals']):
                # logging.info("Individual: {}".format(i+1))
                sample_params = self.master_params + noise_samples[i]
                rewards[i], success, this_counts = self.run_simulation(sample_params, model, p, counts)
                n_individual_target_reached += success
                # logging.info("Individual {} Reward: {}\n".format(i+1, rewards[i]))
                for k in this_counts.keys():
                    batch_counts[k] = self.lookup(k, batch_counts, base=0) + this_counts[k]
            for k in batch_counts.keys():
                counts[k] = self.lookup(k, counts) + batch_counts[k]/self.config['n_individuals']
            self.update(noise_samples, rewards, n_individual_target_reached)
            n_reached_target.append(n_individual_target_reached)
            population_rewards.append(sum(rewards)/len(rewards))
            self.plot_graphs([range(p+1), range(p+1)], [population_rewards, n_reached_target], ["Average Reward per population", "Number of times target reached per Population"], ["reward.png", "success.png"], ["line", "scatter"])
            if (p % self.config['save_every'] == 0):
                self.model.save(self.model_save_directory, "params_" + str(p) + '.py', self.master_params)
        self.env.post_processing()
        logging.info("Reached Target {} Total Times".format(sum(n_reached_target)))

    def plot_graphs(self, x_axes, y_axes, titles, filenames, types):
        for i in range(len(x_axes)):
            plt.title(titles[i])
            if types[i] == "line":
                plt.plot(x_axes[i], y_axes[i])
            if types[i] == "scatter":
                plt.scatter(x_axes[i], y_axes[i])
                plt.plot(np.unique(x_axes[i]), np.poly1d(np.polyfit(x_axes[i], y_axes[i], 1))(np.unique(x_axes[i])), 'r--')
            plt.savefig(self.training_directory + filenames[i])
            plt.clf()
