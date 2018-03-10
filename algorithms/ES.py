import sys
import inspect
import logging
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.models import resolve_model
from model.rewards import resolve_reward, resolve_multiple_rewards
from environments.env import test_cases, resolve_env


class ES():
    """
    Implementation of ES algorithm by OpenAI: https://arxiv.org/pdf/1703.03864.pdf
    """

    def __init__(self, training_directory, config):
        self.config = config
        self.training_directory = training_directory
        self.model_save_directory = self.training_directory + 'params/'
        self.env = resolve_env(self.config['environment'])(test_cases[self.config['environment']][self.config['environment_index']], self.training_directory, self.config)
        self.env.pre_processing()
        self.model = resolve_model(self.config['model'])(self.config)
        self.reward = resolve_reward(self.config['reward'])
        self.mu = self.config['mu']
        self.MOR_flag = self.config['MOR_flag']
        if (self.MOR_flag):
            self.multiple_rewards = resolve_multiple_rewards(self.config['multiple_rewards'])
        self.multiple_rewards = resolve_multiple_rewards(self.config['multiple_rewards'])
        self.master_params = self.model.init_master_params(self.config['from_file'], self.config['params_file'])
        self.learning_rate = self.config['learning_rate']
        self.noise_std_dev = self.config['noise_std_dev']
        self.moving_success_rate = 0
        if (self.config['from_file']):
            logging.info("\nLoaded Master Params from:")
            logging.info(self.config['params_file'])
        logging.info("\nReward:")
        logging.info(inspect.getsource(self.reward) + "\n")

    def run_simulation(self, sample_params, model, population, master=False):
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
            valid = False
            for t in range(self.config['n_timesteps_per_trajectory']):
                inputs = np.array(self.env.inputs(t)).reshape((1, self.config['input_size']))
                net_output = sess.run(model, self.model.feed_dict(inputs, sample_params))
                action = net_output
                if self.env.discrete:
                    action = np.argmax(net_output)
                valid = self.env.act(action, population, sample_params, master)
            reward += self.reward(self.env.reward_params(valid))
            success = self.env.reached_target()
            self.env.reset()
            return reward, success

    def update(self, noise_samples, rewards, n_individual_target_reached):
        """
        Update function for the master parameters (weights and biases for neural network)
        Args:
            noise_samples (float array): List of the noise samples for each individual in the population
            rewards (float array): List of rewards for each individual in the population
        """
        if self.multi:
            normalized_rewards = np.array(len(rewards), len(rewards[i]))
            for i in range(len(rewards[0])):
                reward = rewards[:,i]
                normalized_reward = (reward - np.mean(reward))
                if np.std(reward) != 0.0:
                    normalized_reward = (reward - np.mean(reward)) / np.std(reward)
                normalized_rewards[:,reward] = normalized_reward
            

            top_mu = []
            pareto_front = {}

            while len(top_mu) + len(pareto_front.keys()) < self.mu:
                top_mu.extend(pareto_front.keys())
                pareto_front = {}
                for ind in range(len(normalized_rewards)):
                    ind_reward = normalized_rewards[ind]
                    ind_sample = noise_samples[ind]
                    for sample, reward in pareto_front.items():
                        if np.all(ind_reward <= reward) and np.any(ind_reward < reward):
                            break
                        if np.all(ind_reward >= reward) and np.any(ind_reward > reward):
                            pareto_front.remove(sample)
                            pareto_front[ind_sample] = ind_reward
                            break

            def crowding_distance(reward, front):
                total = 0
                for i in range(len(reward))):
                    metric = reward[i]
                    comps = [value for key,value in front.items()]
                    upper = max(comps)
                    lower = min(comps)
                    if metric == lower or metric == upper:
                        return -1
                    else:
                        for comp in comps:
                            if comp > metric:
                                upper = min(upper, comp)
                            elif comp < metric:
                                lower = max(lower, comp)
                        total += upper - lower
                return total

            tie_break = np.asarray([(sample, crowding_distance(reward, front)) for sample,reward in front.items()])
            top_mu.extend(i[0] for i in tie_break[:self.mu - len(top_mu)])
            weighted_sum = sum(top_mu)

        else:
            if np.std(rewards) != 0.0:
                normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            weighted_sum = np.dot(noise_samples, normalized_rewards)

        self.moving_success_rate = 1./np.e * float(n_individual_target_reached) / float(self.config['n_individuals']) \
            + (1. - 1./np.e) * self.moving_success_rate
        self.learning_rate = self.config['learning_rate'] * (1 - self.moving_success_rate)
        logging.info("Learning Rate: {}".format(self.learning_rate))
        logging.info("Noise Std Dev: {}".format(self.noise_std_dev))
        self.master_params += (self.learning_rate / (self.config['n_individuals'] * self.noise_std_dev)) * weighted_sum

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
            self.run_simulation(self.master_params, model, p, master=True) # Run master params for progress check, not used for training
            for i in range(self.config['n_individuals']):
                logging.info("Individual: {}".format(i+1))
                sample_params = self.master_params + noise_samples[i]
                rewards[i], success = self.run_simulation(sample_params, model, p)
                n_individual_target_reached += success
                logging.info("Individual {} Reward: {}\n".format(i+1, rewards[i]))
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
