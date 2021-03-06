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
from model.rewards import resolve_reward, resolve_multiple_rewards
from environments.env import test_cases, resolve_env

VALID = 0
INVALID = 1
GAME_OVER = 2
SUCCESS = 3

def approx_equal(a, b):
    return abs(a - b) < 1e-5

def validate_nn_outputs(output, debug=False):
    if debug:
        print output
        print "Difference: " + str(1 - np.sum(output))
    assert approx_equal(np.sum(output), 1)

class EntES():
    """
    MaxEnt reward function.
    """

    def __init__(self, training_directory, config):
        self.config = config
        self.training_directory = training_directory
        self.model_save_directory = self.training_directory + 'params/'
        self.env = resolve_env(self.config['environment'])(test_cases[self.config['environment']][self.config['environment_index']](), self.training_directory, self.config)
        self.env.pre_processing()
        self.model = resolve_model(self.config['model'])(self.config)
        self.reward = resolve_reward(self.config['reward'])
        self.MOR_flag = self.config['MOR_flag']
        if (self.MOR_flag):
            self.multiple_rewards = resolve_multiple_rewards(self.config['multiple_rewards'])
            self.reward_mins = np.zeros(len(self.multiple_rewards))
            self.reward_maxs = np.zeros(len(self.multiple_rewards))
        self.master_params = self.model.init_master_params(self.config['from_file'], self.config['params_file'])
        self.mu = self.config['n_individuals']/4
        self.learning_rate = self.config['learning_rate']
        self.noise_std_dev = self.config['noise_std_dev']
        self.visualize = self.config['visualize']
        self.visualize_every = self.config['visualize_every']
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
            status = 0
            n = self.config['n_timesteps_per_trajectory']
            for t in range(n):
                # print "T: " + str(t)
                inputs = np.array(self.env.inputs(t)).reshape((1, self.config['input_size']))
                # CORRECT: Test input into NN
                # print np.matrix(self.env.interpret_inputs(inputs.flatten()))
                net_output = sess.run(model, self.model.feed_dict(inputs, sample_params))
                # CORRECT: Test net_outputs
                validate_nn_outputs(net_output, debug=False)
                # probs = np.exp(net_output[0]) / np.sum(np.exp(net_output[0]))
                probs = net_output.flatten()
                status = self.env.act(probs, population, sample_params, master)
                reward += entropy(probs)/n + 1/np.sqrt(self.lookup((self.env.current), counts))/n
                this_counts[(self.env.current)] = self.lookup((self.env.current), this_counts, base=0) + 1
                # counts[(self.env.current)] = self.lookup((self.env.current), counts) + 1/self.config['n_individuals']
                if status != VALID:
                    break
            reward += self.reward(self.env.reward_params(status))
            success = self.env.reached_target()
            # print "RESETTING: " + str(self.env.game.timesteps)
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
        if self.MOR_flag:
            normalized_rewards = np.zeros((len(rewards), len(rewards[0])))
            for i in range(len(rewards[0])):
                reward = rewards[:,i]
                self.reward_mins[i] = min(self.reward_mins[i], min(reward))
                self.reward_maxs[i] = max(self.reward_maxs[i], max(reward))
                normalized_reward = (reward - np.mean(reward))
                if np.std(reward) != 0.0:
                    normalized_reward = (reward - np.mean(reward)) / np.std(reward)
                normalized_rewards[:,i] = normalized_reward


            top_mu = []
            pareto_front = {}
            samples_left = set(range(len(normalized_rewards)))

            while len(top_mu) + len(pareto_front.keys()) < self.mu:
                top_mu.extend([noise_samples[i] for i in pareto_front.keys()])
                pareto_front = {}
                new_keys = []
                iter_samples = list(samples_left)
                for ind in iter_samples:
                    dominated = False
                    ind_reward = normalized_rewards[ind]
                    ind_sample = noise_samples[ind]
                    comp_front = pareto_front.copy()
                    for comp in pareto_front.keys():
                        sample, reward = comp_front[comp]
                        if np.all(ind_reward <= reward) and np.any(ind_reward < reward):
                            dominated = True
                            break
                        if np.all(ind_reward >= reward) and np.any(ind_reward > reward):
                            pareto_front.pop(comp)
                            samples_left.add(comp)
                    if not dominated:
                        pareto_front[ind] = ind_reward
                        samples_left.remove(ind)


            def crowding_distance(reward, front):
                total = 0
                for i in range(len(reward)):
                    metric = reward[i]
                    comps = [value[i] for key,value in front.items()]
                    upper = self.reward_maxs[i]
                    lower = self.reward_mins[i]
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
            normalized_rewards = (rewards - np.mean(rewards))
            if np.std(rewards) != 0.0:
                normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            weighted_sum = np.dot(normalized_rewards, noise_samples)


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
        counts = {}
        for p in range(self.config['n_populations']):
            self.env.toggle_viz(True) if (self.visualize and p%self.visualize_every == 0) else self.env.toggle_viz(False)
            self.env.pre_processing()
            logging.info("Population: {}\n{}".format(p+1, "="*30))
            noise_samples = np.random.randn(self.config['n_individuals'], len(self.master_params))
            if self.MOR_flag:
                rewards = np.zeros((self.config['n_individuals'], len(self.multiple_rewards)))
            else:
                rewards = np.zeros(self.config['n_individuals'])
            n_individual_target_reached = 0
            self.run_simulation(self.master_params, model, p, counts, master=True) # Run master params for progress check, not used for training
            batch_counts = {}
            for i in range(self.config['n_individuals']):
                # logging.info("Individual: {}".format(i+1))
                sample_params = self.master_params + noise_samples[i]
                rewards[i], success, this_counts = self.run_simulation(sample_params, model, p, counts)
                n_individual_target_reached += success
                logging.info("Individual {} Reward: {}".format(i+1, rewards[i]))
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
