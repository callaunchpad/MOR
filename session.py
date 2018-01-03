import os
import sys
import logging
import warnings
import argparse
import numpy as np
import tensorflow as tf
from shutil import copyfile, rmtree
from datetime import datetime
from algorithm import NES

def config(log_file):
	logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(message)s')
	warnings.simplefilter('ignore', np.RankWarning)

def set_seeds(seed):
	np.random.seed(0)
	tf.set_random_seed(0)

def get_args():
	parser = argparse.ArgumentParser(description="Flags to toggle options")
	parser.add_argument('-d', action='store_true', default=False, help='Debug flag, Delete log after training is finished.')
	return parser.parse_args()

def create_training_contents():
	timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
	try:
		os.makedirs("ext/" + timestamp)
	except OSError as e:
		if e.errno != errno.EEXIST: raise
	training_directory = "ext/" + timestamp + "/"
	log_file = training_directory + timestamp + '.log'
	return training_directory, log_file, timestamp

if __name__ == "__main__":
	args = get_args()
	training_directory, log_file, timestamp = create_training_contents()
	config(log_file)
	copyfile("Config.yaml", training_directory + timestamp + ".yaml")

	try:
		set_seeds(0)
		algorithm = NES(training_directory)
		print("Running NES Algorithm...")
		print("Check {} for progress".format(log_file))
		algorithm.run()
	except KeyboardInterrupt:
		if args.d:
			print("\nDeleted: {}".format(training_directory))
			rmtree(training_directory)
		sys.exit(1)
	
	if args.d:
		print("\nDeleted Training Folder: {}".format(training_directory))
		rmtree(training_directory)
