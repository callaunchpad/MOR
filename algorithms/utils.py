import numpy as np

def save_array(filepath, array):
	return np.save(open(filepath, "w"), array)

def load_array(filepath):
	return np.load(filepath)