import math
import numpy as np

def resolve_reward(name):
	rewards = {
		"manhattan_distance": manhattan_distance,
		"euclidean_distance": euclidean_distance,
		"binary": binary,
		"mo_time_score": mo_time_score,
		"mo_death": mo_death,
		"mo_success": mo_success,
		"mo_compound": mo_compound
	}
	return rewards[name]
# def resolve_multiple_rewards(names):
# 	functions = names.split(",")
# 	for i in range(len(functions)):
# 		functions[i] = resolve_reward(functions[i].strip())
# 	print("functions:", functions)
# 	return functions

def resolve_multiple_rewards(names):
	functions = names.split(",")
	for i in range(len(functions)):
		functions[i] = resolve_reward(functions[i].strip())
	# print("functions:", functions)
	return functions

def manhattan_distance(params):
	"""
	Manhattan distance from current position to target
	Args:
	    current (tuple): x, y coordinates of the current position
	    target (tuple): x, y coordinates of the target
	Returns:
		(float): Manhattan distance from current position to target
	"""
	current, target, solution = params
	if not solution:
		return -100
	dist = abs(target[0] - current[0]) + abs(target[1] - current[1])
	target_reached = dist == 0
	return -dist + (100 * target_reached)

def euclidean_distance(params):
	current, target = params
	# if not solution:
	# 	return -100
	max_dist = np.sqrt(2*10**2)
	# dist = -np.linalg.norm(np.subtract(np.array(current), np.array(target)))
	dist = -np.sqrt( (current[0] - target[0])**2 + (current[1] - target[1])**2 )
	norm_dist = dist/max_dist
	# print("dist, max_dist, norm_dist:", (dist, max_dist, norm_dist))
	return norm_dist

def binary(params):
	current, target, solution = params
	if not solution:
		return -100
	if list(current) == list(target):
		return 1
	return -1

def mo_compound(params):
	time_score, distance, died, success = params
	# return (-distance*time_score*(1-died*-1) + 7000)*.001
	max_dist = np.sqrt(2*10**2)
	max_time = 40
	# print([-distance/max_dist, time_score/max_time, (1-died*-1)])
	return -distance/max_dist + time_score/max_time + (1-died*-1)

def mo_time_score(params):
	score = params
	return score

def mo_death(params):
	died = params
	return died*-1

def mo_success(params):
	success = params
	return success*2