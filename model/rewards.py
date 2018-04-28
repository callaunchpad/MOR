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
		"mo_compound": mo_compound,
		"collision": collision
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
	# max_dist = np.sqrt(2*10**2)
	dist = -np.linalg.norm(np.subtract(np.array(current), np.array(target)))
	# dist = -np.sqrt( (current[0] - target[0])**2 + (current[1] - target[1])**2 )
	# norm_dist = dist/max_dist
	# print("dist, max_dist, norm_dist:", (dist, max_dist, norm_dist))
	return dist

def collision(params):
	def is_inside(current, obstacle):
		""" 
		Returns whether current is inside obstacle
		"""
		a = obstacle.pt1
		b = obstacle.pt2
		num_total = 0
		if current[0] in range(int(math.floor(min(a[0], b[0]))), int(math.ceil(max(a[0], b[0]) + 1))):
			num_total += 1
		if current[1] in range(int(math.floor(min(a[1], b[1]))), int(math.ceil(max(a[1], b[1]) + 1))):
			num_total += 1
		if current[2] in range(int(math.floor(min(a[2], b[2]))), int(math.ceil(max(a[2], b[2]) + 1))):
			num_total += 1
		if num_total == 3:
			return True
		return False
		

	(current, obstacles) = params
	for obstacle in obstacles:
		if is_inside(current, obstacle):
			return -50
	return -1

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
