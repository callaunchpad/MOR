import numpy as np
import logging
from abstract import Environment

class Maze(Environment):

	def __init__(self, matrix, training_directory):
		self.discrete = True
		self.training_directory = training_directory
		self.walls = []
		self.empty = []
		self.start = (0,0)
		self.target = (0,0)
		self.boundaries = (len(matrix),len(matrix[0]))
		self.map = matrix
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				if matrix[i][j] == 3:
					self.start = (i, j)
				if matrix[i][j] == 4:
					self.target = (i, j)
				if matrix[i][j] == 1:
					self.walls.append((i, j))
				else:
					self.empty.append((i, j))
		self.current = self.start
		logging.info("Maze: Target at {}".format(self.target))
		logging.info(np.asmatrix(self.map))

	def solution_exists(self):
		"""
		BFS to check if solution exists for the given maze.
		Returns:
			(bool): True if path from start to target exists, else False.
		"""
		print(np.asmatrix(self.map))
		queue = [self.start]
		visited = []
		while queue:
			x, y = queue.pop(0)
			visited.append((x, y))
			if self.map_location(x, y) == 4:
				return True
			for i in range(4):
				next = self.next((x, y), i)
				if not self.is_wall((x, y), i) and next not in visited:
					queue.append(next)
		return False

	def reset(self):
		"""
		Reset current position to beginning.
		"""
		self.current = self.start

	def direction_name(self, n):
		"""
		Map int directions to their corresponding strings for printing
		Args:
		    n (int): Int corresponding to a direction.
		Returns:
			(string): String form of direction.
		"""
		if n == 0: return "^"
		if n == 1: return "v"
		if n == 2: return "<"
		if n == 3: return ">"

	def map_location(self, x, y):
		"""
		Given expected map location by human perception.
		Args:
		    x (int): Row coordinate on map
		    y (int): Column coordinate on map
		Returns:
			(int): Item in map at the expected location
		"""
		return self.map[y][x]

	def next(self, current_position, direction):
		"""
		Position reached if moved in direction from current_position
		Args:
		    current_position (int tuple): x, y coordinates of a position
		    direction (int): Direction int
		Returns:
			(tuple): Destination coordinates after moving in direction from current_position
		"""
		if direction == 0:
			y_next = current_position[1] - 1
			if y_next < 0 or self.is_wall(current_position, 0):
				y_next = current_position[1]
			return (current_position[0], y_next)
		if direction == 1:
			y_next = current_position[1] + 1
			if y_next >= self.boundaries[1] or self.is_wall(current_position, 1):
				y_next = current_position[1]
			return (current_position[0], y_next)
		if direction == 2:
			x_next = current_position[0] - 1
			if x_next < 0 or self.is_wall(current_position, 2):
				x_next = current_position[0]
			return (x_next, current_position[1])
		if direction == 3:
			x_next = current_position[0] + 1
			if x_next >= self.boundaries[0] or self.is_wall(current_position, 3):
				x_next = current_position[0]
			return (x_next, current_position[1])

	def act(self, direction, population, master):
		"""
		Move current position coordinate in the given direction.
		"""
		success = True
		past = self.current
		self.current = self.next(self.current, direction)
		logging.info("Action: {}, Current Position: {}".format(self.direction_name(direction), self.current))
		return success

	def is_wall(self, current_position, direction):
		"""
		Check if the object in direction from current_position is a wall
		Args:
		    current_position (int tuple): x, y coordinates of a position
		    direction (int): Direction int
		Returns:
			(int): 1 if object in direction from current_position is a wall, else 0
		"""
		if direction == 0:
			y_next = current_position[1] - 1
			if y_next < 0 or self.map_location(current_position[0], y_next) == 1:
				return 1
			return 0
		if direction == 1:
			y_next = current_position[1] + 1
			if y_next >= self.boundaries[1] or self.map_location(current_position[0], y_next) == 1:
				return 1
			return 0
		if direction == 2:
			x_next = current_position[0] - 1
			if x_next < 0 or self.map_location(x_next, current_position[1]) == 1:
				return 1
			return 0
		if direction == 3:
			x_next = current_position[0] + 1
			if x_next >= self.boundaries[0] or self.map_location(x_next, current_position[1]) == 1:
				return 1
			return 0

	def inputs(self, t):
		"""
		Return the inputs for the neural network
		"""
		inputs = [self.current[0], self.current[1], self.target[0], self.target[1], self.is_wall(self.current, 0), self.is_wall(self.current, 1), self.is_wall(self.current, 2), self.is_wall(self.current, 3), t+1]
		return inputs

	def reward_params(self, success):
		"""
		Return the parameters for the proposed reward function
		"""
		params = [self.current, self.target, success]
		return params

	def pre_processing(self):
		"""
		Complete any pre processing tasks
		"""
		if not self.solution_exists():
			raise Exception("No solution exists for the given environment.")
		else:
			print("{} Solution exists for the given environment.\n".format('\x1b[6;30;42m' + 'Success' + '\x1b[0m'))

	def post_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass

