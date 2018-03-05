import numpy as np
import logging
from abstract import Environment

VALID = 0
INVALID = 1
END_GAME = 2

#################################################
# Start with score of 1000						#
# For every valid move, score -=1  				#
# Valid: Normal square							#
# Valid: Get on lava (but die)					#
# Invalid: Hit wall 							#
# Potential additions: moving goal, 			#
#################################################
class MOGame(Environment):

	def __init__(self, start_board, start_score, start_agent_loc, training_directory, config):
		Environment.__init__(self)
			self.discrete = False
		self.training_directory = training_directory
			self.config = config
		self.start_board = start_board
		self.start_score = start_score
		self.start_agent_loc = start_agent_loc
		self.height = len(start_board)
		self.width = len(start_board[0])
		self.reset()

	def reset(self):
		"""
		Reset current position to beginning.
		"""
		self.board = self.start_state
		self.score = self.start_score
		self.agent_loc = self.start_agent_loc

	def get_next_loc(self, action):
		act_probs = action
		move = None
		if self.discrete:
			move = np.argmax(act_probs)
		else:
			move = np.random.choice(np.arange(len(act_probs)), 1, act_probs)

		next_loc = self.agent_loc
		if move is 0:
			pass
		elif move is 1:
			next_loc[1] -= 1
			pass
		elif move is 2:
			next_loc[1] += 1
			pass
		elif move is 3:
			next_loc[0] += 1
			pass
		elif move is 4:
			next_loc[0] -= 1
		return move, next_loc
  
	def act(self, action, population, params, master):
		"""
		Move end effector to the given location
		(Do nothing), N, S, E, W = [0, 1, 2, 3, 4]
		# TODO: during training time do probablistic choice, during test time do deterministic choice

		"""
		if not np.linalg.norm(action): #No action selected, may not be best way to handle this
			return INVALID 

		move, next_loc = self.get_next_loc(action)
		next_x, next_y = next_loc
		invalid = False

		if not (0 <= next_x < self.width and 0 <= next_y < self.height): #Runs off of board
			invalid = True
		elif board[next_y][next_x] is '#': #Ran into wall
			invalid = True
		  
		#--------------POTENTIAL THING TO CHANGE---------------#
		if invalid:
			modified_actions = np.copy(action)
			modified_actions[move] = 0
			modified_actions = modified_actions / np.linalg.norm(modified_actions)
			return act(self, modified_actions, params, master)

		agent_loc = next_loc

		if self.board[next_y][next_x] is 'L' or self.board[next_y][next_x] is 'G':
			return END
		else:
			return VALID

	def inputs(self, timestep):
		"""
		Return the inputs for the neural network
		"""

    def encode_point(row, col):
		encoded = [0] * 5 # [Valid, Wall, Lava, Win, Agent]
		if row == self.agent_loc[0] and col == self.agent_loc[1]:
			encoded[4] = 1

		if self.board[row][col] is ' ':
			encoded[0] = 1
		elif self.board[row][col] is '#':
			encoded[1] = 1
		elif self.board[row][col] is 'L':
			encoded[2] = 1
		elif self.board[row][col] is 'G':
			encoded[3] = 1
		return encoded

	encoded = []
	for row in range(self.height):
		for col in range(self.weight):
			encoded.extend(encode_point(row, col))
	    
	return np.array(encoded)

	def reward_params(self, moved):
		"""
		Return the parameters for the proposed reward function
		"""
		# # = Wall
	    #   = Valid
	    # L = Lava
	    # G = Goal
	    pass

	def pre_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass

	def post_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		pass

	def reached_target(self):
		"""
		Check if the target goal was achieved
		"""
		pass