import pygame
from pygame.locals import *
import numpy as np
import logging
from time import sleep
import sys
from ..abstract import Environment

VALID = 0
INVALID = 1
GAME_OVER = 2
SUCCESS = 3

#################################################
# Start with score of 1000						#
# For every valid move, score -=1  				#
# Valid: Normal square							#
# Valid: Get on lava (but die)					#
# Invalid: Hit wall 							#
# Potential additions: moving goal, adversaries #
#################################################
class MOGame(Environment):

	def __init__(self, mo_env, training_directory, config):
		Environment.__init__(self)
		self.discrete = False
		self.training_directory = training_directory
		self.config = config
		self.visualize = self.config['visualize']
		if self.visualize:
			# Create an instance of the Game class
			print "CREATE GAME"
			self.game = mo_env.game
		
		self.board = mo_env.board
		self.start_score = mo_env.score
		self.start_agent_loc = mo_env.current
		self.goal = mo_env.goal
		self.status = VALID
		self.height = mo_env.height
		self.width = mo_env.width
		self.types = mo_env.types

		self.current = self.start_agent_loc
		self.score = self.start_score

		self.reset()

	def reset(self):
		"""
		Reset current position to beginning.
		"""
		self.score = self.start_score
		self.current = self.start_agent_loc
		if self.visualize:
			self.game.reset()

	def toggle_viz(self, val):
		self.visualize = val

	def get_next_loc(self, action):
		if self.discrete:
			move = np.argmax(action)
		else:
			move = np.random.choice(np.arange(len(action)), p=action)
		assert move <= 4 and move >=0

		next_loc = [0, 0]
		# print "CURRENT: " + str(self.current)
		next_loc[0], next_loc[1] = self.current[0], self.current[1]
		if move == 1:	# North
			next_loc[0] -= 1
		elif move == 2: # South
			next_loc[0] += 1
		elif move == 3: # East
			next_loc[1] += 1
		elif move == 4: # West
			next_loc[1] -= 1

		return move, tuple(next_loc)

	def act(self, action, population, params, master):
		"""
		Move end effector to the given location
		(Do nothing), N, S, E, W = [0, 1, 2, 3, 4]
		# TODO: during training time do probablistic choice, during test time do deterministic choice

		"""
		self.score -= 1
		self.game.timesteps += 1

		# if (np.sum(action) != 1): #No action selected, may not be best way to handle this
		# 	self.status = INVALID
		# 	return INVALID 

		move, next_loc = self.get_next_loc(action)
		next_x, next_y = next_loc
		invalid = False

		if not (0 <= next_x and next_x < self.width and 0 <= next_y and next_y < self.height): #Runs off of board
			invalid = True
		if self.board[next_x][next_y] == '#': #Ran into wall
			invalid = True

		#--------------POTENTIAL THING TO CHANGE---------------#
		if invalid:
			# print "DETECTED WALL"
			# print "CANNOT MOVE: " + str(move)
			# sleep(20)
			self.score += 1
			modified_actions = np.copy(action)
			modified_actions[move] = 0
			modified_actions = np.divide(modified_actions, np.sum(modified_actions))
			return self.act(modified_actions, population, params, master)

		# print "AT: " + str(self.current)
		if self.visualize and not self.game.done:
			# Process events (keystrokes, mouse clicks, etc)
			self.game.process_events(self.current, move)
			# Update object positions, check for collisions
			self.game.run_logic(next_loc)
			
			# Draw the current frame
			self.game.display_frame(self.screen)
			# Pause for the next frame
			self.game.clock.tick(60)
			# self.game.clock.tick(10)
			# sleep(0.0001)

		# print "MOVING: " + str(move)

		self.current = next_loc
		# print "NOW AT: " + str(next_loc)

		if self.board[next_x][next_y] == 'L':
			self.status = GAME_OVER
		elif self.board[next_x][next_y] == 'G':
			self.status = SUCCESS
		else:
			self.status = VALID

		if self.visualize:
			assert self.status == self.game.status

		return self.status

	def inputs(self, timestep):
		"""
		Return the inputs for the neural network
		"""
		def encode_point(row, col):
			encoded = [0] * len(self.types) # [Valid, Wall, Lava, Win, Agent]
			if row == self.current[0] and col == self.current[1]:
				encoded[4] = 1

			if self.board[row][col] == ' ':
				encoded[0] = 1
			elif self.board[row][col] == '#':
				encoded[1] = 1
			elif self.board[row][col] == 'L':
				encoded[2] = 1
			elif self.board[row][col] == 'G':
				encoded[3] = 1
			return encoded

		encoded = []
		for row in range(self.height):		# from top
			for col in range(self.width):	# from left
				encoded.extend(encode_point(row, col))

		return np.array(encoded)

	# Only used for sanity checking NN inputs
	def interpret_inputs(self, embed):
		print "TYPES:" + str(len(self.types))
		width = np.sqrt(len(embed)/len(self.types))
		split_embed = np.split(embed, len(embed)/len(self.types))
		board = []
		row = []
		for i in range(len(split_embed)):
			if (i != 0 and i%width == 0):
				board.append(row)
				row = []
			item = ' '
			if split_embed[i][1] == 1:
				item = '#'
			elif split_embed[i][2] == 1:
				item = 'L'
			elif split_embed[i][3] == 1:
				item = 'G'
			elif split_embed[i][4] == 1:
				item = 'A'
			row.append(item)
		board.append(row)
		return board

	def reward_params(self, moved):
		"""
		Return the parameters for the proposed reward function
		"""
		# # = Wall
		#   = Valid
		# L = Lava
		# G = Goal
		time_score = self.score
		died = self.status == GAME_OVER
		success = self.status == SUCCESS
		# # return [[time_score], [died], [success]]
		# # return time_score
		distance = abs(self.current[0]-self.goal[0]) + abs(self.current[1]-self.goal[1])
		# return time_score, distance, died, success
		return (self.current, self.goal), died, success
		# return died
		# return self.current, self.goal, True

	def pre_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		if self.visualize:
			# Initialize Pygame and set up the window
			pygame.init()
			# pygame.display.init()
			# pygame.font.init()
			# print "INITIALIZING"

			size = [self.game.scale*self.game.width, self.game.scale*self.game.height]
			# size = [mo_env.width, mo_env.height]
			self.screen = pygame.display.set_mode(size, DOUBLEBUF | RESIZABLE)

			pygame.display.set_caption("Multi-Objective Game")
			pygame.mouse.set_visible(False)

			# Process events (keystrokes, mouse clicks, etc)
			self.game.process_events(self.current, (0, 0))
			# Draw the current frame
			self.game.display_frame(self.screen)
			# Pause for the next frame
			self.game.clock.tick(60)
			# sleep(0.0001)

	def post_processing(self):
		"""
		Complete any pending post processing tasks
		"""
		if self.visualize:
			# Close window and exit
			# print "CLOSING WINDOWS"
			self.game.done = True
			# pygame.display.quit()
			# for i in range(10):
			pygame.quit()
			# print "DONE"
			# sleep(10)
			# sys.exit(0)
			# Process events (keystrokes, mouse clicks, etc)
			# self.game.process_events(self.current, (0, 0))
			# # Draw the current frame
			# self.game.display_frame(self.screen)

	def reached_target(self):
		"""
		Check if the target goal was achieved
		"""
		return self.status == SUCCESS
