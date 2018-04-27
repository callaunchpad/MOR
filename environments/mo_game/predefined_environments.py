import pygame
import numpy as np
from time import sleep
from mo_environment import MOEnvironment

VALID = 0
INVALID = 1
GAME_OVER = 2
SUCCESS = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
DARK_YELLOW = (200, 200, 0)

def valid(board, pos):
	if not pos:
		return False
	item = board[pos[0]][pos[1]]
	if pos[0] < 0 or pos[1] < 0 or pos[1] >= len(board[0]) or pos[0] >= len(board):
		return False
	if item == '#' or item == 'L':
		return False
	return True

def move(cur, dir):
	if dir == 0:
		return cur[0] - 1, cur[1]
	if dir == 1:
		return cur[0] + 1, cur[1]
	if dir == 2:
		return cur[0], cur[1] - 1
	if dir == 3:
		return cur[0], cur[1] + 1

def backtrace(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def solution_exists(board, start):
	if (not board or len(board) == 0):
		return None
	parent = {}
	queue = [start]
	visited = []
	while queue:
		cur = queue.pop(0)
		visited.append(cur)
		if board[cur[0]][cur[1]] == 'G':
			return backtrace(parent, start, cur)
		for i in range(4):
			next = move(cur, i)
			if valid(board, next) and next not in visited:
				parent[next] = cur
				queue.append(next)
	return None

def draw_solution(board, solution_path):
	sol = np.copy(board)
	c = " "
	sol[solution_path[0][0]][solution_path[0][1]] = "A"
	for i in range(1, len(solution_path) - 1):
		sol[solution_path[i][0]][solution_path[i][1]] = "O"
	print "Solution:\n" + str(np.asmatrix(sol))

def manhattan_distance(current, target):
	if not current or not target:
		return 0
	return abs(target[0] - current[0]) + abs(target[1] - current[1])

def set_goal_start(board, width, height):
	goal, start = None, None
	while (not valid(board, goal)):
		goal = (np.random.choice(range(1, height-1)), np.random.choice(range(1, width-1)))
	t = 0
	while (not valid(board, start)) or manhattan_distance(start, goal) < (min(width, height)):
		start = (np.random.choice(range(1, height-1)), np.random.choice(range(1, width-1)))
		t += 1
		if (t == 30):
			break
	if (t == 30 and ((not valid(board, start)) or manhattan_distance(start, goal) < (min(width, height)))):
		return set_goal_start(board, width, height)
	return goal, start

def generate_test(width, height, types, probabilities):
	print "Generating Board..."
	board, start = None, None
	solution_path = None
	attempt = 0
	while (solution_path is None and attempt < 10):
		attempt += 1
		board = []
		start = (0, 0)
		for i in range(height):
			row = []
			for j in range(width):
				if i == 0 or j == 0 or i == height - 1 or j == width - 1:
					row.append('#')
				else:
					# row.append(np.random.choice(types))
					row.append(np.random.choice(types, p=probabilities))
			board.append(row)
		goal, start = set_goal_start(board, width, height)
		board[goal[0]][goal[1]] = 'G'
		solution_path = solution_exists(board, start)
	if solution_path:
		print("{} Solution exists for the given environment.\n".format('\x1b[6;30;42m' + 'Success' + '\x1b[0m'))
		draw_solution(board, solution_path)
		return board, start, goal
	else:
		return generate_test(width, height, types, probabilities)

class Player(pygame.sprite.Sprite):
	""" This class represents the bar at the bottom that the player
	controls. """

	# Constructor function
	def __init__(self, top, left, scale):
		# Call the parent's constructor
		super(Player, self).__init__()

		self.scale = scale
		# Set height, width
		self.image = pygame.Surface([1*self.scale, 1*self.scale])
		self.image.fill(BLUE)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = top
		self.rect.x = left

		self.start = (left, top)

		# Set speed vector
		self.change_x = 0
		self.change_y = 0
		self.walls = None

	def move(self, top, left):
		""" Queue the move of the player. """
		self.change_x = left*self.scale
		self.change_y = top*self.scale

	def update(self):
		""" Update the player position. """
		# Move left/right
		self.rect.x += self.change_x
		self.rect.y += self.change_y

		# Did this update cause us to hit a wall?
		# block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
		# for block in block_hit_list:
		# 	# If we are moving right, set our right side to the left side of
		# 	# the item we hit
		# 	if self.change_x > 0:
		# 		self.rect.right = block.rect.left
		# 	else:
		# 		# Otherwise if we are moving left, do the opposite.
		# 		self.rect.left = block.rect.right
		# if (len(block_hit_list) > 0):
		# 	print "Hit wall"
		# 	sleep(5)
			# self.rect.x -= self.change_x

		# Move up/down
		# self.rect.y += self.change_y

		# Check and see if we hit anything
		# block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
		# for block in block_hit_list:

		# 	# Reset our position based on the top/bottom of the object.
		# 	if self.change_y > 0:
		# 		self.rect.bottom = block.rect.top
		# 	else:
		# 		self.rect.top = block.rect.bottom
		# if (len(block_hit_list) > 0):
		# 	print "Hit wall by moving vertically"
		# 	sleep(5)
			# self.rect.y -= self.change_y

		self.move(0, 0)

	def reset(self):
		self.rect.x = self.start[0]
		self.rect.y = self.start[1]

class Wall(pygame.sprite.Sprite):
	""" Wall the player can run into. """
	def __init__(self, top, left, width, height):
		""" Constructor for the wall that the player can run into. """
		# Call the parent's constructor
		super(Wall, self).__init__()

		# Make a blue wall, of the size specified in the parameters
		self.image = pygame.Surface([width, height])
		self.image.fill(BLACK)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = top
		self.rect.x = left

class Lava(pygame.sprite.Sprite):
	""" Wall the player can run into. """
	def __init__(self, top, left, width, height):
		""" Constructor for the wall that the player can run into. """
		# Call the parent's constructor
		super(Lava, self).__init__()

		# Make a blue wall, of the size specified in the parameters
		self.image = pygame.Surface([width, height])
		self.image.fill(RED)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = top
		self.rect.x = left

class Goal(pygame.sprite.Sprite):
	""" Wall the player can run into. """
	def __init__(self, top, left, width, height):
		""" Constructor for the wall that the player can run into. """
		# Call the parent's constructor
		super(Goal, self).__init__()

		# Make a blue wall, of the size specified in the parameters
		self.image = pygame.Surface([width, height])
		self.image.fill(YELLOW)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = top
		self.rect.x = left

class Game(object):
	""" This class represents an instance of the game. If we need to
		reset the game we'd just need to create a new instance of this
		class. """

	def __init__(self, board, current, scale):
		""" Constructor. Create all our attributes and initialize
		the game. """
		self.scale = scale
		# List to hold all the sprites
		self.all_sprite_list = pygame.sprite.Group()
		 
		# Make the walls. (x_pos, y_pos, width, height)
		self.wall_list = pygame.sprite.Group()
		self.lava_list = pygame.sprite.Group()
		self.goal_list = pygame.sprite.Group()

		# Create the player object
		self.player = Player(current[0]*self.scale, current[1]*self.scale, self.scale)

		self.start = current

		self.board = board
		self.width = len(board[0])
		self.height = len(board)
		self.load_board(board, current)
		 
		self.clock = pygame.time.Clock()
		self.timesteps = 0
		self.done = False

	def reset(self):
		self.player.reset()
		self.done = False
		self.load_board(self.board, self.start)
		self.all_sprite_list.update()
		self.timesteps = 0
		# sleep(2)

	def resolve_scale(self, i, j):
		return self.scale*i, self.scale*j

	def load_board(self, board, current):
		# Flip rows and columns to be consistent with pygame matrix indices
		for i in range(len(board)):			# from top
			for j in range(len(board[0])):	# from left
				if board[i][j] == '#':
					wall = Wall(i*self.scale, j*self.scale, 1*self.scale, 1*self.scale)
					self.wall_list.add(wall)
					self.all_sprite_list.add(wall)
				elif board[i][j] == 'L':
					lava = Lava(i*self.scale, j*self.scale, 1*self.scale, 1*self.scale)
					self.lava_list.add(lava)
					self.all_sprite_list.add(lava)
				elif board[i][j] == 'G':
					goal = Goal(i*self.scale, j*self.scale, 1*self.scale, 1*self.scale)
					self.goal_list.add(goal)
					self.all_sprite_list.add(goal)

		self.player.walls = self.wall_list
		self.all_sprite_list.add(self.player)

	def process_events(self, current, action):
		""" Process all of the events. Return a "True" if we need
			to close the window. """

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				print("GAME OVER")
				self.done = True
				return self.done

		# print "self.player.rect.x: " + str(self.player.rect.x)
		# print "self.player.rect.y: " + str(self.player.rect.y)
		# print "current[0]*self.scale: " + str(current[0]*self.scale)
		# print "current[1]*self.scale: " + str(current[1]*self.scale)
		assert self.player.rect.x == current[1]*self.scale
		assert self.player.rect.y == current[0]*self.scale

		if action == 0:		# None
			self.player.move(0, 0)
		elif action == 1:	# North
			self.player.move(-1, 0)
		elif action == 2: 	# South
			self.player.move(1, 0)
		elif action == 3: 	# East
			self.player.move(0, 1)
		elif action == 4: 	# West
			self.player.move(0, -1)

		self.done = False
		return self.done

	def run_logic(self, next_loc):
		"""
		This method is run each time through the frame. It
		updates positions and checks for collisions.
		"""
		if not self.done:
			# Move all the sprites
			self.all_sprite_list.update()

			# print "self.player.rect.x: " + str(self.player.rect.x)
			# print "self.player.rect.y: " + str(self.player.rect.y)
			# print "next_loc[0]*self.scale: " + str(next_loc[1]*self.scale)
			# print "next_loc[1]*self.scale: " + str(next_loc[0]*self.scale)
			assert self.player.rect.x == next_loc[1]*self.scale
			assert self.player.rect.y == next_loc[0]*self.scale

		# See if the player block has collided with anything.
		wall_hit_list = pygame.sprite.spritecollide(self.player, self.wall_list, True)
		lava_hit_list = pygame.sprite.spritecollide(self.player, self.lava_list, True)
		goal_hit_list = pygame.sprite.spritecollide(self.player, self.goal_list, True)

		# Check the list of collisions.
		for wall in wall_hit_list:
			self.done = True
			self.status = INVALID
			return self.status

		for lava in lava_hit_list:
			self.done = True
			self.status = GAME_OVER
			return self.status

		for goal in goal_hit_list:
			self.done = True
			self.status = SUCCESS
			return self.status

		self.status = VALID
		return self.status

	def display_frame(self, screen):
		""" Display everything to the screen for the game. """
		screen.fill(WHITE)

		if self.done:
			self.all_sprite_list.draw(screen)
			# font = pygame.font.Font("Serif", 25)
			font = pygame.font.SysFont("sansserif", 30)
			text = font.render("Game Over", True, DARK_YELLOW)
			center_x = (len(self.board[0])*self.scale // 2) - (text.get_width() // 2)
			center_y = (len(self.board)*self.scale // 2) - (text.get_height() // 2)
			screen.blit(text, [center_x, center_y])
		else:
			self.all_sprite_list.draw(screen)
			font = pygame.font.SysFont("sansserif", 30)
			text = font.render(str(self.timesteps), True, WHITE)
			center_x = (self.player.image.get_width() // 2) - (text.get_width() // 2)
			center_y = (self.player.image.get_height() // 2) - (text.get_height() // 2)
			self.player.image.fill(BLUE)
			self.player.image.blit(text, [center_x, center_y])
		pygame.display.flip()

def get_easy_environment(config):
	# board = [['#','#','#','#','#','#'],
	# 		 ['#',' ','L','L','L','#'],
	# 		 ['#',' ','L','L','#','#'],
	# 		 ['#',' ',' ',' ','#','#'],
	# 		 ['#','#','L',' ','G','#'],
	# 		 ['#','#','#','#','#','#']]
	width, height = 10, 10
	scale = 500//min(width, height)
	board, current, goal = generate_test(width, height, [' ', '#', 'L'], [0.9, 0.1, 0.0])
	print "CURRENT: " + str(current)
	print "GOAL: " + str(goal)
	print(np.asmatrix(board))
	score = config['n_timesteps_per_trajectory']
	# current = (1,1)
	# goal = (5,5)

	types = [' ', '#', 'L', 'G', 'A']
	flat_dim = len(board) * len(board[0]) * len(types)

	game = Game(board, current, scale)
	assert valid(board, current)
	return MOEnvironment(game, board, score, current, goal, types, flat_dim)

def get_medium_environment(config):
	width, height = 10, 10
	scale = 500//min(width, height)
	board, current, goal = generate_test(width, height, [' ', '#', 'L'], [0.7, 0.15, 0.15])
	print "CURRENT: " + str(current)
	print "GOAL: " + str(goal)
	print(np.asmatrix(board))
	score = config['n_timesteps_per_trajectory']
	types = [' ', '#', 'L', 'G', 'A']
	flat_dim = len(board) * len(board[0]) * len(types)

	game = Game(board, current, scale)
	assert valid(board, current)
	return MOEnvironment(game, board, score, current, goal, types, flat_dim)

def get_hard_environment(config):
	width, height = 10, 10
	scale = 500//min(width, height)
	board, current, goal = generate_test(width, height, [' ', '#', 'L'], [0.6, 0.2, 0.2])
	print "CURRENT: " + str(current)
	print "GOAL: " + str(goal)
	print(np.asmatrix(board))
	score = config['n_timesteps_per_trajectory']
	types = [' ', '#', 'L', 'G', 'A']
	flat_dim = len(board) * len(board[0]) * len(types)

	game = Game(board, current, scale)
	assert valid(board, current)
	return MOEnvironment(game, board, score, current, goal, types, flat_dim)
