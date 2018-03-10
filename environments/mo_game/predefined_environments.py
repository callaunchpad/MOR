import pygame
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

def is_valid(board_item):
	if board_item == ' ' or board_item == 'L' or board_item == 'G':
		return True
	return False

class Player(pygame.sprite.Sprite):
	""" This class represents the bar at the bottom that the player
	controls. """

	# Constructor function
	def __init__(self, x, y):
		# Call the parent's constructor
		super(Player, self).__init__()

		# Set height, width
		self.image = pygame.Surface([15, 15])
		self.image.fill(BLUE)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = y
		self.rect.x = x

		# Set speed vector
		self.change_x = 0
		self.change_y = 0
		self.walls = None

	def move(self, x, y):
		""" Queue the move of the player. """
		self.change_x = x
		self.change_y = y

	def update(self):
		""" Update the player position. """
		# Move left/right
		self.rect.x += self.change_x

		# Did this update cause us to hit a wall?
		block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
		# for block in block_hit_list:
		# 	# If we are moving right, set our right side to the left side of
		# 	# the item we hit
		# 	if self.change_x > 0:
		# 		self.rect.right = block.rect.left
		# 	else:
		# 		# Otherwise if we are moving left, do the opposite.
		# 		self.rect.left = block.rect.right
		if len(block_hit_list > 0):
			self.rect.x -= self.change_x

		# Move up/down
		self.rect.y += self.change_y

		# Check and see if we hit anything
		block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
		# for block in block_hit_list:

		# 	# Reset our position based on the top/bottom of the object.
		# 	if self.change_y > 0:
		# 		self.rect.bottom = block.rect.top
		# 	else:
		# 		self.rect.top = block.rect.bottom
		if len(block_hit_list > 0):
			self.rect.y -= self.change_y

		self.move(0, 0)


class Wall(pygame.sprite.Sprite):
	""" Wall the player can run into. """
	def __init__(self, x, y, width, height):
		""" Constructor for the wall that the player can run into. """
		# Call the parent's constructor
		super(Wall, self).__init__()

		# Make a blue wall, of the size specified in the parameters
		self.image = pygame.Surface([width, height])
		self.image.fill(BLACK)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = y
		self.rect.x = x

class Lava(pygame.sprite.Sprite):
	""" Wall the player can run into. """
	def __init__(self, x, y, width, height):
		""" Constructor for the wall that the player can run into. """
		# Call the parent's constructor
		super(Lava, self).__init__()

		# Make a blue wall, of the size specified in the parameters
		self.image = pygame.Surface([width, height])
		self.image.fill(RED)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = y
		self.rect.x = x

class Goal(pygame.sprite.Sprite):
	""" Wall the player can run into. """
	def __init__(self, x, y, width, height):
		""" Constructor for the wall that the player can run into. """
		# Call the parent's constructor
		super(Goal, self).__init__()

		# Make a blue wall, of the size specified in the parameters
		self.image = pygame.Surface([width, height])
		self.image.fill(YELLOW)

		# Make our top-left corner the passed-in location.
		self.rect = self.image.get_rect()
		self.rect.y = y
		self.rect.x = x

class Game(object):
	""" This class represents an instance of the game. If we need to
		reset the game we'd just need to create a new instance of this
		class. """

	def __init__(self, board, current):
		""" Constructor. Create all our attributes and initialize
		the game. """
		# List to hold all the sprites
		self.all_sprite_list = pygame.sprite.Group()
		 
		# Make the walls. (x_pos, y_pos, width, height)
		self.wall_list = pygame.sprite.Group()
		self.lava_list = pygame.sprite.Group()
		self.goal_list = pygame.sprite.Group()

		# Create the player object
		self.player = Player(current[0], current[1])

		self.board = board
		self.load_board(board, current)
		 
		self.clock = pygame.time.Clock()
		 
		self.done = False

	def load_board(self, board, current):
		for i in range(len(board)):
			for j in range(len(board[0])):
				if board[i][j] == '#':
					wall = Wall(i, j, 1, 1)
					self.wall_list.add(wall)
					self.all_sprite_list.add(wall)
				elif board[i][j] == 'L':
					lava = Lava(i, j, 1, 1)
					self.lava_list.add(lava)
					self.all_sprite_list.add(lava)
				elif board[i][j] == 'G':
					goal = Goal(i, j, 1, 1)
					self.goal_list.add(goal)
					self.all_sprite_list.add(goal)

		self.player.walls = self.wall_list
		self.all_sprite_list.add(self.player)

	def process_events(self, current, move):
		""" Process all of the events. Return a "True" if we need
			to close the window. """

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				print("GAME OVER")
				self.done = True
				return self.done
	 
			# elif event.type == pygame.KEYDOWN:
			#	 if event.key == pygame.K_LEFT:
			#		 player.move(-1, 0)
			#	 elif event.key == pygame.K_RIGHT:
			#		 player.move(1, 0)
			#	 elif event.key == pygame.K_UP:
			#		 player.move(0, -1)
			#	 elif event.key == pygame.K_DOWN:
			#		 player.move(0, 1)
	 
			# elif event.type == pygame.KEYUP:
			#	 if event.key == pygame.K_LEFT:
			#		 player.move(1, 0)
			#	 elif event.key == pygame.K_RIGHT:
			#		 player.move(-1, 0)
			#	 elif event.key == pygame.K_UP:
			#		 player.move(0, 1)
			#	 elif event.key == pygame.K_DOWN:
			#		 player.move(0, -1)

		assert self.player.rect.x == current[0]
		assert self.player.rect.y == current[1]

		if action is 0:		#None
			player.move(0, 0)
		elif action is 1:	# North
			player.move(0, -1)
		elif action is 2: 	# South
			player.move(0, 1)
		elif action is 3: 	# East
			player.move(1, 0)
		elif action is 4: 	# West
			player.move(-1, 0)

		self.done = False
		return self.done

	def run_logic(self, next_loc):
		"""
		This method is run each time through the frame. It
		updates positions and checks for collisions.
		"""
		if not self.done:
			# Move all the sprites
			self.all_sprites_list.update()

			assert self.player.rect.x == next_loc[0]
			assert self.player.rect.y == next_loc[1]

			# See if the player block has collided with anything.
			lava_hit_list = pygame.sprite.spritecollide(self.player, self.lava_list, True)
			goal_hit_list = pygame.sprite.spritecollide(self.player, self.goal_list, True)

			# Check the list of collisions.
			for lava in lava_hit_list:
				self.done = True
				self.status = GAME_OVER
				return self.status

			for goal in goal_hit_list:
				self.done = True
				self.status = SUCCESS
				return self.status
		else:
			self.status = GAME_OVER
			return self.status

	def display_frame(self, screen):
		""" Display everything to the screen for the game. """
		screen.fill(WHITE)

		if self.done:
			# font = pygame.font.Font("Serif", 25)
			font = pygame.font.SysFont("serif", 25)
			text = font.render("Game Over", True, BLACK)
			center_x = (len(self.board[0]) // 2) - (text.get_width() // 2)
			center_y = (len(self.board) // 2) - (text.get_height() // 2)
			screen.blit(text, [center_x, center_y])
		else:
			self.all_sprites_list.draw(screen)
		pygame.display.flip()

def get_easy_environment():
	board = [['#','#','#','#','#','#'],
			 ['#',' ','L','L','L','#'],
			 ['#',' ','L','L','#','#'],
			 ['#',' ',' ',' ','#','#'],
			 ['#','#','L',' ','G','#'],
			 ['#','#','#','#','#','#']]
	score = 1000
	current = (1,1)
	types = [' ', '#', 'L', 'G', 'A']
	flat_dim = len(board) * len(board[0]) * len(types)

	game = Game(board, current)
	assert is_valid(board[current[0]][current[1]])
	return MOEnvironment(game, board, score, current, types, flat_dim)