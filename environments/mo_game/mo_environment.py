
class MOEnvironment(object):
	def __init__(self, game, board, score, current, goal, types, flat_dim):
		self.game = game
		self.board = board
		self.score = score
		self.current = current
		self.goal = goal
		self.types = types
		self.flat_dim = flat_dim
		self.height = len(board)
		self.width = len(board[0])
