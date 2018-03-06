from mo_environment import MOEnvironment

def is_valid(board_item):
	if board_item == ' ' or board_item == 'L' or board_item == 'G':
		return True
	return False

def get_easy_environment():
	board = [['#','#','#','#','#','#'],
			 ['#',' ','L','L','L','#'],
			 ['#',' ','L','L','#','#'],
			 ['#',' ',' ',' ','#','#'],
			 ['#','#','L',' ','G','#'],
			 ['#','#','#','#','#','#']]
	score = 1000
	current = (1,1)
	goal = (4, 4)
	types = [' ', '#', 'L', 'G', 'A']
	flat_dim = len(board) * len(board[0]) * len(types)
	assert is_valid(board[current[0]][current[1]])
	return MOEnvironment(board, score, current, goal, types, flat_dim)