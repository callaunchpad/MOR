import logging
import numpy as np
from maze.maze import Maze
import maze.predefined_environments as maze_envs
from robot_arm.robot_arm import RobotArm
import robot_arm.maddux.predefined_environments as robot_arm_envs
from mo_game.mo_game import MOGame
import mo_game.predefined_environments as mo_game_envs

def resolve_env(name):
	envs = {
		"Maze": Maze,
		"RobotArm": RobotArm,
		"MOGame": MOGame
	}
	return envs[name]

test_cases = {
	"Maze": [
				maze_envs.get_easy_environment,
				maze_envs.get_medium_environment
			],
	"RobotArm": [
				robot_arm_envs.get_empty_3dof_environment,
				robot_arm_envs.get_easy_3dof_environment
			],
	"MOGame": [
				mo_game_envs.get_easy_environment,
				mo_game_envs.get_medium_environment,
				mo_game_envs.get_hard_environment
			]
}