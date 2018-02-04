import logging
import numpy as np
from maze import Maze
from robot_arm.robot_arm import RobotArm
import robot_arm.maddux.predefined_environments as robot_arm_envs

def resolve_env(name):
	envs = {
		"Maze": Maze,
		"RobotArm": RobotArm
	}
	return envs[name]

test_cases = {
	"Maze": [
				[[3,0,0,0],
				 [1,0,0,1],
				 [1,1,0,1],
				 [1,1,0,4]],
				[[3,0,0,0,0,0,0,0,1,1],
				 [1,0,0,0,1,0,1,0,1,1],
				 [1,1,1,0,1,0,1,0,1,1],
				 [1,1,1,0,0,0,0,0,1,1],
				 [1,1,0,1,1,0,0,1,1,1],
				 [1,1,0,1,1,0,0,0,1,1],
				 [1,0,0,0,0,0,1,0,1,1],
				 [1,1,1,0,1,0,1,0,1,1],
				 [1,1,1,0,0,0,4,0,1,1],
				 [1,1,1,1,1,1,0,0,1,1]]
			],
	"RobotArm": [
				robot_arm_envs.get_empty_3dof_environment(),
				robot_arm_envs.get_easy_3dof_environment()
			]
}