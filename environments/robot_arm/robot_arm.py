# import numpy as np
# from maddux.objects import Obstacle, Ball
# from maddux.environment import Environment
# from maddux.robots import simple_human_arm

# obstacles = [Obstacle([1, 2, 1], [2, 2.5, 1.5]),
#              Obstacle([3, 2, 1], [4, 2.5, 1.5])]
# ball = Ball([2.5, 2.5, 2.0], 0.25)

# q0 = np.array([0, 0, 0, np.pi / 2, 0, 0, 0])
# human_arm = simple_human_arm(2.0, 2.0, q0, np.array([3.0, 1.0, 0.0]))

# env = Environment(dimensions=[10.0, 10.0, 20.0],
#                   dynamic_objects=[ball],
#                   static_objects=obstacles,
#                   robot=human_arm)

# q_new = human_arm.ikine(ball.position)
# human_arm.update_angles(q_new)
# env.plot()

import numpy as np
from maddux.environment import Environment
from maddux.objects import Ball
from maddux.robots import simple_human_arm

def arm_animation():
    """Animate the arm moving to touch a ball"""

    # Declare a human arm
    q0 = np.array([0.5, 0.2, 0, 0.5, 1.5])
    human_arm = simple_human_arm(2.0, 2.0, q0, np.array([2.0, 2.0, 0.0]))
    
    # Create a ball as our target
    ball = Ball(np.array([3.0, 2.0, 3.0]), 0.15, target=True)
    
    # Create our environment
    env = Environment([5.0, 5.0, 5.0], dynamic_objects=[ball],
                      robot=human_arm)
    
    # Run inverse kinematics to find a joint config that lets arm touch ball
    human_arm.ikine(ball.position)
    # human_arm.save_path("../../ext/tutorial_path.npy")
    
    # Animate
    env.animate(duration=5.0, save_path="../../ext/tutorial.mp4")

if __name__ == '__main__':
    arm_animation()
