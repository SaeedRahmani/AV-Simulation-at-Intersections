import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt

from lib.obstacles import BoxObstacle, CircleObstaclels
from lib.scenario import Scenario

def free_area(test_no = 1, angle: float = 0.0, start_pos: float = 0.0, goal_distance = 20, acceptable_error = np.pi / 16) -> Scenario:
    allowed_goal_theta_difference = acceptable_error

    # 1: south, 2: west, 3: north, 4, east
    start_positions = (start_pos, start_pos, 0.)
    angle = angle
    test_no = test_no
    
    # 1: turn left, 2: go straight, 3: turn right
    goal_distance = goal_distance
    start = start_positions
    match test_no:
        case 1:
            print(1)
            goal = (start_pos + goal_distance*np.cos(angle), start_pos + goal_distance*np.sin(angle), angle)
        case 2:
            print(2)
            goal = (start_pos + goal_distance*np.cos(angle), start_pos + goal_distance*np.sin(angle), 0)
    
    print(goal, type(goal), goal[0], goal[1], angle)
    goal_area = BoxObstacle(xy_width=(4 * 1.8, 4), height=0.5, xy_center=(goal[0], goal[1]))
    obstacles = []

    return Scenario(
        start=start,
        goal_point=goal,
        goal_area=goal_area,
        allowed_goal_theta_difference=allowed_goal_theta_difference,
        obstacles=obstacles,
    )
    

# If you want to see how it looks like, uncomment the following and execute the py file
if __name__ == '__main__':
    scenario = free_area(test_no=1, start_pos=0, angle=0)
    
    for obstacle in scenario.obstacles:
        obstacle.draw(plt.gca(), color='b')
        
    free_area()