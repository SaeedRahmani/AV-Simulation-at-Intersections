from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import sys
sys.path.append('..')

from envs.free_area import free_area
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.helpers import measure_time
from lib.motion_primitive import load_motion_primitives
#from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.mp_search import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points

if __name__ == '__main__':
    fig, ax = plt.subplots()
    
    ##### TYPE OF ANALYSIS #####
    test_no = 2 # 1: circle; 2: forward 3: backward
    ############################
    
    #Scenario Parameters
    start_pos=0.
    goal_distance = 20
    angle = np.pi

    
    match test_no:
        case 1:   
            goal_distance = 20
            for i in np.arange(0, 1 + 1/4, 1/4): # lopp over several angles from 0 to 2*pi at 45 degree intervals
                for version in ['bicycle_model']:
                    mps = load_motion_primitives(version=version)
                    scenario = free_area(test_no, angle=i*np.pi, start_pos=start_pos, goal_distance = goal_distance)
                    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

                    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

                    draw_scenario(scenario, mps, car_dimensions, search, ax,
                                draw_obstacles=False, draw_goal=False, draw_car=True, draw_mps=False, draw_collision_checking=False,
                                draw_car2=False, draw_mps2=False, mp_name='right1')
                                        # Perform The Search:

                    @measure_time
                    def run_search():
                        return search.run(debug=True)


                    try:
                        cost, path, trajectory = run_search()

                        cx = trajectory[:, 0]
                        cy = trajectory[:, 1]
                        # cyaw = trajectory[:, 2]
                        # sp = np.ones_like(cyaw) * 8.3

                        # Draw trajectory
                        ax.plot(cx, cy, color='b')
                    except KeyboardInterrupt:
                        pass  # Break The Search On Keyboard Interrupt
        case 2:
            goal_distance = 15
            for i in (0, 1/16, -1/16, 2/16, -2/16, 3/16, -3/16): # lopp over several angles from 0 to 2*pi at 45 degree intervals
                print(i)
                for version in ['bicycle_model']:
                    mps = load_motion_primitives(version=version)
                    scenario = free_area(test_no, angle=i*np.pi, start_pos=start_pos, goal_distance = goal_distance, acceptable_error=0.1)
                    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

                    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

                    draw_scenario(scenario, mps, car_dimensions, search, ax,
                                draw_obstacles=False, draw_goal=False, draw_car=True, draw_mps=False, draw_collision_checking=False,
                                draw_car2=False, draw_mps2=False, mp_name='right1')

                    # Perform The Search:

                    @measure_time
                    def run_search():
                        return search.run(debug=True)


                    try:
                        cost, path, trajectory = run_search()

                        cx = trajectory[:, 0]
                        cy = trajectory[:, 1]
                        # cyaw = trajectory[:, 2]
                        # sp = np.ones_like(cyaw) * 8.3

                        # Draw trajectory
                        ax.plot(cx, cy, color='b')
                    except KeyboardInterrupt:
                        pass  # Break The Search On Keyboard Interrupt

            # Draw All Search Points
            # sc = draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)
            # plt.colorbar(sc)

    # ego_vehicle = mlines.Line2D([], [], color='r', marker='s', ls='', label='Ego Vehicle', markersize=marker_size)
    # moving_obs = mlines.Line2D([], [], color='c', marker='s', ls='', label='Other vehicles', markersize=marker_size)
    # goal_area = mlines.Line2D([], [], color=(1, 0.8, 0.8), marker='o', ls='', label='Goal area', markersize=marker_size)
    # trajectory = mlines.Line2D([], [], color='b', marker='_', ls='', label='Path from MP', markeredgewidth=3,
    #                            markersize=marker_size)
    # obstacles = mlines.Line2D([], [], color=(0.5, 0.5, 0.5), marker='o', ls='', label='Obstacles',
    #                           markersize=marker_size)

    match test_no:
        case 1:
            radius = 20  # Radius of the half-circle
            theta = np.linspace(0, np.pi, 100)  # angles from 0 to 180 degrees
            x = radius * np.cos(theta)  # x coordinates
            y = radius * np.sin(theta)  # y coordinates
            ax.plot(x, y, 'g-', label='Half-circle')  # Plotting the half-circle
        case 2:
            pass

    
    marker_size = 10
    trajectory = mlines.Line2D([], [], color='b', marker='', ls='-', label='Path from MP', markeredgewidth=3,
                            markersize=marker_size)
    goal_area = mlines.Line2D([], [], color=(1, 0.8, 0.8), marker='s', ls='', label='Goal area', markersize=marker_size)
    goal_direction = mlines.Line2D([], [], color='r', marker='$\u279C$', ls='', label='Goal direction',
                                markersize=marker_size)
    obstacles = mlines.Line2D([], [], color='b', marker='s', ls='', label='Obstacles', markersize=marker_size)
    forbidden = mlines.Line2D([], [], color=(0.8, 0.8, 0.8), marker='s', ls='', label='Forbidden by traffic rules',
                            markersize=marker_size)
    ego_vehicle = mlines.Line2D([], [], color='g', marker='s', ls='', label='Ego vehicle', markersize=marker_size,
                                fillstyle='none')
    mp_search = mlines.Line2D([], [], color=(47 / 255, 108 / 255, 144 / 255), marker='.', ls='',
                            label='Visited points A*', markersize=marker_size)

    plt.legend(handles=[trajectory, goal_area, goal_direction, obstacles, ego_vehicle, forbidden, mp_search])


    ax.axis('equal')
    plt.show()
