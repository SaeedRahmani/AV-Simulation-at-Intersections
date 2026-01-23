from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import sys
import os
sys.path.append('..')

from envs.intersection import intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.helpers import measure_time
from lib.motion_primitive import load_motion_primitives
#from lib.motion_primitive_search import MotionPrimitiveSearch
# from lib.motion_primitive_search_single_lane import MotionPrimitiveSearch
from lib.mp_search_ww_generic import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points

if __name__ == '__main__':
    fig, ax = plt.subplots()
    
    # Read parameters from environment (set by GUI) or use defaults
    start_pos = int(os.environ.get('AV_PARAM_START_POS', '3'))
    turn_indicator = int(os.environ.get('AV_PARAM_TURN_INDICATOR', '1'))
    print(f"[Config] start_pos = {start_pos}, turn_indicator = {turn_indicator}")
    
    for version in ['bicycle_model']:
        mps = load_motion_primitives(version=version)
        scenario = intersection(turn_indicator=turn_indicator, start_pos=start_pos)
        car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

        search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

        draw_scenario(scenario, mps, car_dimensions, search, ax,
                      draw_obstacles=True, draw_goal=True, draw_car=True, draw_mps=False, draw_collision_checking=False,
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
        sc = draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)
        plt.colorbar(sc)

    # ego_vehicle = mlines.Line2D([], [], color='r', marker='s', ls='', label='Ego Vehicle', markersize=marker_size)
    # moving_obs = mlines.Line2D([], [], color='c', marker='s', ls='', label='Other vehicles', markersize=marker_size)
    # goal_area = mlines.Line2D([], [], color=(1, 0.8, 0.8), marker='o', ls='', label='Goal area', markersize=marker_size)
    # trajectory = mlines.Line2D([], [], color='b', marker='_', ls='', label='Path from MP', markeredgewidth=3,
    #                            markersize=marker_size)
    # obstacles = mlines.Line2D([], [], color=(0.5, 0.5, 0.5), marker='o', ls='', label='Obstacles',
    #                           markersize=marker_size)

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
