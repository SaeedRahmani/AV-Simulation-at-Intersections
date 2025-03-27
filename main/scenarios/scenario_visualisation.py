import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
import numpy as np

import sys
sys.path.append('..')

from envs.t_intersection import t_intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.motion_primitive import load_motion_primitives
from lib.mp_search import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.visualisation import create_animation

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(20, 20))
    mps = load_motion_primitives(version='bicycle_model')

    scenario = t_intersection()
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

    draw_scenario(scenario, mps, car_dimensions, search, ax,
                  draw_obstacles=True, draw_goal=True, draw_car=False, draw_mps=False, draw_collision_checking=False,
                  draw_car2=False, draw_mps2=False)

    # Perform The Search:
    try:
        cost, path, trajectory = search.run(debug=True)

        cx = trajectory[:, 0]
        cy = trajectory[:, 1]
        # cyaw = trajectory[:, 2]
        # sp = np.ones_like(cyaw) * 8.3

        # Draw trajectory
        ax.plot(cx, cy, color='b')
    except KeyboardInterrupt:
        pass  # Break The Search On Keyboard Interrupt

    # Draw All Search Points
    draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)

    # simulation time:
    t = 15  # s
    dt = 10e-3  # has to be the same as in the bicycle model
    length = int(t / dt)

    # Create the moving obstacles
    obstacles = [MovingObstacleTIntersection(car_dimensions, 1, True, 8.3),
                 MovingObstacleTIntersection(car_dimensions, 1, True, 8.3, offset=3),
                 MovingObstacleTIntersection(car_dimensions, -1, True, 8.3),
                 MovingObstacleTIntersection(car_dimensions, -1, True, 8.3, offset=5)]
    positions_obstacles = np.zeros((len(obstacles), length, 6))  # To log the positions of the moving obstacles
    positions_car = np.zeros((length, 3))  # x, y, theta

    # ================= The simulation loop =================
    for t in range(length):  # 5 seconds, because dt of bicycle model is 10e-3
        # Update the moving obstacles
        for i_obs, obstacle in enumerate(obstacles):
            obstacle.step()
            positions_obstacles[i_obs, t] = obstacle.get()

        # Check for collision

        # Step the MPC
        positions_car[t] = trajectory[t % len(trajectory), :3]  # Just an arbitrary value for plotting

    #  ================= Start animation =================
    animate = create_animation(ax,
                               positions_car,
                               positions_obstacles,
                               draw_car=True,
                               draw_car_center=True,
                               draw_obstacles=True,
                               draw_obstacles_center=True)

    # The interval is not working well: It seems to be the delay between frames after the calculation,
    # so it depends on the computation time
    # total should be 10ms, computation seems to take about 4ms
    ani = animation.FuncAnimation(fig, animate, frames=length, interval=5, blit=True, repeat=True)

    # Code to save the animation
    # FFwriter = animation.FFMpegWriter(fps=int(1/dt))
    # ani.save('visualisation.mp4', writer=FFwriter)
    #  ================= End animation =================

    ax.axis([-40, 40, -40, 12])
    ax.set_aspect(1)
    font_size = 20
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    plt.rcParams['font.size'] = font_size

    marker_size = 15
    ego_vehicle = mlines.Line2D([], [], color='r', marker='s', ls='', label='Ego Vehicle', markersize=marker_size)
    moving_obs = mlines.Line2D([], [], color='c', marker='s', ls='', label='Other vehicles', markersize=marker_size)
    goal_area = mlines.Line2D([], [], color=(1, 0.8, 0.8), marker='o', ls='', label='Goal area', markersize=marker_size)
    trajectory = mlines.Line2D([], [], color='b', marker='_', ls='', label='Path from MP', markeredgewidth=3, markersize=marker_size)
    obstacles = mlines.Line2D([], [], color=(0.5, 0.5, 0.5), marker='o', ls='', label='Obstacles', markersize=marker_size)

    plt.legend(handles=[ego_vehicle, moving_obs, obstacles, goal_area, trajectory])

    plt.show()
