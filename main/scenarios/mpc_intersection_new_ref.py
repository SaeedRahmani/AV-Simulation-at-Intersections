import itertools
import math
from typing import List

import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt

# from envs.t_intersection import t_intersection
from envs.intersection import intersection, plot_intersection
from lib.car_dimensions import CarDimensions, BicycleModelDimensions
from lib.collision_avoidance import check_collision_moving_cars, get_cutoff_curve_by_position_idx
from lib.motion_primitive import load_motion_primitives
# from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.mp_search_ww_generic import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc_with_speed import MPC, MAX_ACCEL
from lib.plotting import draw_car
from lib.simulation import State, Simulation, History, HistorySimulation
from lib.trajectories import resample_curve, calc_nearest_index_in_direction
import time

def main():
    #########
    # INIT ENVIRONMENT
    #########
    ###################### Scenario Parameters #####################
    DT = 0.2
    MAX_SPEED = 30 / 3.6  # m/s
    
    mps = load_motion_primitives(version='bicycle_model')
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    start_pos = 1
    turn_indicator = 1
    scenario = intersection(start_pos=start_pos, turn_indicator=turn_indicator)
    # scenario = t_intersection(turn_left=True)

    moving_obstacles: List[MovingObstacleTIntersection] = [
        MovingObstacleTIntersection(car_dimensions, direction=1, offset=1., turning=False, speed=25 / 3.6, dt=DT),
        MovingObstacleTIntersection(car_dimensions, direction=-1, offset=4., turning=True, speed=25 / 3.6, dt=DT)
    ]
    
    #########
    # MOTION PRIMITIVE SEARCH
    #########

    start_time = time.time()
    
    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)    
    _, path, trajectory_full = search.run(debug=False)
    
    end_time = time.time()
    search_runtime = end_time - start_time
    print('search runtime is: {}'.format(search_runtime))

    #########
    # INIT MPC
    #########
    dl = np.linalg.norm(trajectory_full[0, :2] - trajectory_full[1, :2])

    cv = np.full(trajectory_full[:, 1].shape, MAX_SPEED)
    mpc = MPC(cx=trajectory_full[:, 0], cy=trajectory_full[:, 1], cv = cv, cyaw=trajectory_full[:, 2], dl=dl, dt=DT,
              car_dimensions=car_dimensions)
    state = State(x=trajectory_full[0, 0], y=trajectory_full[0, 1], yaw=trajectory_full[0, 2], v=0.0)

    simulation = HistorySimulation(car_dimensions=car_dimensions, sample_time=DT, initial_state=state)
    history = simulation.history  # gets updated automatically as simulation runs

    #########
    # SIMULATE
    #########
    TIME_HORIZON = 7.
    FRAME_WINDOW = 20
    EXTRA_CUTOFF_MARGIN = 4 * int(
        math.ceil(car_dimensions.radius / dl))  # no. of frames - corresponds approximately to car length

    traj_agent_idx = 0
    tmp_trajectory = None

    loop_runtimes = []
    
    # creating a list to store the location of obstacles through time for visulization
    obstacles_positions = [[] for _ in moving_obstacles]
    
    start_time = time.time()
    for i in itertools.count():
        loop_start_time = time.time()
        if mpc.is_goal(state):
            break

        # cutoff the trajectory by the closest future index
        # but don't do it if the trajectory is exactly one point already,
        # so that the car doesn't move slowly forwards
        if tmp_trajectory is None or np.any(tmp_trajectory[traj_agent_idx, :] != tmp_trajectory[-1, :]):
            traj_agent_idx = calc_nearest_index_in_direction(state, trajectory_full[:, 0], trajectory_full[:, 1],
                                                             start_index=traj_agent_idx, forward=True)
        trajectory_res = trajectory = trajectory_full[traj_agent_idx:]

        # compute trajectory to correspond to a car that starts from its current speed and accelerates
        # as much as it can -> this is a prediction for our own agent
        if state.v < Simulation.MAX_SPEED:
            resample_dl = np.zeros((trajectory_res.shape[0],)) + MAX_ACCEL
            resample_dl = np.cumsum(resample_dl) + state.v
            resample_dl = DT * np.minimum(resample_dl, Simulation.MAX_SPEED)
            trajectory_res = resample_curve(trajectory_res, dl=resample_dl)
        else:
            trajectory_res = resample_curve(trajectory_res, dl=DT * Simulation.MAX_SPEED)

        # predict the movement of each moving obstacle, and retrieve the predicted trajectories
        trajs_moving_obstacles = [
            np.vstack(MovingObstaclesPrediction(*o.get(), sample_time=DT, car_dimensions=car_dimensions)
                      .state_prediction(TIME_HORIZON)).T
            for o in moving_obstacles]

        # find the collision location
        collision_xy = check_collision_moving_cars(car_dimensions, trajectory_res, trajectory, trajs_moving_obstacles,
                                                   frame_window=FRAME_WINDOW)
        cutoff_idx = 999
        # cutoff the curve such that it ends right before the collision (and some margin)
        if collision_xy is not None:
            cutoff_idx = get_cutoff_curve_by_position_idx(trajectory_full, collision_xy[0],
                                                          collision_xy[1]) - EXTRA_CUTOFF_MARGIN
            cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
            # cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
            
            # tmp_trajectory = trajectory_full[:cutoff_idx]
            tmp_trajectory = trajectory_full
            cutoff_idx = cutoff_idx 
            
        else:
            cutoff_idx = 999
            tmp_trajectory = trajectory_full

        # pass the cut trajectory to the MPC
        mpc.set_trajectory_fromarray(tmp_trajectory, cutoff_idx = cutoff_idx)

        # compute the MPC
        delta, acceleration = mpc.step(state)
        
        # runtime calculation
        loop_end_time = time.time()
        loop_runtime = loop_end_time - loop_start_time
        loop_runtimes.append(loop_runtime)

        # show the computation results
        visualize_frame(DT, FRAME_WINDOW, car_dimensions, collision_xy, i, moving_obstacles, mpc, scenario, simulation,
                        state, tmp_trajectory, trajectory_res, trajs_moving_obstacles)

        # move all obstacles forward
        for i_obs, o in enumerate(moving_obstacles):
            obstacles_positions[i_obs].append((i, o.get())) # i is time here
            o.step()

        # step the simulation (i.e. move our agent forward)
        state = simulation.step(a=acceleration, delta=delta, xref_deviation=mpc.get_current_xref_deviation())

    # printing runtimes
    end_time = time.time()
    loops_total_runtime = sum(loop_runtimes)
    total_runtime = end_time - start_time
    print('total loops run time is: {}'.format(loops_total_runtime))
    print('total run time is: {}'.format(total_runtime))
    print('each mpc runtime is: {}'.format(loops_total_runtime / len(loop_runtimes)))

    # visualize final
    visualize_final(simulation.history)
    
    # ploting the trajectories and conflicts
    plot_trajectories(obstacles_positions, simulation.history)

import matplotlib.ticker as ticker

def plot_trajectories(obstacles_positions, ego_positions: History):
    # Create a new figure and get the current axes
    fig = plt.figure()
    ax = plt.gca()

    # Get colormap
    cmap = plt.cm.get_cmap('viridis')

    # Time step duration
    dt = 0.2  # seconds

    # For the ego vehicle
    times, ego_x, ego_y = ego_positions.t, ego_positions.x, ego_positions.y
    ego_positions = np.column_stack((ego_x, ego_y))
    # Convert time to seconds and normalize for color mapping
    times = np.array(times) * dt  # convert times to a numpy array and to seconds
    times_norm = times / max(times)
    # Plot each segment of the trajectory with a color corresponding to its time
    for i_time in range(1, len(times)):
        color = cmap(times_norm[i_time])
        plt.plot(ego_positions[(i_time-1):(i_time+1), 0], ego_positions[(i_time-1):(i_time+1), 1], color=color, linewidth=8)

   # For each obstacle
    for i_obstacle, obstacle_positions in enumerate(obstacles_positions):
        # Unpack the positions and times
        times, positions = zip(*obstacle_positions)
        positions = np.array(positions)
        # Convert time to seconds and normalize for color mapping
        times = np.array(times) * dt  # convert times to a numpy array and to seconds
        times_norm = times / max(times)
        # Plot each segment of the trajectory with a color corresponding to its time
        for i_time in range(1, len(times)):
            color = cmap(times_norm[i_time])
            plt.plot(positions[(i_time-1):(i_time+1), 0], positions[(i_time-1):(i_time+1), 1], color=color, linewidth=4)
            
    # Add a colorbar indicating the time progression in seconds
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(times)))
    cb = fig.colorbar(sm, ax=ax)  # add the colorbar to the current axes

    # Set colorbar ticks to every 2 seconds
    tick_locator = ticker.MultipleLocator(base=2)
    cb.locator = tick_locator
    cb.update_ticks()

    # Set colorbar label font size
    cb.set_label('Time (seconds)', size=12)
    
    # Set colorbar tick label font size
    cb.ax.tick_params(labelsize=10)

    # Set labels and title with smaller font size
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.title('Trajectories of Moving Obstacles', fontsize=12)

    # Set axis limits
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)

    # Reduce tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Show the plot
    plt.show()

def visualize_final(history: History):
    fontsize = 25

    plt.figure()
    plt.rcParams['font.size'] = fontsize
    plt.plot(history.t, np.array(history.v) * 3.6, "-r", label="speed")
    plt.grid(True)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Speed [km/h]", fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.rcParams['font.size'] = fontsize
    plt.plot(history.t, history.a, "-r", label="acceleration")
    plt.grid(True)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Acceleration [$m/s^2$]", fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.rcParams['font.size'] = fontsize
    plt.plot(history.t, history.xref_deviation, "-r", label="Deviation from reference trajectory")
    plt.grid(True)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Deviation [m]", fontsize=fontsize)
    plt.tight_layout()
    plt.show()
    
def visualize_frame(dt, frame_window, car_dimensions, collision_xy, i, moving_obstacles, mpc, scenario, simulation,
                    state, tmp_trajectory, trajectory_res, trajs_moving_obstacles):
    if i >= 0:
        plt.cla()
        plt.plot(tmp_trajectory[:, 0], tmp_trajectory[:, 1], color='b')

        if collision_xy is not None:
            plt.scatter([collision_xy[0]], [collision_xy[1]], color='r')

        plt.scatter([state.x], [state.y], color='r')
        plt.scatter([trajectory_res[0, 0]], [trajectory_res[0, 1]], color='b')

        for tr in [*trajs_moving_obstacles]:
            plt.plot(tr[:, 0], tr[:, 1], color='b')

        for obstacle in scenario.obstacles:
            obstacle.draw(plt.gca(), color='b')

        for mo in moving_obstacles:
            x, y, _, theta, _, _ = mo.get()
            draw_car((x, y, theta), car_dimensions, ax=plt.gca())

        plt.plot(simulation.history.x, simulation.history.y, '-r')

        if mpc.ox is not None:
            plt.plot(mpc.ox, mpc.oy, "+r", label="MPC")

        plt.plot(mpc.xref[0, :], mpc.xref[1, :], "+k", label="xref")

        draw_car((state.x, state.y, state.yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=plt.gca(), color='k')

        plt.title("Time: %.2f [s]" % (i * dt))
        plt.axis("equal")
        plt.grid(False)

        plt.xlim((-45, 45))
        plt.ylim((-45, 45))
        # if i == 35:
        #     time.sleep(5000)
        plt.pause(0.001)
        # plt.show()

if __name__ == '__main__':
    main()

