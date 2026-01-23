import itertools
import math
from typing import List
import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt

# from envs.t_intersection import t_intersection
from envs.intersection import intersection
from lib.car_dimensions import CarDimensions, BicycleModelDimensions
from lib.collision_avoidance import check_collision_moving_cars, get_cutoff_curve_by_position_idx
from lib.motion_primitive import load_motion_primitives
# from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.mp_search_ww_generic import MotionPrimitiveSearch  # Fixed: was mp_search_multiWeight which doesn't exist
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.other_agents_prediction import OtherAgentsPrediction
from lib.mpc import MPC, MAX_ACCEL
from lib.plotting import draw_car
from lib.simulation import State, Simulation, History, HistorySimulation
from lib.trajectories import resample_curve, calc_nearest_index_in_direction
from lib.ego_instance import Ego_instance

import time

def main():
    #########
    # INIT ENVIRONMENT
    #########
    ###################### Scenario Parameters #####################
    DT = 0.2
    mps = load_motion_primitives(version='bicycle_model')
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)
    
    ### For one ego vehicle's code ###
    # start_pos = 3
    # turn_indicator = 3
    # scenario = intersection(start_pos=1, turn_indicator=1)
    # scenario = t_intersection(turn_left=True)

    # moving_obstacles: List[MovingObstacleTIntersection] = [
    #     MovingObstacleTIntersection(car_dimensions, direction=1, offset=2., turning=False, speed=25 / 3.6, dt=DT),
    #     MovingObstacleTIntersection(car_dimensions, direction=-1, offset=4., turning=False, speed=25 / 3.6, dt=DT),
    # ]
    # ego_vehicles: List[Ego_instance] =[
    #     Ego_instance(start_position=1, turn_indicator=1),
    #     Ego_instance(start_position=3, turn_indicator=2)        
    # ]
    ego_vehicles: List[Ego_instance] =[
        Ego_instance(start_position=1, turn_indicator=1)        
    ]
    #########
    # MOTION PRIMITIVE SEARCH
    #########
    # search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)    
    # _, _, trajectory_full = search.run(debug=False)
    
    start_time = time.time()  
    
    # paths = [[] for _ in ego_vehicles]
    paths = []
    for i, ego in enumerate(ego_vehicles):
        _, _, trajectory_full = MotionPrimitiveSearch(ego.scenario, car_dimensions, mps, margin=car_dimensions.radius).run(debug=False)
        #paths[i].append(trajectory_full) # paths.append(trajectory_full) Video 23:14
        paths.append(trajectory_full) # paths.append(trajectory_full) Video 23:14

    end_time = time.time()
    search_runtime = end_time - start_time
    print('search runtime is: {}'.format(search_runtime))

    #########
    # INIT MPC
    #########
    dl = np.linalg.norm(paths[0][0, :2] - paths[0][1, :2])
    
    TIME_HORIZON = 7.
    FRAME_WINDOW = 20
    EXTRA_CUTOFF_MARGIN = 4 * int(math.ceil(car_dimensions.radius / dl))  # no. of frames - corresponds approximately to car length
    
    mpc = []
    state = []
    simulation = []
    history = []
    
    traj_agent_idx = [0 for _ in ego_vehicles]
    tmp_trajectory = [None for _ in ego_vehicles]
    # tmp_trajectory = []

    loop_runtimes = []
    
    trajectory_res = [None for _ in ego_vehicles]
    trajectory = [None for _ in ego_vehicles]
    
    resample_dl = [0 for _ in ego_vehicles]
    
    collision_xy = [None for _ in ego_vehicles]
    cutoff_idx = [0 for _ in ego_vehicles]
    
    trajs_other_agents = [0 for _ in ego_vehicles]
    
    delta = [0. for _ in ego_vehicles]
    acceleration = [0. for _ in ego_vehicles]
    
    # creating a list to store the location of obstacles through time for visulization
    other_agents_pos = [None for _ in ego_vehicles]
    
    for i, ego in enumerate(ego_vehicles):
        mpc.append(MPC(cx=paths[i][:, 0], cy=paths[i][:, 1], cyaw=paths[i][:, 2], dl=dl, dt=DT,
              car_dimensions=car_dimensions))
        state.append(State(x=paths[i][0, 0], y=paths[i][0, 1], yaw=paths[i][0, 2], v=0.0))

        simulation.append(HistorySimulation(car_dimensions=car_dimensions, sample_time=DT, initial_state=state[i]))
        history.append(simulation[i].history)  # gets updated automatically as simulation runs

        #########
        # SIMULATE
        #########
    
    start_time = time.time()
    for i in itertools.count():
        loop_start_time = time.time()
        for j, ego in enumerate(ego_vehicles):
            other_agents = ego_vehicles[:j] + ego_vehicles[j+1:]
            if mpc[j].is_goal(state[j]):
                break

            # cutoff the trajectory by the closest future index
            # but don't do it if the trajectory is exactly one point already,
            # so that the car doesn't move slowly forwards
            if tmp_trajectory[j] is None or np.any(tmp_trajectory[j][traj_agent_idx[j], :] != tmp_trajectory[j][-1, :]):
                traj_agent_idx[j] = calc_nearest_index_in_direction(state[j], paths[j][:, 0], paths[j][:, 1],
                                                                start_index=traj_agent_idx[j], forward=True)
            trajectory_res[j] = trajectory[j] = paths[j][traj_agent_idx[j]:]

            # compute trajectory to correspond to a car that starts from its current speed and accelerates
            # as much as it can -> this is a prediction for our own agent
            if state[j].v < Simulation.MAX_SPEED:
                resample_dl[j] = np.zeros((trajectory_res[j].shape[0],)) + MAX_ACCEL
                resample_dl[j] = np.cumsum(resample_dl[j]) + state[j].v
                resample_dl[j] = DT * np.minimum(resample_dl[j], Simulation.MAX_SPEED)
                trajectory_res[j] = resample_curve(trajectory_res[j], dl=resample_dl[j])
            else:
                trajectory_res[j] = resample_curve(trajectory_res[j], dl=DT * Simulation.MAX_SPEED)

            # predict the movement of each moving obstacle, and retrieve the predicted trajectories
            # Inputs: (self, x, y, v, yaw, a, steering_angle, sample_time: float, car_dimensions: CarDimensions)
            # trajs_other_agents[j] = [
            #     np.vstack(OtherAgentsPrediction(
            #         state[k].x,
            #         state[k].y,
            #         state[k].yaw,
            #         state[k].v,
            #         steering_angle=0, # This is not correct. But because we don't have access to the steering angle of ego I used zero steerig angles
            #         sample_time=DT,
            #         car_dimensions=car_dimensions).state_prediction(TIME_HORIZON)).T
            #     # for k, ego in other_agents]
            #     for k, ego in enumerate(other_agents)]
            for k, ego in enumerate(ego_vehicles):
                if k != j:
                    trajs_other_agents[j] = [
                        np.vstack(OtherAgentsPrediction(
                        state[k].x,
                        state[k].y,
                        state[k].yaw,
                        state[k].v,
                        steering_angle=0, # This is not correct. But because we don't have access to the steering angle of ego I used zero steerig angles
                        sample_time=DT,
                        car_dimensions=car_dimensions).state_prediction(TIME_HORIZON)).T]

            # find the collision location
            collision_xy[j] = check_collision_moving_cars(car_dimensions, trajectory_res[j], trajectory[j], trajs_other_agents[j],
                                                   frame_window=FRAME_WINDOW)

        # cutoff the curve such that it ends right before the collision (and some margin)
            if collision_xy[j] is not None:
                cutoff_idx[j] = get_cutoff_curve_by_position_idx(paths[j], collision_xy[j][0],
                                                            collision_xy[j][1]) - EXTRA_CUTOFF_MARGIN
                cutoff_idx[j] = max(traj_agent_idx[j] + 1, cutoff_idx[j])
                # cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
                tmp_trajectory[j] = paths[j][:cutoff_idx[j]]
            else:
                tmp_trajectory[j] = paths[j]

            # pass the cut trajectory to the MPC
            mpc[j].set_trajectory_fromarray(tmp_trajectory[j])

            # compute the MPC
            delta[j], acceleration[j] = mpc[j].step(state[j])
            
            # runtime calculation
            loop_end_time = time.time()
            loop_runtime = loop_end_time - loop_start_time
            loop_runtimes.append(loop_runtime)

            # show the computation results
        ########## VISUALIZE #############
        # visualize_frame(DT, FRAME_WINDOW, car_dimensions, collision_xy, i, other_agents, mpc, ego_vehicles[0].scenario , simulation,
        #                state, tmp_trajectory, trajectory_res, trajs_other_agents)

        # move all obstacles forward
        # for k, o in enumerate(ego_vehicles):
        #     if k != j:
        #         other_agents_pos[j] = ((i,(state[k].x, state[k].y))) # i is time here
        #         o.step()

        # step the simulation (i.e. move our agent forward)
        for i_ego, ego in enumerate(ego_vehicles):
            state[i_ego] = simulation[i_ego].step(a=acceleration[i_ego], delta=delta[i_ego], xref_deviation=mpc[i_ego].get_current_xref_deviation())

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
    plot_trajectories(other_agents_pos, simulation.history)

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
        plt.plot(ego_positions[(i_time-1):(i_time+1), 0], ego_positions[(i_time-1):(i_time+1), 1], color=color, linewidth=6)

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
            plt.plot(positions[(i_time-1):(i_time+1), 0], positions[(i_time-1):(i_time+1), 1], color=color, linewidth=3)
            
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
    plt.ylim(-20, 10)

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

