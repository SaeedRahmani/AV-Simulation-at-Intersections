import json
import itertools
import math
import os
from typing import List
import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
# plt.rcParams['text.usetex'] = True

# from envs.t_intersection import t_intersection
from envs.intersection import intersection, plot_intersection
from lib.car_dimensions import CarDimensions, BicycleModelDimensions
from lib.collision_avoidance import check_collision_moving_cars, get_cutoff_curve_by_position_idx
from lib.motion_primitive import load_motion_primitives
# from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.mp_search import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc_sensitivity import MPC, MAX_ACCEL
# from lib.mpc_jerk import MPC, MAX_ACCEL
from lib.plotting import draw_car
from lib.simulation import State, Simulation, History, HistorySimulation
from lib.trajectories import resample_curve, calc_nearest_index_in_direction
import time

def reset_config(config_file_path: str):
    # Define the default configuration
    default_config = {
        "NX": 4,
        "NU": 2,
        "T": 13,
        "w_perp": 20.0,
        "w_para": 1.0,
        "R": [0.1, 0.01],
        "Rd": [10, 1.0],
        "Q_v_yaw": [0.0, 0.5],
        "Qf": [1.0, 1.0, 0.0, 0.5],
        "GOAL_DIS": 1.5,
        "STOP_SPEED": 0.1389,
        "MAX_TIME": 13.0,
        "MAX_ITER": 1,
        "DU_TH": 0.1,
        "MAX_DSTEER": 30.0,
        "MAX_ACCEL": 2.0,
        "MAX_DECEL": -10
    }
    # Old configuration
    # {
    #     "NX": 4,
    #     "NU": 2,
    #     "T": 13,
    #     "w_perp": 20.0,
    #     "w_para": 1.0,
    #     "R": [0.1, 0.01],
    #     "Rd": [10, 1.0],
    #     "Q_v_yaw": [0.0, 0.5],
    #     "Qf": [1.0, 1.0, 0.0, 0.5],
    #     "GOAL_DIS": 1.5,
    #     "STOP_SPEED": 0.1389,
    #     "MAX_TIME": 13.0,
    #     "MAX_ITER": 1,
    #     "DU_TH": 0.1,
    #     "MAX_DSTEER": 30.0,
    #     "MAX_ACCEL": 2.0,
    #     "MAX_DECEL": -10
    # }

    # Read the existing configuration file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Overwrite with default configuration
    config.update(default_config)

    # Save the updated configuration back to the file
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=4)


def main():
    #########
    # INIT ENVIRONMENT
    #########    
    
    # Reseting the configuration file
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(_script_dir, '..', 'config', 'mpc_config_sensitivity.json')
    reset_config(config_file_path)
    # Loading the configuration file
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    # parameter_in_study_name = '$w_{{\\parallel}}$' # NOTE: for plots
    
    # Parameters:
    # parameter_in_study_name = '$w_{{\\perp}}$' # NOTE: for plots
    # parameter_name = 'w_perp' # NOTE: Should match the key in the config file
    # parameter_in_study_name_save = 'w_perp' # NOTE: for saving the plots
    # parameter_in_study = [0, 1, 10, 20, 50] # NOTE: Change 'Parameter' in line 88: mpc = MPC in order to specify the parameter to be studied
    
    # parameter_in_study_name = '$w_{{\\parallel}}$' # NOTE: for plots
    # parameter_name = 'w_para' # NOTE: Should match the key in the config file
    # parameter_in_study_name_save = 'w_para' # NOTE: for saving the plots
    # parameter_in_study = [0, 0.1, 1, 5, 10] # NOTE: Change 'Parameter' in line 88: mpc = MPC in order to specify the parameter to be studied
    
    # parameter_in_study_name = '$R_{{acc}}$' # NOTE: for plots
    # parameter_name = 'R' # NOTE: Should match the key in the config file
    # parameter_in_study_name_save = 'R_acc' # NOTE: for saving the plots
    # parameter_in_study = [[0, 0.01],[0.01, 0.01],[0.1, 0.01],[1, 0.01],[10, 0.01]] # NOTE: Change 'Parameter' in line 88: mpc = MPC in order to specify the parameter to be studied
    
    # parameter_in_study_name = '$R_{{steer}}$' # NOTE: for plots
    # parameter_name = 'R' # NOTE: Should match the key in the config file
    # parameter_in_study_name_save = 'R_steer' # NOTE: for saving the plots
    # parameter_in_study = [[0.1, 0.0],[0.1, 0.01],[0.1, 0.1],[0.1, 1],[0.1, 10]] # NOTE: Change 'Parameter' in line 88: mpc = MPC in order to specify the parameter to be studied
    
    # parameter_in_study_name = '$Rd_{{acc}}$' # NOTE: for plots
    # parameter_name = 'Rd' # NOTE: Should match the key in the config file
    # parameter_in_study_name_save = 'Rd_acc' # NOTE: for saving the plots
    # parameter_in_study = [[0, 1.0],[1, 1.0],[5, 1.0],[10, 1.0],[20, 1.0]] # NOTE: Change 'Parameter' in line 88: mpc = MPC in order to specify the parameter to be studied
    
    parameter_in_study_name = '$Rd_{{steer}}$' # NOTE: for plots
    parameter_name = 'Rd' # NOTE: Should match the key in the config file
    parameter_in_study_name_save = 'Rd_steer' # NOTE: for saving the plots
    parameter_in_study = [[10, 0],[10, 0.01],[10, 0.1],[10, 1],[10, 10]] # NOTE: Change 'Parameter' in line 88: mpc = MPC in order to specify the parameter to be studied
    

    ###################### Scenario Parameters #####################
    DT = 0.2
    mps = load_motion_primitives(version='bicycle_model')
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    start_pos = 1
    turn_indicator = 1
    scenario = intersection(start_pos=start_pos, turn_indicator=turn_indicator)
    other_vehicles = False

    if other_vehicles:
        moving_obstacles: List[MovingObstacleTIntersection] = [
            MovingObstacleTIntersection(car_dimensions, direction=1, offset=2., turning=False, speed=25 / 3.6, dt=DT),
            MovingObstacleTIntersection(car_dimensions, direction=-1, offset=4., turning=True, speed=25 / 3.6, dt=DT)
        ]
    else:
        moving_obstacles: List[MovingObstacleTIntersection] = []

    #########
    # MOTION PRIMITIVE SEARCH
    #########

    start_time = time.time()
    
    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)    
    _, _, trajectory_full = search.run(debug=False)
    
    end_time = time.time()
    search_runtime = end_time - start_time
    print('search runtime is: {}'.format(search_runtime))

    #########
    # INIT MPC
    #########
    dl = np.linalg.norm(trajectory_full[0, :2] - trajectory_full[1, :2])

    histories = []
    labels = []
    mpcs = []
    states = []
    simulations = []
    tmp_trajectories = []
    trajectory_ress = []
    trajs_moving_obstacles_list = []
    obstacles_positions_list = []
    total_runtimes = []

    for parameter in parameter_in_study:
        config[parameter_name] = parameter
        
        # Save the updated configuration
        with open(config_file_path, 'w') as f:
            json.dump(config, f, indent=4)    
        
        mpc = MPC(cx=trajectory_full[:, 0], cy=trajectory_full[:, 1], cyaw=trajectory_full[:, 2], dl=dl, dt=DT,
                  car_dimensions=car_dimensions)
        state = State(x=trajectory_full[0, 0], y=trajectory_full[0, 1], yaw=trajectory_full[0, 2], v=0.0)
        simulation = HistorySimulation(car_dimensions=car_dimensions, sample_time=DT, initial_state=state)
        history = simulation.history

        #########
        # SIMULATE
        #########
        TIME_HORIZON = 7.
        FRAME_WINDOW = 20
        EXTRA_CUTOFF_MARGIN = 4 * int(math.ceil(car_dimensions.radius / dl))  # no. of frames - corresponds approximately to car length

        traj_agent_idx = 0
        tmp_trajectory = None
        loop_runtimes = []

        obstacles_positions = [[] for _ in moving_obstacles]

        start_time = time.time()
        for i in itertools.count():
            loop_start_time = time.time()
            if mpc.is_goal(state):
                break

            if tmp_trajectory is None or np.any(tmp_trajectory[traj_agent_idx, :] != tmp_trajectory[-1, :]):
                traj_agent_idx = calc_nearest_index_in_direction(state, trajectory_full[:, 0], trajectory_full[:, 1],
                                                                start_index=traj_agent_idx, forward=True)
            trajectory_res = trajectory = trajectory_full[traj_agent_idx:]

            if state.v < Simulation.MAX_SPEED:
                resample_dl = np.zeros((trajectory_res.shape[0],)) + MAX_ACCEL
                resample_dl = np.cumsum(resample_dl) + state.v
                resample_dl = DT * np.minimum(resample_dl, Simulation.MAX_SPEED)
                trajectory_res = resample_curve(trajectory_res, dl=resample_dl)
            else:
                trajectory_res = resample_curve(trajectory_res, dl=DT * Simulation.MAX_SPEED)

            trajs_moving_obstacles = [
                np.vstack(MovingObstaclesPrediction(*o.get(), sample_time=DT, car_dimensions=car_dimensions)
                          .state_prediction(TIME_HORIZON)).T
                for o in moving_obstacles]

            collision_xy = check_collision_moving_cars(car_dimensions, trajectory_res, trajectory, trajs_moving_obstacles,
                                                       frame_window=FRAME_WINDOW)

            if collision_xy is not None:
                cutoff_idx = get_cutoff_curve_by_position_idx(trajectory_full, collision_xy[0],
                                                              collision_xy[1]) - EXTRA_CUTOFF_MARGIN
                cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
                tmp_trajectory = trajectory_full[:cutoff_idx]
            else:
                tmp_trajectory = trajectory_full

            mpc.set_trajectory_fromarray(tmp_trajectory)

            delta, acceleration = mpc.step(state)
            loop_end_time = time.time()
            loop_runtime = loop_end_time - loop_start_time
            loop_runtimes.append(loop_runtime)

            for i_obs, o in enumerate(moving_obstacles):
                obstacles_positions[i_obs].append((i, o.get()))
                o.step()

            state = simulation.step(a=acceleration, delta=delta, xref_deviation=mpc.get_current_xref_deviation())
            if mpc.is_goal(state):
                visualize_frame(DT, FRAME_WINDOW, car_dimensions, collision_xy, i, moving_obstacles, [mpc], [scenario], [simulation],
                                [state], [tmp_trajectory], [trajectory_res], [trajs_moving_obstacles])
        
        end_time = time.time()
        loops_total_runtime = sum(loop_runtimes)
        total_runtime = end_time - start_time
        total_runtimes.append(total_runtime)
        print('total loops run time is: {}'.format(loops_total_runtime))
        print('total run time is: {}'.format(total_runtime))
        print('each mpc runtime is: {}'.format(loops_total_runtime / len(loop_runtimes)))

        histories.append(simulation.history)
        labels.append(f'{parameter_in_study_name}={parameter}')
        obstacles_positions_list.append(obstacles_positions)

    # visualize final results for all runs
    visualize_final(histories, labels, parameter_in_study_name_save)

    # plot the trajectories and conflicts for all runs
    plot_trajectories(obstacles_positions_list, histories)
    plot_trajectories_comparison(obstacles_positions_list, histories, trajectory_full, parameter_in_study, parameter_in_study_name, parameter_in_study_name_save, total_runtimes)


def plot_trajectories(obstacles_positions_list: List[List[List[tuple]]], ego_positions_list: List[History]):
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size here
    font_size = 14
    cmap = plt.cm.get_cmap('viridis')

    dt = 0.2  # seconds

    for idx, (ego_positions, obstacles_positions) in enumerate(zip(ego_positions_list, obstacles_positions_list)):
        times, ego_x, ego_y = ego_positions.t, ego_positions.x, ego_positions.y
        ego_positions_arr = np.column_stack((ego_x, ego_y))
        times = np.array(times) * dt  # convert times to a numpy array and to seconds
        times_norm = times / max(times)
        for i_time in range(1, len(times)):
            color = cmap(times_norm[i_time])
            ax.plot(ego_positions_arr[(i_time-1):(i_time+1), 0], ego_positions_arr[(i_time-1):(i_time+1), 1], color=color, linewidth=2, label=f'Ego Trajectory {idx + 1}' if i_time == 1 else "")

        for i_obstacle, obstacle_positions in enumerate(obstacles_positions):
            times, positions = zip(*obstacle_positions)
            positions = np.array(positions)
            times = np.array(times) * dt  # convert times to a numpy array and to seconds
            times_norm = times / max(times)
            for i_time in range(1, len(times)):
                color = cmap(times_norm[i_time])
                ax.plot(positions[(i_time-1):(i_time+1), 0], positions[(i_time-1):(i_time+1), 1], color=color, linewidth=1, label=f'Obstacle {i_obstacle + 1}' if i_time == 1 else "")
            
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(times)))
    cb = fig.colorbar(sm, ax=ax)

    tick_locator = ticker.MultipleLocator(base=2)
    cb.locator = tick_locator
    cb.update_ticks()

    cb.set_label('Time (seconds)', size=font_size)
    cb.ax.tick_params(labelsize=font_size)

    ax.set_xlabel('X', fontsize=font_size)
    ax.set_ylabel('Y', fontsize=font_size)
    ax.set_title('Trajectories of Moving Obstacles', fontsize=font_size)
    ax.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()
    plt.close()

def visualize_final(histories: List[History], labels: List[str], parameter_in_study_name_save):
    fontsize = 14
    parameter_in_study_name_save = parameter_in_study_name_save
    plt.figure(figsize=(6, 4))  # Adjust figure size here
    plt.rcParams['font.size'] = fontsize
    
    line_styles = ['--', '-.', ':']
    colors = ['b', 'k', 'r', 'c', 'm', 'y', 'g']

    for idx, (history, label) in enumerate(zip(histories, labels)):
        line_style = line_styles[idx % len(line_styles)]
        color = colors[idx % len(colors)]
        plt.plot(history.t, np.array(history.v) * 3.6, line_style, color=color, label=f"{label}")
    plt.grid(color='lightgray', alpha=0.5)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Speed [km/h]", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"../results/mpc_sensitivity/{parameter_in_study_name_save}_speed.pdf")
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.rcParams['font.size'] = fontsize
    for idx, (history, label) in enumerate(zip(histories, labels)):
        line_style = line_styles[idx % len(line_styles)]
        color = colors[idx % len(colors)]
        plt.plot(history.t, history.a, line_style, color=color, label=f"{label}")
    plt.grid(color='lightgray', alpha=0.5)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Acceleration [$m/s^2$]", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"../results/mpc_sensitivity/{parameter_in_study_name_save}_acceleration.pdf")
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.rcParams['font.size'] = fontsize
    for idx, (history, label) in enumerate(zip(histories, labels)):
        line_style = line_styles[idx % len(line_styles)]
        color = colors[idx % len(colors)]
        plt.plot(history.t, history.xref_deviation, line_style, color=color, label=f"{label}")
    plt.grid(color='lightgray', alpha=0.5)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Deviation [m]", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"../results/mpc_sensitivity/{parameter_in_study_name_save}_deviation.pdf")
    plt.show()
    plt.close()

def visualize_frame(dt, frame_window, car_dimensions, collision_xy, i, moving_obstacles, mpcs, scenarios, simulations, states, tmp_trajectories, trajectory_ress, trajs_moving_obstacles):
    fontsize = 14
    if i >= 0:
        plt.cla()
        for mpc, scenario, simulation, state, tmp_trajectory, trajectory_res, trajs_moving_obstacle in zip(mpcs, scenarios, simulations, states, tmp_trajectories, trajectory_ress, trajs_moving_obstacles):
            plt.plot(tmp_trajectory[:, 0], tmp_trajectory[:, 1], color='b')

            if collision_xy is not None:
                plt.scatter([collision_xy[0]], [collision_xy[1]], color='r')

            plt.scatter([state.x], [state.y], color='r')
            plt.scatter([trajectory_res[0, 0]], [trajectory_res[0, 1]], color='b')

            plt.plot(simulation.history.x, simulation.history.y, '-r')

            if mpc.ox is not None:
                plt.plot(mpc.ox, mpc.oy, "+r", label="MPC")

            plt.plot(mpc.xref[0, :], mpc.xref[1, :], "+k", label="xref")

            draw_car((state.x, state.y, state.yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=plt.gca(), color='k')

        plt.title("Time: %.2f [s]" % (i * dt))
        plt.axis("equal")
        plt.grid(False)
        plt.legend(fontsize=fontsize)

        plt.xlim((-40, 10))
        plt.ylim((-45, 20))
        plt.pause(0.001)
        plt.close()

def plot_trajectories_comparison(obstacles_positions_list: List[List[List[tuple]]], ego_positions_list: List[History], reference_trajectory: np.ndarray, parameter_in_study: List[int], parameter_in_study_name, parameter_in_study_name_save, total_runtimes: List[float]):
    
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size here
    fontsize = 14
    # Define line styles and colors
    line_styles = ['--', '-.', ':']
    colors = ['b', 'k', 'r', 'c', 'm', 'y', 'g']

    # Plot the reference trajectory
    ax.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], '-', color='k', linewidth=1, label='Reference Trajectory')

    for idx, (ego_positions, obstacles_positions) in enumerate(zip(ego_positions_list, obstacles_positions_list)):
        times, ego_x, ego_y = ego_positions.t, ego_positions.x, ego_positions.y
        ego_positions_arr = np.column_stack((ego_x, ego_y))
        line_style = line_styles[idx % len(line_styles)]
        color = colors[idx % len(colors)]
        parameter_in_study_name = parameter_in_study_name
        parameter_value = parameter_in_study[idx]
        runtime = total_runtimes[idx]
        ax.plot(ego_positions_arr[:, 0], ego_positions_arr[:, 1], line_style, color=color, linewidth=1, label=f'{parameter_in_study_name}={parameter_value} (t: {runtime:.2f}s)')
        for i_obstacle, obstacle_positions in enumerate(obstacles_positions):
            times, positions = zip(*obstacle_positions)
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1], line_style, color=color, linewidth=0.5, label=f'Obstacle {i_obstacle + 1}' if idx == 0 else "")

    ax.set_xlabel('X', fontsize=fontsize)
    ax.set_ylabel('Y', fontsize=fontsize)
    plt.grid(color='lightgray', alpha=0.5)
    # ax.set_title('Trajectories of Moving Obstacles', fontsize=12)
    ax.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f"../results/mpc_sensitivity/{parameter_in_study_name_save}_trajectories.pdf")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
