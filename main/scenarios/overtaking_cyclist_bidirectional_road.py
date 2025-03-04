import itertools
import math
import os
import csv
import subprocess
from typing import List
import sys

sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt

# from envs.t_intersection import t_intersection
from main.lib.obstacles import BoxObstacle
from main.envs.arterial_multi_lanes import ArterialMultiLanes
from main.lib.car_dimensions import CarDimensions, BicycleModelDimensions, BicycleRealDimensions
from main.lib.collision_avoidance import check_collision_moving_cars, get_cutoff_curve_by_position_idx, \
    check_collision_moving_bicycle
from main.lib.motion_primitive import load_motion_primitives
# from lib.motion_primitive_search import MotionPrimitiveSearch
from main.lib.motion_primitive_search_modified import MotionPrimitiveSearch
from main.lib.moving_obstacles import MovingObstacleArterial
from main.lib.moving_obstacles_prediction import MovingObstaclesPrediction
from main.lib.mpc import MPC, MAX_ACCEL
from main.lib.plotting import draw_car, draw_bicycle
from main.lib.simulation import State, Simulation, History, HistorySimulation
from main.lib.trajectories import resample_curve, calc_nearest_index_in_direction
from main.lib.plotting import draw_astar_search_points
from main.lib.reasons_evaluation import evaluate_distance_to_centerline, evaluate_time_following, \
    evaluate_distance_to_obstacle
import time


def plot_motion_primitives(search, scenario, path, car_dimensions):
    fig, ax = plt.subplots()
    draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)

    # Plot scenario with car
    car_state = State(x=scenario.start[0], y=scenario.start[1], yaw=scenario.start[2], v=0.0)
    plot_scenario_with_car(scenario, car_state, car_dimensions, ax=ax)
    plot_path(path, ax)
    plt.show()


def plot_path(path, ax):
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], label='Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Path')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')


def plot_scenario_with_car(scenario, car_state, car_dimensions, ax):
    # Plot obstacles
    for obstacle in scenario.obstacles:
        obstacle.draw(ax, color='b')

    # Plot car
    draw_car((car_state.x, car_state.y, car_state.yaw), car_dimensions, ax=ax, color='r')

    # Plot goal area
    goal_area = scenario.goal_area
    goal_area.draw(ax, color='g')

    plt.title('Scenario with Car Location')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def main(supervision=False):
    #########
    # INIT ENVIRONMENT
    #########
    ###################### Scenario Parameters #####################
    DT = 0.2

    ###################### Reasons Parameters #####################
    time_passed_driver = 0  # for driver reasons
    distance_ref_driver = 10  # below 10 meters is the trigger of the driver patience
    distance_buffer_driver = 2  # +- 2 meters is the buffer for the driver patience
    time_threshold_driver = 10  # 10 seconds is the threshold for the driver patience

    time_passed_cyclist = 0
    distance_ref_cyclist = 8  # below 5 meters is the trigger of the driver patience
    distance_buffer_cyclist = 1  # +- 1 meters is the buffer for the driver patience
    time_threshold_cyclist = 5  # 5 seconds is the threshold for the driver patience
    reasons_policymaker_values = []
    reasons_driver_values = []
    reasons_cyclist_values = []
    distance_values = []
    speed_values = []
    xref_deviation_values = []
    time_values = []


    # A variable that saves information whether a reason has already triggered a replan
    replan_tracker = False

    # Prepare folder to save the results
    # Define the folder path
    save_folder = os.path.join("..", "results", "reasons_evaluation")

    # Ensure the folder exists (create it if it doesn't)
    os.makedirs(save_folder, exist_ok=True)

    # Delete all existing .jpg files in the folder
    for file in os.listdir(save_folder):
        if file.endswith(".jpg"):
            os.remove(os.path.join(save_folder, file))

    # Load motion primitives
    mps = load_motion_primitives(version='bicycle_model')
    # get car dimensions
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)
    # get bicycle dimensions
    bicycle_dimensions = BicycleRealDimensions(skip_back_circle_collision_checking=False)

    # when defining the scenario, there will be no moving obstacles
    arterial = ArterialMultiLanes(num_lanes=2, goal_lane=1)
    scenario_no_obstacles = arterial.create_scenario()

    # scenario = t_intersection(turn_left=True)
    print('scenario created')
    spawn_location_x = scenario_no_obstacles.start[0] + 1.7
    spawn_location_y = scenario_no_obstacles.start[1] + 10  # offset location to give distance to the ego vehicle
    moving_obstacles: List[MovingObstacleArterial] = [
        MovingObstacleArterial(bicycle_dimensions, spawn_location_x, spawn_location_y, 5 / 3.6, True, DT), ]

    #########
    # MOTION PRIMITIVE SEARCH
    #########

    start_time = time.time()

    # Initial search before facing any obstacles
    search = MotionPrimitiveSearch(scenario_no_obstacles, car_dimensions, mps, margin=car_dimensions.radius)
    # search.run will run a* search and return the cost, path, and trajectory
    cost, path, trajectory_full = search.run(debug=True)
    print('search finished')
    plot_motion_primitives(search, scenario_no_obstacles, path, car_dimensions)
    end_time = time.time()
    search_runtime = end_time - start_time
    print('search runtime is: {}'.format(search_runtime))

    #########
    # INIT MPC
    #########
    dl = np.linalg.norm(trajectory_full[0, :2] - trajectory_full[1, :2])

    mpc = MPC(cx=trajectory_full[:, 0], cy=trajectory_full[:, 1], cyaw=trajectory_full[:, 2], dl=dl, dt=DT,
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
    CENTERLINE_LOCATION = 0.0

    traj_agent_idx = 0
    tmp_trajectory = None

    loop_runtimes = []

    # creating a list to store the location of obstacles through time for visualization
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
            np.vstack(MovingObstaclesPrediction(*o.get(), sample_time=DT, car_dimensions=bicycle_dimensions)
                      .state_prediction(TIME_HORIZON)).T
            for o in moving_obstacles]

        # find the collision location
        collision_xy = check_collision_moving_bicycle(car_dimensions, bicycle_dimensions, trajectory_res, trajectory,
                                                      trajs_moving_obstacles,
                                                      frame_window=FRAME_WINDOW)

        # Evaluate reasons
        car_width = car_dimensions.bounding_box_size[0]
        reasons_policymaker_reg_compliance = evaluate_distance_to_centerline(state.x, car_width, CENTERLINE_LOCATION)
        reasons_driver_time_eff, time_passed_driver = evaluate_time_following('driver_reasons', DT,
                                                                              distance_buffer_driver,
                                                                              distance_ref_driver,
                                                                              time_threshold_driver, moving_obstacles,
                                                                              state, time_passed_driver)
        reasons_cyclist_time_eff, time_passed_cyclist = evaluate_time_following('cyclist_reasons', DT,
                                                                                distance_buffer_cyclist,
                                                                                distance_ref_cyclist,
                                                                                time_threshold_cyclist,
                                                                                moving_obstacles, state,
                                                                                time_passed_cyclist)
        # print('time efficiency: ', reasons_cyclist_time_eff)
        reasons_cyclist_distance = evaluate_distance_to_obstacle(distance_buffer_cyclist, distance_ref_cyclist,
                                                                 moving_obstacles, state)
        # print('distance: ', reasons_cyclist_distance)
        reasons_cyclist_comfort = reasons_cyclist_time_eff * reasons_cyclist_distance

        if supervision == True:
            replan_needed = False  # Flag to determine if a replan should happen
            reasons_threshold = 0.7  # Threshold for reasons to trigger replan
            if reasons_policymaker_reg_compliance < reasons_threshold or reasons_driver_time_eff < reasons_threshold or reasons_cyclist_comfort < reasons_threshold:
                if replan_tracker == False:
                    if reasons_policymaker_reg_compliance < reasons_threshold:
                        print('Reasons below 70%, Policymaker Compliance, Replan')
                    if reasons_driver_time_eff < reasons_threshold:
                        print('Reasons below 70%, Driver Time Efficiency, Replan')
                    if reasons_cyclist_comfort < reasons_threshold:
                        print('Reasons below 70%, Cyclist TComfort, Replan')
                    replan_tracker = True
                    replan_needed = True
            else:
                replan_tracker = False

            # Execute replan only if a new drop below 0.8 was detected
            if replan_needed:
                # Get current moving_obstacles x, y position
                moving_obstacles_x = [o.get()[0] for o in moving_obstacles]
                moving_obstacles_y = [o.get()[1] for o in moving_obstacles]

                # Create a new scenario with updated obstacle positions
                # NEXTTODO: Change spawn location and av location in the same variable. If possible to make it possible to
                #  spawn more vehicles in the same or different heading
                scenario_obstacles = arterial.create_scenario(
                    moving_obstacles=True,
                    moving_obstacles_trajectory=trajs_moving_obstacles,
                    spawn_location_x=moving_obstacles_x[0],
                    spawn_location_y=moving_obstacles_y[0],
                    av_location_x=state.x,
                    av_location_y=state.y
                )

                print(f"Initial position: {scenario_obstacles.start}")

                # Perform motion primitive search
                search = MotionPrimitiveSearch(scenario_obstacles, car_dimensions, mps, margin=car_dimensions.radius)
                cost, path, trajectory_full = search.run(debug=True)

                tmp_trajectory = trajectory_full
                traj_agent_idx = 0  # Reset trajectory index
                # Reset MPC with new trajectory
                mpc = MPC(
                    cx=trajectory_full[:, 0],
                    cy=trajectory_full[:, 1],
                    cyaw=trajectory_full[:, 2],
                    dl=dl, dt=DT, car_dimensions=car_dimensions
                )

                collision_xy = None  # Reset collision tracker if needed

        # cutoff the curve such that it ends right before the collision (and some margin)
        if collision_xy is not None:
            cutoff_idx = get_cutoff_curve_by_position_idx(trajectory_full, collision_xy[0],
                                                          collision_xy[1]) - EXTRA_CUTOFF_MARGIN
            cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
            ## add if to go back to the global planner ##################
            # evaluate the current position of the vehicle and the moving obstacles, evaluate the reasons.
            # need to first define the reasons. Use the one created in the power point.
            # recreate scenario with MotionPrimitiveSearch, with the current location of the vehicle and the moving obstacles
            # when go back to the planner, add moving obstacles to the scenario
            tmp_trajectory = trajectory_full[:cutoff_idx]
        else:
            tmp_trajectory = trajectory_full

        '''
        CASE WHERE REPLANNER IS PERFORMED WHEN COLLISION_XY is TRUE
        if collision_xy is not None:
            cutoff_idx = get_cutoff_curve_by_position_idx(trajectory_full, collision_xy[0],
                                                          collision_xy[1]) - EXTRA_CUTOFF_MARGIN
            cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
            # IF REASONS < 80% THEN GO BACK TO THE PLANNER
            if supervision == True:
                if reasons_policymaker_reg_compliance < 0.8 or reasons_driver_time_eff < 0.8 or reasons_cyclist_time_eff < 0.8:
                    # I want to print which reasons is below 0.8
                    if reasons_policymaker_reg_compliance < 0.8:
                        # print reasons policmaker compliance
                        print(reasons_policymaker_values)
                        print('Reasons below 80%, Policymaker Compliance, Replan')
                    if reasons_driver_time_eff < 0.8:
                        print('Reasons below 80%, Driver Time Efficiency, Replan')
                    if reasons_cyclist_comfort < 0.8:
                        print('Reasons below 80%, Cyclist Comfort, Replan')

                    # get current moving_obstacles x position
                    moving_obstacles_x = [o.get()[0] for o in moving_obstacles]
                    # get current moving_obstacles y position
                    moving_obstacles_y = [o.get()[1] for o in moving_obstacles]
                    scenario_obstacles = arterial.create_scenario(moving_obstacles=True,
                                                                  spawn_location_x=moving_obstacles_x[0],
                                                                  spawn_location_y=moving_obstacles_y[0],
                                                                  av_location_x=state.x, av_location_y=state.y)
                    print(f"Initial position: {scenario_obstacles.start}")
                    search = MotionPrimitiveSearch(scenario_obstacles, car_dimensions, mps,
                                                   margin=car_dimensions.radius)
                    # search.run will run a* search and return the cost, path, and trajectory
                    cost, path, trajectory_full = search.run(debug=True)
                    tmp_trajectory = trajectory_full
                    # reset index of the trajectory of the agent
                    traj_agent_idx = 0
                    # reset mpc with new trajectory_full
                    mpc = MPC(cx=trajectory_full[:, 0], cy=trajectory_full[:, 1], cyaw=trajectory_full[:, 2], dl=dl,
                              dt=DT,
                              car_dimensions=car_dimensions)
                else:
                    tmp_trajectory = trajectory_full[:cutoff_idx]
            else:
                tmp_trajectory = trajectory_full[:cutoff_idx]
        else:
            tmp_trajectory = trajectory_full
        '''

        # pass the cut trajectory to the MPC
        mpc.set_trajectory_fromarray(tmp_trajectory)

        # compute the MPC
        delta, acceleration = mpc.step(state)

        # runtime calculation
        loop_end_time = time.time()
        loop_runtime = loop_end_time - loop_start_time
        loop_runtimes.append(loop_runtime)
        xref_deviation_value = mpc.get_current_xref_deviation()
        # show the computation results
        visualize_frame(DT, car_dimensions, bicycle_dimensions, collision_xy, i, moving_obstacles, mpc,
                        scenario_no_obstacles, simulation,
                        state, tmp_trajectory, trajectory_res,
                        reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, distance_values,
                        # empty arrays
                        reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                        speed_values, time_values, xref_deviation_values, xref_deviation_value)  # value to the empty arrays

        # move all obstacles forward
        for i_obs, o in enumerate(moving_obstacles):
            obstacles_positions[i_obs].append((i, o.get()))  # i is time here
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

    '''

    '''
    # save the vehicle data
    time_simulation = time_values[:-1]
    acceleration_vehicle = simulation.history.a[:-1]
    velocity_vehicle = simulation.history.v[:-1]
    x_position_vehicle = simulation.history.x[:-1]
    y_position_vehicle = simulation.history.y[:-1]
    yaw_vehicle = simulation.history.yaw[:-1]
    x_ref_deviation_vehicle = simulation.history.xref_deviation[:-1]
    if supervision == True:
        # Change directory to the desired location
        os.chdir('/Users/lsuryana/PycharmProjects/MPC_for_AV_at_Intersection/main/results/reasons_evaluation/')
        # Use the absolute path for the save file
        save_path = os.path.join(os.getcwd(), "vehicle_data_spv.csv")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'acceleration_spv', 'v_spv', 'x_spv', 'y_spv', 'yaw_spv', 'x_ref_dev_spv',
                             'reasons_policymaker_spv', 'reasons_driver_spv', 'reasons_cyclist_spv'])
            # Write rows: Iterate over all lists simultaneously
            for row in zip(
                    time_simulation, acceleration_vehicle, velocity_vehicle, x_position_vehicle,
                    y_position_vehicle, yaw_vehicle, x_ref_deviation_vehicle,
                    reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values
            ):
                writer.writerow(row)
            print('vehicle data saved')
    else:
        # Change directory to the desired location
        os.chdir('/Users/lsuryana/PycharmProjects/MPC_for_AV_at_Intersection/main/results/reasons_evaluation/')
        # Use the absolute path for the save file
        save_path = os.path.join(os.getcwd(), "vehicle_data_unspv.csv")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'acceleration_unspv', 'v_unspv', 'x_unspv', 'y_unspv', 'yaw_unspv', 'x_ref_dev_unspv',
                             'reasons_policymaker_unspv', 'reasons_driver_unspv', 'reasons_cyclist_unspv'])
            # Write rows: Iterate over all lists simultaneously
            for row in zip(
                    time_simulation, acceleration_vehicle, velocity_vehicle, x_position_vehicle,
                    y_position_vehicle, yaw_vehicle, x_ref_deviation_vehicle,
                    reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values
            ):
                writer.writerow(row)

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
        plt.plot(ego_positions[(i_time - 1):(i_time + 1), 0], ego_positions[(i_time - 1):(i_time + 1), 1], color=color,
                 linewidth=8)

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
            plt.plot(positions[(i_time - 1):(i_time + 1), 0], positions[(i_time - 1):(i_time + 1), 1], color=color,
                     linewidth=4)

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

def plot_deviation(ax,time_values,  xref_deviation_value):
    fontsize = 25
    ax.clear()
    ax.plot(time_values, xref_deviation_value, "-r", label="Deviation from reference trajectory")
    ax.grid(True)
    ax.set_xlabel("Time [s]", fontsize=fontsize)
    ax.set_ylabel("Deviation [m]", fontsize=fontsize)

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

# Dictionary to store moving obstacle history
obstacle_history = {}  # Key: obstacle, Value: list of (x, y, theta) over time
def plot_car_and_obstacles(ax, tmp_trajectory, collision_xy, state, trajectory_res, scenario, moving_obstacles,
                           simulation, mpc, car_dimensions, bicycle_dimensions, i, dt, historical_plot=True):
    # Plot the vehicle's trajectory and obstacles (Original plot)
    global obstacle_history  # Use global variable to track history across function calls

    ax.set_facecolor('#AFABAB')
    ax.plot(tmp_trajectory[:, 0], tmp_trajectory[:, 1], color='b')

    if collision_xy is not None:
        ax.scatter([collision_xy[0]], [collision_xy[1]], color='r')

    ax.scatter([state.x], [state.y], color='r')
    ax.scatter([trajectory_res[0, 0]], [trajectory_res[0, 1]], color='b')

    # Explicitly call draw method on ax for each obstacle
    for obstacle in scenario.obstacles:
        obstacle.draw(ax, color='#9ED386')  # Draw obstacles on ax1

    # Vertical lines
    ax.axvline(x=0 + 0.3, color='#FFBD00')
    ax.axvline(x=0 - 0.3, color='#FFBD00')
    ax.axvline(x=0 + 3.8, color='#FFFFFF')
    ax.axvline(x=0 - 3.8, color='#FFFFFF')

    # **Track moving obstacle history**
    for mo in moving_obstacles:
        x, y, _, theta, _, _ = mo.get()  # Get current state

        if mo not in obstacle_history:  # First time seeing this obstacle
            obstacle_history[mo] = []

        obstacle_history[mo].append((x, y, theta))  # Store position

    ax.plot(simulation.history.x, simulation.history.y, '-r')

    if mpc.ox is not None:
        ax.plot(mpc.ox, mpc.oy, "+r", label="MPC")

    ax.plot(mpc.xref[0, :], mpc.xref[1, :], "+k", label="xref")

    # **Determine time steps for historical plotting**
    time_elapsed = i * dt  # Current simulation time
    final_time = time_elapsed  # End of simulation time

    # Generate time points: 0, 5, 10, ..., final_time (if not already included)
    plot_times = list(range(0, int(final_time) + 1, 5))  # Every 5 seconds
    if final_time not in plot_times:  # Ensure last time step is included
        plot_times.append(int(final_time))

    if historical_plot:
        # **Plot historical cars**
        for plot_time in plot_times:
            time_index = int(plot_time / dt)  # Convert time to index in history
            if time_index < len(simulation.history.x):  # Ensure index is within bounds
                x, y, yaw = (simulation.history.x[time_index],
                             simulation.history.y[time_index],
                             simulation.history.yaw[time_index])

                draw_car((x, y, yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=ax,
                         color='black', draw_collision_circles=False)

                # Label the time step **to the left** of the car
                ax.text(x - 2.0, y + 0.5, f"{plot_time}s", fontsize=12, color='black', ha='center')

        # **Plot historical bicycles (moving obstacles)**
        for plot_time in plot_times:
            time_index = int(plot_time / dt)  # Convert time to index

            for mo in moving_obstacles:
                if mo in obstacle_history and time_index < len(obstacle_history[mo]):
                    x, y, theta = obstacle_history[mo][time_index]  # Retrieve past position

                    draw_bicycle((x, y, theta), bicycle_dimensions, ax=ax,
                                 draw_collision_circles=False, color='blue')

                    # Label the time step **to the right** of the bicycle
                    ax.text(x + 1.5, y + 0.5, f"{plot_time}s", fontsize=12, color='blue', ha='center')

    else:
        # **Plot only current car**
        draw_car((state.x, state.y, state.yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=ax,
                 color='black', draw_collision_circles=False)

        # **Plot only current moving obstacles as bicycles**
        for mo in moving_obstacles:
            x, y, _, theta, _, _ = mo.get()  # Get current position
            draw_bicycle((x, y, theta), bicycle_dimensions, ax=ax,
                         draw_collision_circles=False, color='blue')

        # **Plot trajectory history as dashed lines**
        ax.plot(simulation.history.x, simulation.history.y, '--k', alpha=0.5, label="Car Path")

        if mpc.ox is not None:
            ax.plot(mpc.ox, mpc.oy, "+r", label="MPC Reference")

        ax.plot(mpc.xref[0, :], mpc.xref[1, :], "+k", label="xref")

        # **Legend for clarity**
        ax.legend(fontsize=12, loc="upper right")

        # **Axis labels and formatting**
        ax.set_title(f"Time: {i * dt:.2f} [s]", fontsize=20)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.axis("equal")
        ax.grid(True)
        ax.set_xlim((-10, 10))
        ax.set_ylim((-50, 52))



def plot_reasons(ax, time_values, reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values):
    # Plot reasons values over time
    # Define colors for better balance
    colors = ['royalblue', 'crimson', 'seagreen']

    ax.clear()  # Clear the axis for each new update

    ax.plot(time_values, reasons_policymaker_values, label=r'$R_{policymaker}$', color=colors[0],
             linestyle='-.', linewidth=2)
    ax.plot(time_values, reasons_driver_values, label=r'$R_{driver}$', color=colors[1], linestyle='-.',
             linewidth=2)
    # If you see cyclist reasons intermitten it is because the time_passed is updated every time it is below the threshold.
    # When it is above threshold, for instance 5, it will decrease. But when the score above threshold again, it becomes 1
    ax.plot(time_values, reasons_cyclist_values, label=r'$R_{cyclist}$', color=colors[2], linestyle='-.',
             linewidth=2)

    ax.set_xlabel('Time [s]', fontsize=25)
    ax.set_ylabel('Reasons [0-1]', fontsize=25)
    # ax.set_title('Reasons Values Over Time', fontsize=20)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=14, loc='lower left')
    ax.grid(True)

def plot_velocity(ax, time_values, speed_values):
    # Plot reasons values over time
    ax.clear()  # Clear the axis for each new update

    # Plot the distance values
    ax.plot(time_values, speed_values, label='Speed of the Vehicle')

    # Set axis labels and title
    ax.set_xlabel('Time [s]', fontsize=25)
    ax.set_ylabel('Speed [km/h]', fontsize=25)
    # ax.set_title('Speed of the Vehicle', fontsize=20)

    # Add legend
    # ax.legend(fontsize=16)

    # Enable grid
    ax.grid(True)

def plot_distance(ax, time_values, distance_values):
    # Plot reasons values over time
    ax.clear()  # Clear the axis for each new update

    # Plot the distance values
    ax.plot(time_values, distance_values, label='Distance between Car and Bicycle')

    # Add vertical lines at 12m and 9m distance
    # Use axvline to add vertical lines at the time corresponding to distances of 12 and 9 meters
    for i, dist in enumerate(distance_values):
        if dist <= 12 and distance_values[i - 1] > 12:  # Check when the distance crosses 12 meters
            ax.axvline(time_values[i], color='r', linestyle='--',
                       label='12m Threshold' if '12m Threshold' not in [line.get_label() for line in
                                                                        ax.get_lines()] else '')
        if dist <= 9 and distance_values[i - 1] > 9:  # Check when the distance crosses 9 meters
            ax.axvline(time_values[i], color='g', linestyle='--',
                       label='9m Threshold' if '9m Threshold' not in [line.get_label() for line in
                                                                        ax.get_lines()] else '')

    # Set axis labels and title
    ax.set_xlabel('Time Frame', fontsize=15)
    ax.set_ylabel('Distance', fontsize=15)
    ax.set_title('Distance Between Car and Bicycle', fontsize=18)

    # Add legend
    ax.legend(fontsize=12)

    # Enable grid
    ax.grid(True)


import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec  # Import gridspec to merge subplots

def visualize_frame(dt, car_dimensions, bicycle_dimensions, collision_xy, i, moving_obstacles, mpc,
                    scenario, simulation, state, tmp_trajectory, trajectory_res,
                    reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, distance_values,
                    reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance, speed_values, time_values,  xref_deviation_values, xref_deviation_value):
    # Define the folder path
    save_folder = os.path.join("..", "results", "reasons_evaluation")

    # Ensure the folder exists (create it if it doesn't)
    os.makedirs(save_folder, exist_ok=True)

    if i >= 0:
        # Create figure and grid layout (2 rows, 2 columns)
        fig = plt.figure(figsize=(8, 10))  # Adjust size for better visualization
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # First row taller

        # Create subplots
        ax1 = plt.subplot(gs[:, 0])  # Merge (1,1) and (2,1) into a single tall plot
        ax2 = plt.subplot(gs[0, 1])  # Top-right plot
        ax3 = plt.subplot(gs[1, 1])  # Bottom-right plot
        ax4 = plt.subplot(gs[2, 1])

        # Update ax1 with the car, obstacles, and other related information
        plot_car_and_obstacles(ax1, tmp_trajectory, collision_xy, state, trajectory_res, scenario, moving_obstacles,
                               simulation, mpc, car_dimensions, bicycle_dimensions, i, dt, historical_plot=True)

        # Update ax2 with reasons values over time
        time_value = i * dt  # Time value for current simulation step
        time_values.append(time_value)
        reasons_policymaker_values.append(reasons_policymaker_reg_compliance)
        reasons_driver_values.append(reasons_driver_time_eff)
        reasons_cyclist_values.append(reasons_cyclist_comfort)
        speed_values.append(state.v * 3.6) # times 3.6 to convert from m/s to km/h
        distance_values.append(np.linalg.norm(
            [moving_obstacles[0].get()[0] - state.x,
             moving_obstacles[0].get()[1] - state.y]))  # Placeholder for distance values
        xref_deviation_values.append(xref_deviation_value)
        # Plot the reasons and velocity in separate subplots
        plot_reasons(ax2, time_values, reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values)
        plot_velocity(ax3, time_values, speed_values)
        plot_deviation(ax4, time_values, xref_deviation_values)

        # Adjust layout and show the plot
        plt.tight_layout()

        # Save the figure to the specified folder
        save_path = os.path.join(save_folder, f"frame_{i:04d}.jpg")
        plt.savefig(save_path, format='jpg', dpi=300)

        # plt.pause(0.001)
        plt.close()



if __name__ == '__main__':
    # if supervision=True, the vehicle will be replanned if the reasons are below 80%
    main(supervision=False)
    # Change directory
    print(os.getcwd())
    os.chdir('/Users/lsuryana/PycharmProjects/MPC_for_AV_at_Intersection/main/results/reasons_evaluation/')
    # Run the ffmpeg command
    subprocess.run(['ffmpeg', '-framerate', '5', '-i', 'frame_%04d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    'output_video.mp4'])


