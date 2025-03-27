# Standard library
import csv
import itertools
import math
import os
import subprocess
import sys
import time
from typing import List

# Third-party libraries
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

# Local modules
sys.path.append('..')
from envs.arterial_multi_lanes import ArterialMultiLanes
from lib.car_dimensions import CarDimensions, BicycleModelDimensions, BicycleRealDimensions
from lib.collision_avoidance import get_cutoff_curve_by_position_idx, check_collision_moving_bicycle
from lib.motion_primitive import load_motion_primitives
from lib.mp_search_ww_generic import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleArterial
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc import MPC, MAX_ACCEL
from lib.plotting import draw_car, draw_bicycle, draw_astar_search_points
from lib.reasons_evaluation import evaluate_distance_to_centerline, evaluate_distance_to_obstacle, evaluate_time_following
from lib.simulation import History, HistorySimulation, Simulation, State
from lib.trajectories import calc_nearest_index_in_direction, resample_curve

from dataclasses import dataclass

# Constants
@dataclass
class ScenarioParameters:
    DT = 0.1  # Time step
    CENTERLINE_LOCATION = 0.0  # Centerline location for evaluation

@dataclass
class ReasonParameters:
    REASONS_THRESHOLD = 0.7  # Threshold for reasons to trigger replan

@dataclass
class MPCParameters:
    TIME_HORIZON = 7.0  # Time horizon for predictions
    FRAME_WINDOW = 20  # Frame window for collision checking
    MAX_SPEED_FREEWAY = 30 / 3.6  # Maximum speed for freeway

@dataclass
class DriverParameters:
    DISTANCE_REF = 10.0  # Reference distance for driver patience
    DISTANCE_BUFFER = 2.0  # Buffer distance for driver patience
    TIME_THRESHOLD = 10.0  # Time threshold for driver patience

@dataclass
class CyclistParameters:
    DISTANCE_REF = 8.0  # Reference distance for cyclist patience
    DISTANCE_BUFFER = 1.0  # Buffer distance for cyclist patience
    TIME_THRESHOLD = 5.0  # Time threshold for cyclist patience
    SPEED = 5 / 3.6 # Cyclist speed [km/h]

# Initialize logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(replanner: bool = False) -> None:
    """
    Main function to simulate the scenario of an AV overtaking a cyclist in a bidirectional road.

    Args:
        replanner (bool): If True, enables replanner mode for the simulation.
    """

    # Initialization
    TIME_ELAPSED_DRIVER = 0  # time elapsed since the driver followed the bicycle at a distance less than DriverParameters.DISTANCE_REF
    TIME_PASSED_CYCLIST = 0 # time elapsed since the driver followed the bicycle at a distance less than CyclistParameters.DISTANCE_REF
    IS_FOLLOWING = True  # Flag to determine if the driver will be following the vehicle in front

    # Lists to store simulation data
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

    # Initialize simulation
    mps, car_dimensions, bicycle_dimensions, arterial, scenario_no_obstacles, scenario_visualization, moving_obstacles = initialize_simulation()

    # Run motion primitive search
    cost, path, trajectory_full, search_runtime = run_motion_primitive_search(scenario_no_obstacles, car_dimensions,
                                                                              mps)

    # Initialize MPC, set max speed to cyclist speed if the AV is following the cyclist
    if IS_FOLLOWING == True:
        mpc, state, dl = initialize_mpc(trajectory_full, car_dimensions, max_speed=CyclistParameters.SPEED)
    else:
        mpc, state, dl = initialize_mpc(trajectory_full, car_dimensions, max_speed=MPCParameters.MAX_SPEED_FREEWAY)

    simulation = HistorySimulation(car_dimensions=car_dimensions, sample_time=ScenarioParameters.DT, initial_state=state)
    history = simulation.history  # gets updated automatically as simulation runs

    # Number of index to cut off the trajectory before a collision occurs, change 2 for the larger index cut off
    EXTRA_CUTOFF_MARGIN = 2 * int(
        math.ceil(car_dimensions.radius / dl))  # no. of frames - corresponds approximately to car length

    traj_agent_idx = 0
    tmp_trajectory = None

    loop_runtimes = []

    # Create a list to store the location of obstacles through time for visualization
    obstacles_positions = [[] for _ in moving_obstacles]

    start_time = time.time()
    for i in itertools.count():
        loop_start_time = time.time()
        if mpc.is_goal(state):
            break

        # Update trajectory index to the nearest future index in the trajectory based on the current state
        traj_agent_idx = update_trajectory_index(state, tmp_trajectory, traj_agent_idx, trajectory_full)

        # Get the trajectory from the current index to the end, removing the points that have already passed
        trajectory_res = trajectory = trajectory_full[traj_agent_idx:]

        # Compute the predicted trajectory for the car
        trajectory_res = compute_predicted_trajectory(state, trajectory_res)

        # Predict the movement of each moving obstacle, and retrieve the predicted trajectories
        trajs_moving_obstacles = [
            np.vstack(MovingObstaclesPrediction(*o.get(), sample_time=ScenarioParameters.DT, car_dimensions=bicycle_dimensions)
                      .state_prediction(MPCParameters.TIME_HORIZON)).T
            for o in moving_obstacles]

        # Evaluate reasons
        reasons_policymaker_reg_compliance, reasons_driver_time_eff, reasons_cyclist_comfort, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST = evaluate_reasons(
            state, moving_obstacles, car_dimensions, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST)

        # Find the collision location
        collision_xy = check_collision_moving_bicycle(car_dimensions, bicycle_dimensions, trajectory_res, trajectory,
                                                      trajs_moving_obstacles,
                                                      frame_window=MPCParameters.FRAME_WINDOW)

        # If replanner is enabled, based on reasons evaluation, determine if a replan is needed
        if replanner == True:
            # Flag to determine if a replan should happen
            replan_needed = False

            # Evaluate reasons and determine if a replan is needed, replan_tracker is used to prevent multiple replans
            replan_needed, replan_tracker = reasons_evaluation(reasons_cyclist_comfort, reasons_driver_time_eff,
                                               reasons_policymaker_reg_compliance, replan_needed, replan_tracker)

            # Execute replan only if reasons value drop below ScenarioParameters.REASONS_THRESHOLD was detected
            if replan_needed:
                # Perform replan, reset the trajectory, MPC, and collision status
                IS_FOLLOWING = False
                # change max_speed of the MPC to 30/3.6
                collision_xy, mpc, traj_agent_idx, trajectory_full, scenario_obstacles = perform_replan(arterial, car_dimensions, dl, moving_obstacles, mps, state, trajs_moving_obstacles, max_speed=MPCParameters.MAX_SPEED_FREEWAY, is_following=IS_FOLLOWING)
                scenario = scenario_obstacles

        # Cut off the trajectory before a collision occurs, with an additional margin
        if collision_xy is not None:
            tmp_trajectory = cutoff_trajectory_before_collision(EXTRA_CUTOFF_MARGIN, collision_xy, traj_agent_idx,
                                                                trajectory_full)
        else:
            tmp_trajectory = trajectory_full


        # Pass the cut trajectory to the MPC
        mpc.set_trajectory_fromarray(tmp_trajectory)

        # Compute the MPC
        delta, acceleration = mpc.step(state)

        # Runtime calculation
        loop_end_time = time.time()
        loop_runtime = loop_end_time - loop_start_time
        loop_runtimes.append(loop_runtime)
        xref_deviation_value = mpc.get_current_xref_deviation()
        # show the computation results
        visualize_frame(ScenarioParameters.DT, car_dimensions, bicycle_dimensions, collision_xy, i, moving_obstacles, mpc,
                        scenario_visualization, simulation,
                        state, tmp_trajectory, trajectory_res,
                        reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, distance_values,
                        reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                        speed_values, time_values, xref_deviation_values, xref_deviation_value,
                        static_x_axis=True, max_time=15) # static_x_axis=False)


        # Move all obstacles one step ahead
        for i_obs, o in enumerate(moving_obstacles):
            obstacles_positions[i_obs].append((i, o.get()))  # i is time here
            o.step()

        # Step the simulation (i.e. move our agent forward)
        state = simulation.step(a=acceleration, delta=delta, xref_deviation=mpc.get_current_xref_deviation())

    # Print runtimes
    end_time = time.time()
    loops_total_runtime = sum(loop_runtimes)
    total_runtime = end_time - start_time
    logger.info('total loops run time is: {}'.format(loops_total_runtime))
    logger.info('total run time is: {}'.format(total_runtime))
    logger.info('each mpc runtime is: {}'.format(loops_total_runtime / len(loop_runtimes)))

    # Visualize final
    visualize_final(simulation.history)

    # Save vehicle data
    save_vehicle_data(simulation, time_values, reasons_policymaker_values, reasons_driver_values,
                      reasons_cyclist_values, replanner)

    # Plot Trajectories and Conflicts
    plot_trajectories(obstacles_positions, simulation.history)


def cutoff_trajectory_before_collision(EXTRA_CUTOFF_MARGIN, collision_xy, traj_agent_idx,
                                       trajectory_full) -> np.ndarray:
    """
    Cut off the trajectory before a collision occurs, with an additional margin.

    Args:
        EXTRA_CUTOFF_MARGIN: Additional margin to ensure safety.
        collision_xy: Coordinates of the collision point (x, y), or None if no collision.
        traj_agent_idx: Current index in the trajectory.
        trajectory_full: The full trajectory from the motion primitive search.

    Returns:
        np.ndarray: The truncated trajectory.
    """
    # cutoff the curve such that it ends right before the collision (and some margin)
    cutoff_idx = get_cutoff_curve_by_position_idx(trajectory_full, collision_xy[0],
                                                  collision_xy[1]) - EXTRA_CUTOFF_MARGIN
    cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
    tmp_trajectory = trajectory_full[:cutoff_idx]

    return tmp_trajectory


def compute_predicted_trajectory(state, trajectory_res):
    """
    Compute the predicted trajectory for the car based on its current speed and maximum acceleration.

    Args:
        state: The current state of the car (including velocity).
        trajectory_res: The current trajectory to be resampled.

    Returns:
        np.ndarray: The resampled trajectory based on acceleration.
    """
    if state.v < Simulation.MAX_SPEED:
        resample_dl = np.zeros((trajectory_res.shape[0],)) + MAX_ACCEL
        resample_dl = np.cumsum(resample_dl) + state.v
        resample_dl = ScenarioParameters.DT * np.minimum(resample_dl, Simulation.MAX_SPEED)
        trajectory_res = resample_curve(trajectory_res, dl=resample_dl)
    else:
        trajectory_res = resample_curve(trajectory_res, dl=ScenarioParameters.DT * Simulation.MAX_SPEED)
    return trajectory_res


def update_trajectory_index(state, tmp_trajectory, traj_agent_idx, trajectory_full):
    """
    Update the trajectory index based on the current state.

    Args:
        state: The current state of the vehicle.
        tmp_trajectory: The temporary trajectory.
        traj_agent_idx: The current index of the agent in the trajectory.
        trajectory_full: The full trajectory.

    Returns:
        int: The updated index of the agent in the trajectory
    """

    if tmp_trajectory is None or np.any(tmp_trajectory[traj_agent_idx, :] != tmp_trajectory[-1, :]):
        traj_agent_idx = calc_nearest_index_in_direction(state, trajectory_full[:, 0], trajectory_full[:, 1],
                                                         start_index=traj_agent_idx, forward=True)
    return traj_agent_idx


def perform_replan(arterial, car_dimensions, dl, moving_obstacles, mps, state, trajs_moving_obstacles, max_speed, is_following=True):
    """
    Perform a replan based on the current state and moving obstacles.

    Args:
        arterial: The arterial scenario.
        car_dimensions: Dimensions of the car.
        dl: Distance between trajectory points.
        moving_obstacles: List of moving obstacles.
        mps: Motion primitives.
        state: The current state of the vehicle.
        trajs_moving_obstacles: Predicted trajectories of moving obstacles.

    Returns:
        tuple: A tuple containing the updated trajectory, MPC, and collision status.
    """
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
        av_location_y=state.y,
        is_following=is_following
    )
    logger.info(f"Initial position: {scenario_obstacles.start}")
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
        dl=dl, speed=max_speed, dt=ScenarioParameters.DT, car_dimensions=car_dimensions
    )
    collision_xy = None  # Reset collision tracker if needed
    return collision_xy, mpc, traj_agent_idx, trajectory_full, scenario_obstacles


def reasons_evaluation(reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                       replan_needed, replan_tracker):
    """
    Check if any reason is below the threshold and trigger replanning if needed.

    Args:
        reasons_policymaker_reg_compliance: Policymaker compliance reason.
        reasons_driver_time_eff: Driver time efficiency reason.
        reasons_cyclist_comfort: Cyclist comfort reason.
        replan_tracker: Whether a replan has already been triggered.
        replan_needed: Whether a replan is needed.

    Returns:
        tuple: A tuple containing whether replan is needed and the updated replan tracker.
    """
    reasons = {
        "Policymaker Compliance": reasons_policymaker_reg_compliance,
        "Driver Time Efficiency": reasons_driver_time_eff,
        "Cyclist Comfort": reasons_cyclist_comfort,
    }

    # Check if any reason is below the threshold
    if any(value < ReasonParameters.REASONS_THRESHOLD for value in reasons.values()):
        if not replan_tracker:
            for reason_name, value in reasons.items():
                if value < ReasonParameters.REASONS_THRESHOLD:
                    logger.info(f'Reasons below {ReasonParameters.REASONS_THRESHOLD * 100}%, {reason_name}, Replan')
            replan_tracker = True
            replan_needed = True
    else:
        replan_tracker = False
        replan_needed = False

    return replan_needed, replan_tracker


def run_motion_primitive_search(scenario_no_obstacles, car_dimensions, mps) -> tuple:
    """
    Run the motion primitive search algorithm.

    Args:
        scenario_no_obstacles: The scenario without obstacles.
        car_dimensions: Dimensions of the car.
        mps: Motion primitives.

    Returns:
        tuple: A tuple containing the cost, path, and trajectory.
    """
    start_time = time.time()
    search = MotionPrimitiveSearch(scenario_no_obstacles, car_dimensions, mps, margin=car_dimensions.radius)
    cost, path, trajectory_full = search.run(debug=True)
    logger.info("Search finished")
    plot_motion_primitives(search, scenario_no_obstacles, path, car_dimensions)
    end_time = time.time()
    search_runtime = end_time - start_time
    logger.info(f"Search runtime: {search_runtime}")
    return cost, path, trajectory_full, search_runtime

def initialize_simulation() -> tuple:
    """
    Initialize simulation parameters, motion primitives, and the scenario.

    Returns:
        tuple: A tuple containing initialized objects and parameters.
    """
    # Load motion primitives
    mps = load_motion_primitives(version='bicycle_model')
    car_dimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)
    bicycle_dimensions = BicycleRealDimensions(skip_back_circle_collision_checking=False)

    # Define the scenario
    arterial = ArterialMultiLanes(num_lanes=2, goal_lane=1)
    scenario_no_obstacles = arterial.create_scenario()
    scenario_visualization = arterial.create_scenario(frame_visualization=True)

    # Define moving obstacles
    spawn_location_x = scenario_no_obstacles.start[0] + 1.7
    spawn_location_y = scenario_no_obstacles.start[1] + 8.8
    moving_obstacles = [
        MovingObstacleArterial(bicycle_dimensions, spawn_location_x, spawn_location_y, speed = CyclistParameters.SPEED, initial_speed = CyclistParameters.SPEED, offset=True, dt=ScenarioParameters.DT)
    ]

    return mps, car_dimensions, bicycle_dimensions, arterial, scenario_no_obstacles, scenario_visualization, moving_obstacles

def initialize_mpc(trajectory_full, car_dimensions, max_speed) -> tuple:
    """
    Initialize the MPC controller.

    Args:
        trajectory_full: The full trajectory from the motion primitive search.
        car_dimensions: Dimensions of the car.

    Returns:
        tuple: A tuple containing the MPC object, the initial state and the distance between the points in the trajectory.
    """
    dl = np.linalg.norm(trajectory_full[0, :2] - trajectory_full[1, :2])
    mpc = MPC(cx=trajectory_full[:, 0], cy=trajectory_full[:, 1], cyaw=trajectory_full[:, 2], dl=dl, dt=ScenarioParameters.DT, car_dimensions=car_dimensions, speed=max_speed)
    state = State(x=trajectory_full[0, 0], y=trajectory_full[0, 1], yaw=trajectory_full[0, 2], v=CyclistParameters.SPEED)
    return mpc, state, dl

def evaluate_reasons(state, moving_obstacles, car_dimensions, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST) -> tuple:
    """
    Evaluate the reasons for policymaker, driver, and cyclist.

    Args:
        state: Current state of the vehicle.
        moving_obstacles: List of moving obstacles.
        car_dimensions: Dimensions of the car.
        TIME_ELAPSED_DRIVER: Time elapsed for the driver.
        TIME_PASSED_CYCLIST: Time passed for the cyclist.

    Returns:
        tuple: A tuple containing the evaluated reasons and updated times.
    """
    car_width = car_dimensions.bounding_box_size[0]
    reasons_policymaker_reg_compliance = evaluate_distance_to_centerline(state.x, car_width, ScenarioParameters.CENTERLINE_LOCATION)
    reasons_driver_time_eff, TIME_ELAPSED_DRIVER = evaluate_time_following('driver_reasons', ScenarioParameters.DT, DriverParameters.DISTANCE_BUFFER, DriverParameters.DISTANCE_REF, DriverParameters.TIME_THRESHOLD, moving_obstacles, state, TIME_ELAPSED_DRIVER)
    reasons_cyclist_time_eff, TIME_PASSED_CYCLIST = evaluate_time_following('cyclist_reasons', ScenarioParameters.DT, CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF, CyclistParameters.TIME_THRESHOLD, moving_obstacles, state, TIME_PASSED_CYCLIST)
    reasons_cyclist_distance = evaluate_distance_to_obstacle(CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF, moving_obstacles, state)
    reasons_cyclist_comfort = reasons_cyclist_time_eff * reasons_cyclist_distance
    return reasons_policymaker_reg_compliance, reasons_driver_time_eff, reasons_cyclist_comfort, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST

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

def plot_deviation(ax,time_values,  xref_deviation_value):
    fontsize = 25
    ax.clear()
    ax.plot(time_values, xref_deviation_value, "-r", label="Deviation from reference trajectory")
    ax.grid(False)
    ax.set_xlabel("Time [s]", fontsize=fontsize)
    ax.set_ylabel("Deviation [m]", fontsize=fontsize)
    ax.set_ylim(0, 0.035)  # Set the y-axis limit

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
def setup_plot(ax, tmp_trajectory, collision_xy, state, trajectory_res):
    """Set up the basic plot elements."""
    ax.set_facecolor('#AFABAB')
    ax.plot(tmp_trajectory[:, 0], tmp_trajectory[:, 1], color='b')

    if collision_xy is not None:
        ax.scatter([collision_xy[0]], [collision_xy[1]], color='r')

    ax.scatter([state.x], [state.y], color='r')
    ax.scatter([trajectory_res[0, 0]], [trajectory_res[0, 1]], color='b')


def draw_static_elements(ax, scenario):
    """Draw static obstacles and vertical lines."""
    for obstacle in scenario.obstacles:
        obstacle.draw(ax, color='#9ED386')

    # Vertical lines
    ax.axvline(x=0 + 0.3, color='#FFBD00')
    ax.axvline(x=0 - 0.3, color='#FFBD00')
    ax.axvline(x=0 + 3.8, color='#FFFFFF')
    ax.axvline(x=0 - 3.8, color='#FFFFFF')


def update_obstacle_history(moving_obstacles, obstacle_history):
    """Update the history of moving obstacles."""
    for mo in moving_obstacles:
        x, y, _, theta, _, _ = mo.get()  # Get current state

        if mo not in obstacle_history:  # First time seeing this obstacle
            obstacle_history[mo] = []

        obstacle_history[mo].append((x, y, theta))  # Store position


def plot_historical_data(ax, simulation, mpc, car_dimensions, bicycle_dimensions, obstacle_history, plot_times, dt):
    """Plot historical data for the car and moving obstacles."""
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

        # Plot historical bicycles (moving obstacles)
        for mo in obstacle_history:
            if time_index < len(obstacle_history[mo]):
                x, y, theta = obstacle_history[mo][time_index]  # Retrieve past position

                draw_bicycle((x, y, theta), bicycle_dimensions, ax=ax,
                             draw_collision_circles=False, color='blue')

                # Label the time step **to the right** of the bicycle
                ax.text(x + 1.5, y + 0.5, f"{plot_time}s", fontsize=12, color='blue', ha='center')


def plot_current_state(ax, state, mpc, car_dimensions, moving_obstacles, bicycle_dimensions):
    """Plot the current state of the car and moving obstacles."""
    draw_car((state.x, state.y, state.yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=ax,
             color='black', draw_collision_circles=False)

    for mo in moving_obstacles:
        x, y, _, theta, _, _ = mo.get()  # Get current position
        draw_bicycle((x, y, theta), bicycle_dimensions, ax=ax,
                     draw_collision_circles=False, color='blue')


def finalize_plot(ax, simulation, mpc, i, dt):
    """Add final touches to the plot (trajectory, labels, legend, etc.)."""
    ax.plot(simulation.history.x, simulation.history.y, '--k', alpha=0.5)

    if mpc.ox is not None:
        ax.plot(mpc.ox, mpc.oy, "+r")

    ax.plot(mpc.xref[0, :], mpc.xref[1, :], "+k")

    # Legend for clarity
    ax.legend(fontsize=12, loc="upper right")

    # Axis labels and formatting
    ax.set_title(f"Time: {i * dt:.2f} [s]", fontsize=20)
    # ax.set_xlabel('X', fontsize=20)
    # ax.set_ylabel('Y', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.axis("equal")
    ax.grid(True)
    ax.set_xlim((-10, 10))
    ax.set_ylim((-42, 20))

    # remove xticks and yticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_car_and_obstacles(ax, tmp_trajectory, collision_xy, state, trajectory_res, scenario, moving_obstacles,
                           simulation, mpc, car_dimensions, bicycle_dimensions, i, dt, historical_plot=True):
    """Main function to plot the car and obstacles."""
    global obstacle_history  # Use global variable to track history across function calls

    # Set up the plot
    setup_plot(ax, tmp_trajectory, collision_xy, state, trajectory_res)

    # Draw static elements
    draw_static_elements(ax, scenario)

    # Update obstacle history
    update_obstacle_history(moving_obstacles, obstacle_history)

    # Plot simulation history
    ax.plot(simulation.history.x, simulation.history.y, '-r')

    if mpc.ox is not None:
        ax.plot(mpc.ox, mpc.oy, "+r")

    ax.plot(mpc.xref[0, :], mpc.xref[1, :], "+k")

    # Determine time steps for historical plotting
    time_elapsed = i * dt  # Current simulation time
    final_time = time_elapsed  # End of simulation time

    # Generate time points: 0, 5, 10, ..., final_time (if not already included)
    plot_times = list(range(0, int(final_time) + 1, 5))  # Every 5 seconds
    if final_time not in plot_times:  # Ensure last time step is included
        plot_times.append(int(final_time))

    if historical_plot:
        plot_historical_data(ax, simulation, mpc, car_dimensions, bicycle_dimensions, obstacle_history, plot_times, dt)
    else:
        plot_current_state(ax, state, mpc, car_dimensions, moving_obstacles, bicycle_dimensions)

    # Finalize the plot
    finalize_plot(ax, simulation, mpc, i, dt)



def plot_reasons(ax, time_values, reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values):
    # Plot reasons values over time
    # Define colors for better balance
    colors = ['dodgerblue', 'darkorange', 'mediumseagreen']

    ax.clear()  # Clear the axis for each new update

    '''
    ax.plot(time_values, reasons_policymaker_values, label=r'$R_{policymaker}$', color=colors[0],
             linestyle='--', linewidth=2)
    ax.plot(time_values, reasons_driver_values, label=r'$R_{driver}$', color=colors[1], linestyle='--',
             linewidth=2)
    ax.plot(time_values, reasons_cyclist_values, label=r'$R_{cyclist}$', color=colors[2], linestyle='--',
             linewidth=2)    
    '''
    ax.plot(time_values, reasons_policymaker_values, label=r'Regulatory compliance', color=colors[0],
             linestyle='--', linewidth=2)
    ax.plot(time_values, reasons_driver_values, label=r"Driver's patience", color=colors[1], linestyle='--',
             linewidth=2)
    ax.plot(time_values, reasons_cyclist_values, label=r"Cyclist's comfort", color=colors[2], linestyle='--',
             linewidth=2)

    ax.axhline(y=ReasonParameters.REASONS_THRESHOLD, color='red', linestyle=':', linewidth=2, label='Replan Threshold')
    # Annotate the threshold line
    # Annotate the threshold line with multiple lines of text
    ax.text(time_values[0] + 0.2, ReasonParameters.REASONS_THRESHOLD + 0.05,
            "If reasons below \nthreshold, replan",
            color='red', fontsize=11, verticalalignment='bottom')

    ax.set_xlabel('Time [s]', fontsize=25)
    ax.set_ylabel('Reasons [0-1]', fontsize=25)
    # ax.set_title('Reasons Values Over Time', fontsize=20)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(False)

def plot_velocity(ax, time_values, speed_values):
    # Plot reasons values over time
    ax.clear()  # Clear the axis for each new update

    # Plot the distance values
    ax.plot(time_values, speed_values, label='Speed of the Vehicle')

    # Set axis labels and title
    ax.set_xlabel('Time [s]', fontsize=25)
    ax.set_ylabel('Speed [km/h]', fontsize=25)
    ax.set_ylim(0, 35)  # Set the y-axis limit
    # ax.set_title('Speed of the Vehicle', fontsize=20)

    # Add legend
    # ax.legend(fontsize=16)

    # Enable grid
    ax.grid(False)

def plot_distance(ax, time_values, distance_values, DISTANCE_THRESHOLD_CAR, DISTANCE_THRESHOLD_BICYCLE):
    # Clear the axis for each new update
    ax.clear()

    # Plot the distance values
    ax.plot(time_values, distance_values, label='Distance between Car and Bicycle')

    # Define thresholds dynamically based on input values
    thresholds = [
        {'value': DISTANCE_THRESHOLD_CAR, 'color': 'r', 'label': f'{DISTANCE_THRESHOLD_CAR}m Threshold'},
        {'value': DISTANCE_THRESHOLD_BICYCLE, 'color': 'g', 'label': f'{DISTANCE_THRESHOLD_BICYCLE}m Threshold'}
    ]

    # Add vertical lines at the specified thresholds
    for threshold in thresholds:
        for i, dist in enumerate(distance_values):
            if i == 0:  # Skip the first iteration to avoid index issues
                continue
            if dist <= threshold['value'] and distance_values[i - 1] > threshold['value']:
                ax.axvline(
                    time_values[i],
                    color=threshold['color'],
                    linestyle='--',
                    label=threshold['label'] if threshold['label'] not in [line.get_label() for line in ax.get_lines()] else ''
                )

    # Set axis labels and title
    ax.set_xlabel('Time Frame', fontsize=15)
    ax.set_ylabel('Distance', fontsize=15)
    ax.set_title('Distance Between Car and Bicycle', fontsize=18)

    # Add legend
    ax.legend(fontsize=12)

    # Enable grid
    ax.grid(True)

def visualize_frame(dt, car_dimensions, bicycle_dimensions, collision_xy, i, moving_obstacles, mpc,
                    scenario, simulation, state, tmp_trajectory, trajectory_res,
                    reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, distance_values,
                    reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance, speed_values, time_values,  xref_deviation_values, xref_deviation_value,
                    static_x_axis=True, max_time=20):
    """
    Visualize the simulation frame with an option for static or dynamic x-axis.

    Parameters:
        static_x_axis (bool): If True, the x-axis is fixed to `max_time`. If False, the x-axis dynamically adjusts.
        max_time (float): Maximum time for the x-axis when `static_x_axis` is True.
    """
    # Define the folder path
    save_folder = os.path.join("..", "results", "reasons_evaluation")

    # Ensure the folder exists (create it if it doesn't)
    os.makedirs(save_folder, exist_ok=True)

    if i >= 0:
        # Create figure and grid layout (2 rows, 2 columns)
        fig = plt.figure(figsize=(10, 10))  # Adjust size for better visualization
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # First row taller

        # Create subplots
        ax1 = plt.subplot(gs[:, 0])  # Merge (1,1) and (2,1) into a single tall plot
        ax2 = plt.subplot(gs[0, 1])  # Top-right plot
        ax3 = plt.subplot(gs[1, 1])  # Bottom-right plot
        ax4 = plt.subplot(gs[2, 1])

        # Update ax1 with the car, obstacles, and other related information
        plot_car_and_obstacles(ax1, tmp_trajectory, collision_xy, state, trajectory_res, scenario, moving_obstacles,
                               simulation, mpc, car_dimensions, bicycle_dimensions, i, dt, historical_plot=False)

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

        # Set x-axis limits based on static or dynamic option
        if static_x_axis:
            # Use the predefined max_time for static x-axis
            ax2.set_xlim(0, max_time)
            ax3.set_xlim(0, max_time)
            ax4.set_xlim(0, max_time)
        else:
            # Use the current simulation time for dynamic x-axis
            current_max_time = time_value
            ax2.set_xlim(0, current_max_time)
            ax3.set_xlim(0, current_max_time)
            ax4.set_xlim(0, current_max_time)

        # Adjust layout and show the plot
        plt.tight_layout()

        # Save the figure to the specified folder
        save_path = os.path.join(save_folder, f"frame_{i:04d}.jpg")
        plt.savefig(save_path, format='jpg', dpi=300)

        # plt.pause(0.001)
        plt.close()

def save_vehicle_data(simulation, time_values, reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values, supervision):
    """
    Save vehicle data to a CSV file.
    """
    time_simulation = time_values[:-1]
    acceleration_vehicle = simulation.history.a[:-1]
    velocity_vehicle = simulation.history.v[:-1]
    x_position_vehicle = simulation.history.x[:-1]
    y_position_vehicle = simulation.history.y[:-1]
    yaw_vehicle = simulation.history.yaw[:-1]
    x_ref_deviation_vehicle = simulation.history.xref_deviation[:-1]

    save_folder = os.path.join("..", "results", "reasons_evaluation")
    os.makedirs(save_folder, exist_ok=True)

    if supervision:
        save_path = os.path.join(save_folder, "vehicle_data_spv.csv")
    else:
        save_path = os.path.join(save_folder, "vehicle_data_unspv.csv")

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'acceleration', 'v', 'x', 'y', 'yaw', 'x_ref_dev', 'reasons_policymaker', 'reasons_driver', 'reasons_cyclist'])
        for row in zip(time_simulation, acceleration_vehicle, velocity_vehicle, x_position_vehicle, y_position_vehicle, yaw_vehicle, x_ref_deviation_vehicle, reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values):
            writer.writerow(row)
    logger.info("Vehicle data saved")

if __name__ == '__main__':
    # Run the simulation with supervision disabled
    main(replanner=True)

    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the results folder relative to the script's directory
    results_folder = os.path.join(script_dir, "..", "results", "reasons_evaluation")

    # Change to the results folder
    os.chdir(results_folder)
    logger.info(f"Changed directory to: {os.getcwd()}")

    # Remove the last 15 frames before creating the video
    # WHY? IN THE LAST FRAMES WE MAKE A STOP, SO IT IS NOT INTERESTING TO SEE
    frame_files = sorted([f for f in os.listdir(results_folder) if f.startswith('frame_') and f.endswith('.jpg')])
    for frame_file in frame_files[-30:]:
        os.remove(os.path.join(results_folder, frame_file))

    # Run the ffmpeg command to create a video from frames
    subprocess.run([
        'ffmpeg', '-framerate', '5', '-i', 'frame_%04d.jpg',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'output_video.mp4'
    ])

