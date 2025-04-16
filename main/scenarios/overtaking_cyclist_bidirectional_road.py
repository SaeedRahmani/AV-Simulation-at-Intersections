# Standard library
import copy
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
from lib.mp_search_reasoning import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleArterial
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc import MPC, MAX_ACCEL
from lib.plotting import draw_car, draw_bicycle, draw_astar_search_points
from lib.reasons_evaluation import evaluate_distance_to_centerline, evaluate_distance_to_obstacle, evaluate_time_following
from lib.simulation import History, HistorySimulation, Simulation, State
from lib.trajectories import calc_nearest_index_in_direction, resample_curve
# In overtaking_cyclist_bidirectional_road.py
from lib.parameters import CyclistParameters, DriverParameters, ScenarioParameters, MPCParameters, ReasonParameters

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
                collision_xy, mpc, traj_agent_idx, trajectory_full, scenario_obstacles = perform_replan(
                            arterial, car_dimensions, bicycle_dimensions, dl, moving_obstacles, mps, state,
                            trajs_moving_obstacles, scenario_visualization,
                            reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                            reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values,
                            time_values,
                            max_speed=MPCParameters.MAX_SPEED_FREEWAY,
                            is_following=IS_FOLLOWING,
                            time_elapsed_driver=TIME_ELAPSED_DRIVER,
                            time_passed_cyclist=TIME_PASSED_CYCLIST
                        )
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


def compute_predicted_trajectory(state, trajectory_res, last_index=None):
    """
    Compute the predicted trajectory for the car based on its current speed and maximum acceleration.

    Args:
        state: The current state of the car (including velocity).
        trajectory_res: The current trajectory to be resampled.

    Returns:
        np.ndarray: The resampled trajectory based on acceleration.
    """
    if last_index is None:
        if state.v < Simulation.MAX_SPEED:
            resample_dl = np.zeros((trajectory_res.shape[0],)) + MAX_ACCEL
            resample_dl = np.cumsum(resample_dl) + state.v
            resample_dl = ScenarioParameters.DT * np.minimum(resample_dl, Simulation.MAX_SPEED)
            trajectory_res = resample_curve(trajectory_res, dl=resample_dl)
        else:
            trajectory_res = resample_curve(trajectory_res, dl=ScenarioParameters.DT * Simulation.MAX_SPEED)
        return trajectory_res
    else:
        trajectory_res = resample_curve(trajectory_res, dl=ScenarioParameters.DT * state.v)
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


def perform_replan(arterial, car_dimensions, bicycle_dimensions, dl, moving_obstacles, mps, state,
                   trajs_moving_obstacles, scenario_visualization,
                   reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                   reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values,
                   time_values, max_speed,
                   is_following=True, time_elapsed_driver=0.0, time_passed_cyclist=0.0):
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
    # Extract bicycle position and velocity for the motion planner
    bicycle_x = moving_obstacles[0].get()[0]
    bicycle_y = moving_obstacles[0].get()[1]
    bicycle_v = moving_obstacles[0].get()[4]
    bicycle_state = np.array([bicycle_x, bicycle_y, bicycle_v])

    # Create a new scenario with prediction of obstacle positions from current time to the TIME_HORIZON ahead
    scenario_obstacles = arterial.create_scenario(
        moving_obstacles=True,
        moving_obstacles_trajectory=trajs_moving_obstacles,
        spawn_location_x=bicycle_x,
        spawn_location_y=bicycle_y,
        av_location_x=state.x,
        av_location_y=state.y,
        is_following=is_following
    )
    logger.info(f"Initial position: {scenario_obstacles.start}")

    # Perform motion primitive search
    search = MotionPrimitiveSearch(scenario_obstacles, car_dimensions, mps, margin=car_dimensions.radius,
                                   moving_obstacles_state=bicycle_state,
                                    driver_elapsed_time=time_elapsed_driver,
                                    cyclist_elapsed_time=time_passed_cyclist
                                )

    # Calculate all trajectories
    costs, paths, trajectories_full = search.run_all(debug=True)

    # Create a new trajectory for the ego vehicle to follow the cyclist
    follow_trajectory = create_following_trajectory(state, trajectories_full)

    # Add the trajectory to the list of trajectories to the last position
    trajectories_full.append((follow_trajectory,(0.0, 0.0, 0.0, 0.0, 0.0)))

    # Evaluate trajectories based on human-centered reasons
    agent_weights, eval_results = evaluate_trajectories_for_reasons(
        trajectories_full,
        moving_obstacles,
        state,
        car_dimensions,
        bicycle_dimensions,
        reasons_cyclist_comfort,
        reasons_driver_time_eff,
        reasons_policymaker_reg_compliance,
        time_elapsed_driver=time_elapsed_driver,
        time_passed_cyclist=time_passed_cyclist
    )

    # In perform_replan after evaluation

    visualize_trajectory_evaluations(
        eval_results, trajectories_full, moving_obstacles, state, car_dimensions, bicycle_dimensions, scenario_visualization, time_values,
        reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, agent_weights,
        save_path=os.path.join("..", "results", "reasons_evaluation", "trajectory_evaluations.png")
    )

    # Use the best trajectory
    best_idx = eval_results['best_idx']
    trajectory_full = trajectories_full[best_idx]
    trajectory_full = trajectory_full[0]

    logger.info(
        f"Selected trajectory {best_idx} with human-centered score: {eval_results['best_evaluation']['total_score']:.3f}")
    logger.info(f"Policymaker: {eval_results['best_evaluation']['avg_scores']['policymaker']:.2f}, "
                f"Driver: {eval_results['best_evaluation']['avg_scores']['driver']:.2f}, "
                f"Cyclist: {eval_results['best_evaluation']['avg_scores']['cyclist']:.2f}")

    # Reset MPC with new trajectory
    traj_agent_idx = 0  # Reset trajectory index
    mpc = MPC(
        cx=trajectory_full[:, 0],
        cy=trajectory_full[:, 1],
        cyaw=trajectory_full[:, 2],
        dl=dl, speed=max_speed, dt=ScenarioParameters.DT, car_dimensions=car_dimensions
    )
    collision_xy = None  # Reset collision tracker

    # Store evaluation data for later analysis if needed
    scenario_obstacles.trajectory_evaluation = eval_results

    return collision_xy, mpc, traj_agent_idx, trajectory_full, scenario_obstacles


def create_following_trajectory(state, trajectories_full):
    # Add one trajectory if the ego keeps stay on its current lane following the cyclist
    # Pick one of the trajectories_full in order calculate completion_time
    resampled_trajectory = compute_predicted_trajectory(state, trajectories_full[0][0])
    completion_time = calculate_trajectory_completion_time(resampled_trajectory, state)
    # Get initial values
    init_x = resampled_trajectory[0, 0]  # Initial x value
    init_y = resampled_trajectory[0, 1]  # Initial y value
    init_theta = resampled_trajectory[0, 2]  # Initial theta value
    original_length = len(resampled_trajectory)
    # Create a copy of the trajectory to modify
    follow_trajectory = resampled_trajectory.copy()
    # Create an array of y-positions with proper length
    new_y_values = np.arange(
        init_y,
        init_y + (completion_time * state.v),
        (state.v * ScenarioParameters.DT)
    )
    # Ensure the array has the correct length
    if len(new_y_values) != original_length:
        # If too short, extend the array by repeating the last value
        if len(new_y_values) < original_length:
            new_y_values = np.append(
                new_y_values,
                np.repeat(new_y_values[-1], original_length - len(new_y_values))
            )
        # If too long, truncate the array
        else:
            new_y_values = new_y_values[:original_length]
    # Now assign the correctly sized array for y-coordinate (column 1)
    follow_trajectory[:, 1] = new_y_values
    # Keep x-coordinate (column 0) constant at the initial value
    follow_trajectory[:, 0] = init_x
    # Keep orientation (column 2) constant at the initial value
    follow_trajectory[:, 2] = init_theta
    return follow_trajectory


def visualize_trajectory_evaluations(eval_results, trajectories_full, moving_obstacles, state, car_dimensions,
                                     bicycle_dimensions, scenario, time_values,
                                     reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, agent_weights,
                                     save_path=None):
    """
    Visualize trajectory evaluations to compare different options using a 3×2 layout.

    Args:
        eval_results: Evaluation results from evaluate_trajectories_for_reasons
        trajectories_full: List of trajectories
        moving_obstacles: List of moving obstacles
        state: Current vehicle state
        car_dimensions: Vehicle dimensions
        bicycle_dimensions: Bicycle dimensions
        scenario: Scenario object containing environment information
        save_path: Optional path to save the visualization
    """
    if not trajectories_full:
        logger.warning("No trajectories to visualize")
        return

    # Create figure with GridSpec for 3×2 layout
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1.5, 1])

    # Create subplots - trajectory plot spans all rows in column 1
    ax_spatial = plt.subplot(gs[:, 0])  # Left column: Trajectories (spans all rows)
    ax_policy = plt.subplot(gs[0, 1])  # Top-right: Policy scores
    ax_driver = plt.subplot(gs[1, 1])  # Middle-right: Driver scores
    ax_cyclist = plt.subplot(gs[2, 1])  # Bottom-right: Cyclist scores

    # Set background color for spatial plot
    ax_spatial.set_facecolor('#AFABAB')

    # Colors for different trajectories with better visibility
    colors = ['#0000FF', '#000000', '#FF0000', '#BF00C0', '#00BFBF', '#8c564b']
    line_styles = ['--', '-.', '-.', '--', '-.', '--']  # Cycle through styles
    line_widths = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # Varying widths

    # Create trajectory styling with trajectory info and line properties
    trajectory_styles = []
    for i, trajectory in enumerate(trajectories_full):
        # For trajectories with tuple structure (from run_all)
        if isinstance(trajectory, tuple):
            trajectory_points = trajectory[0]
        else:
            trajectory_points = trajectory

        # Extract evaluation data
        eval_data = eval_results['all_evaluations'][i]
        score = eval_data['total_score']

        # Store trajectory info along with styling
        trajectory_styles.append({
            'trajectory': trajectory_points,
            'color': colors[i % len(colors)],
            'style': line_styles[i % len(line_styles)],
            'width': line_widths[i % len(line_widths)],
            'score': score,
            'index': i,
            'eval_data': eval_data
        })

    # Sort trajectory styles by line width (thickest first)
    trajectory_styles.sort(key=lambda x: x['width'], reverse=True)

    # Plot car at current state
    car_polygon = draw_car((state.x, state.y, state.yaw), car_dimensions, ax=ax_spatial, color='black')

    # Plot bicycle/moving obstacles
    for i, obstacle in enumerate(moving_obstacles):
        obstacle_x, obstacle_y, _, obstacle_yaw, _, _ = obstacle.get()
        draw_bicycle((obstacle_x, obstacle_y, obstacle_yaw), bicycle_dimensions, ax=ax_spatial, color='blue')

    # Draw all static elements from scenario
    draw_static_elements(ax_spatial, scenario)

    # Plot trajectories in order of line width (thickest first, thinnest last)
    for style_info in trajectory_styles:
        trajectory_points = style_info['trajectory']
        color = style_info['color']
        style = style_info['style']
        width = style_info['width']
        score = style_info['score']
        i = style_info['index']

        # Plot main trajectory line with simplified label
        ax_spatial.plot(
            trajectory_points[:, 0], trajectory_points[:, 1],
            color=color,
            linestyle=style,
            linewidth=width,
            label=f"Traj {i}: {score:.3f}"  # Simplified label
        )

        # Add title to the spatial plot
        priority_label = get_priority_label(agent_weights)
        weights_label = format_weights(agent_weights)
        ax_spatial.set_title(f"{priority_label} \n {weights_label}", fontsize=16, pad=10)

        # Set plot limits and aspect ratio
        ax_spatial.set_aspect('equal')
        ax_spatial.grid(True, alpha=0.3)  # Keep grid only for spatial plot

        # Add arrow to show direction
        mid_idx = len(trajectory_points) // 2
        if mid_idx > 0:
            arrow_start = trajectory_points[mid_idx - 1, :2]
            arrow_end = trajectory_points[mid_idx, :2]
            dx = arrow_end[0] - arrow_start[0]
            dy = arrow_end[1] - arrow_start[1]
            ax_spatial.arrow(
                arrow_start[0], arrow_start[1], dx, dy,
                head_width=0.3, head_length=0.6, fc=color, ec=color
            )

    # Set plot limits and aspect ratio
    ax_spatial.set_aspect('equal')
    ax_spatial.grid(True, alpha=0.3)  # Keep grid only for spatial plot

    # Set appropriate axis limits
    car_x, car_y = state.x, state.y
    ax_spatial.set_xlim(car_x - 20, car_x + 20)
    ax_spatial.set_ylim(car_y - 5, car_y + 52)

    # Trajectory legend inside the plot (to save space)
    # Sort legend entries by trajectory index for consistency
    handles, labels = ax_spatial.get_legend_handles_labels()
    # Extract trajectory numbers and sort by them
    traj_nums = [int(label.split(':')[0].split(' ')[1]) for label in labels]
    sorted_pairs = sorted(zip(handles, labels, traj_nums), key=lambda x: x[2])
    ax_spatial.legend([pair[0] for pair in sorted_pairs], [pair[1] for pair in sorted_pairs],
                      loc='upper right', fontsize=10)

    # Plot reason scores
    reason_plots = [
        (ax_policy, "Policymaker Reasons", 'policymaker'),
        (ax_driver, "Driver Reasons", 'driver'),
        (ax_cyclist, "Cyclist Reasons", 'cyclist_combined')
    ]

    # Plot each reason with consistent styling, but in order of line width
    for ax, title, score_key in reason_plots:
        ax.set_title(title, fontsize=16)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Time [s]", fontsize=14)
        ax.set_ylabel("Reasons [0-1]", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Remove grid (change #2)
        ax.grid(False)

        # Keep white background (change #2)
        # No gray background - removed the line: ax.axhspan(0.0, 1.0, alpha=0.1, color='gray')

        # Create style info for reason lines, but sorted by width
        reason_styles = []
        # Track minimum scores for this reason type
        min_score = 1.0  # Start with maximum possible (1.0)

        for style_info in trajectory_styles:
            i = style_info['index']
            scores = style_info['eval_data']['detailed_scores'][score_key]
            # Update minimum score if any value is lower
            if len(scores) > 0:
                min_score = min(min_score, np.min(scores))

            reason_styles.append({
                'scores': scores,
                'color': style_info['color'],
                'style': style_info['style'],
                'width': style_info['width'],
                'index': i,
                'completion_time': style_info['eval_data']['completion_time']
            })

        # Map of score keys to reason values lists
        reason_values_map = {
            'policymaker': reasons_policymaker_values,
            'driver': reasons_driver_values,
            'cyclist_combined': reasons_cyclist_values
        }

        # Check historical data for minimum as well
        if isinstance(reason_values_map[score_key], list) and len(reason_values_map[score_key]) > 0:
            min_score = min(min_score, min(reason_values_map[score_key]))

        # Set y-axis limits based on minimum score with some padding
        # Round down to nearest 0.1 to have clean axis limits
        min_y = max(0, math.floor(min_score * 10) / 10)  # Never go below 0
        ax.set_ylim(min_y - 0.1, 1.1)
        # Sort by line width (thickest first)
        reason_styles.sort(key=lambda x: x['width'], reverse=True)

        # Plot each trajectory's scores with consistent styling
        for style_info in reason_styles:
            color = style_info['color']
            style = style_info['style']
            width = style_info['width']
            scores = style_info['scores']
            i = style_info['index']
            completion_time = style_info['completion_time']

            time_values_ = copy.deepcopy(time_values)
            reasons_policymaker_values_ = copy.deepcopy(reasons_policymaker_values)
            reasons_driver_values_ = copy.deepcopy(reasons_driver_values)
            reasons_cyclist_values_ = copy.deepcopy(reasons_cyclist_values)

            # Map of score keys to reason values lists
            reason_values_map = {
                'policymaker': reasons_policymaker_values_,
                'driver': reasons_driver_values_,
                'cyclist_combined': reasons_cyclist_values_
            }

            # Calculate the range of historical data (before extension)
            if isinstance(time_values, list) and len(time_values) > 0:
                historical_start = time_values[0] if np.isscalar(time_values[0]) else time_values[0]
                historical_end = time_values[-1] if np.isscalar(time_values[-1]) else time_values[-1]

                # Add a colored background rectangle for historical data
                historical_rect = plt.Rectangle(
                    (historical_start, 0),
                    width=(historical_end - historical_start),
                    height=1.1,
                    color='#F7E2DA',
                    alpha=0.5,
                    zorder=0  # Ensure it's behind the plotted lines
                )
                ax.add_patch(historical_rect)

                # Optionally add a vertical line at the transition point
                ax.axvline(x=historical_end, color='#D3B1A6', linestyle='--', linewidth=1, alpha=0.7)

                # Add a label
                ax.text(
                    (historical_start + historical_end)/2,
                    1.05,
                    'Before replanner triggered',
                    ha='center',
                    va='center',
                    fontsize=8,
                    bbox=dict(facecolor='#F7E2DA', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='#D3B1A6')
                )

                # Plot the scores - accessing the specific list for this score_key
                ax.plot(
                    time_values_, reason_values_map[score_key],  # Access the list for this specific key
                    color='black')

            # If you need to keep track of all time points
            if isinstance(time_values_, list):
                # For the first call, time_values[-1] might be a number
                last_time = time_values_[-1] if np.isscalar(time_values_[-1]) else time_values_[-1]

                # Generate new time points with fixed 0.1s intervals
                start_time = last_time + ScenarioParameters.DT
                end_time = start_time + completion_time
                num_full_steps = int((end_time - start_time) / 0.1)  # Number of complete 0.1s intervals
                remaining_time = (end_time - start_time) - (num_full_steps * 0.1)  # Any remaining time

                # Generate the regular 0.1s intervals
                regular_points = np.array([start_time + i * 0.1 for i in range(num_full_steps)])

                # Add the final point if there's a remainder
                if remaining_time > 0:
                    time_points = np.append(regular_points, end_time)
                else:
                    time_points = regular_points

                # Ensure we have the right number of points
                if len(time_points) < len(scores):
                    # If we need more points, add intermediate points (this shouldn't normally happen)
                    missing_points = len(scores) - len(time_points)
                    additional_points = np.linspace(start_time, end_time, missing_points + 2)[1:-1]
                    time_points = np.sort(np.concatenate([time_points, additional_points]))
                elif len(time_points) > len(scores):
                    # If we have too many points, truncate
                    time_points = time_points[:len(scores)]

            #     # Use extend to append individual values (not the whole array)
            #     time_values_.extend(time_points)
            #
            #     # Use these new time points for plotting the current trajectory
            #     plot_times = time_points
            # else:
            #     # If time_values is not a list, create new time points from 0
            #     time_points = np.linspace(0, completion_time, len(scores))
            #     plot_times = time_points
            #
            # # Extend the appropriate reason values list
            # if isinstance(reason_values_map[score_key], list):
            #     reason_values_map[score_key].extend(scores)  # Extend with individual values
            # else:
            #     reason_values_map[score_key] = list(scores)  # Convert to list if not already

            # Plot the scores - accessing the specific list for this score_key
            ax.plot(
                time_points, scores,  # Access the list for this specific key
                color=color, linestyle=style, linewidth=width
            )


        # Remove safety threshold line (change #1)
        # Removed the code: ax.axhline(y=0.3, color='r', linestyle=':', linewidth=2, label='Safety Threshold')

    # Add a common legend to the first plot only (top-right)
    # Sort legend entries by trajectory index for consistency
    handles, labels = ax_policy.get_legend_handles_labels()

    # Sort by trajectory number
    traj_labels = [l for l in labels if l.startswith('Traj')]
    if traj_labels:
        # Extract trajectory numbers and sort by them
        traj_nums = [int(label.split(' ')[1]) for label in traj_labels]
        sorted_idx = sorted(range(len(traj_nums)), key=lambda i: traj_nums[i])

        sorted_handles = [handles[i] for i in sorted_idx]
        sorted_labels = [labels[i] for i in sorted_idx]

        ax_policy.legend(sorted_handles, sorted_labels,
                         loc='upper center', ncol=1, fontsize=9, bbox_to_anchor=(0.2, 0.6))

    # Remove redundant legends from other plots
    if ax_driver.get_legend():
        ax_driver.get_legend().remove()
    if ax_cyclist.get_legend():
        ax_cyclist.get_legend().remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=800, bbox_inches='tight')

    plt.show()

def get_priority_label(agent_weights):
    """
    Return a human-readable label based on the agent weights.

    Args:
        agent_weights: Dictionary containing weights for different agents (policymaker, driver, cyclist)

    Returns:
        str: A descriptive label indicating the priority focus
    """
    # Extract weights from the dictionary
    policy_weight = agent_weights.get('policymaker', 0.0)
    driver_weight = agent_weights.get('driver', 0.0)
    cyclist_weight = agent_weights.get('cyclist', 0.0)

    # Determine the priority based on the highest weight
    max_weight = max(policy_weight, driver_weight, cyclist_weight)

    # Small threshold for considering weights approximately equal
    threshold = 0.05

    if max_weight == driver_weight and driver_weight >= 0.5:
        return "Trajectory Scoring Based on Human Reasons \n (Driver Reasons Weighted Higher)"
    elif max_weight == policy_weight and policy_weight >= 0.5:
        return "Trajectory Scoring Based on Human Reasons \n (Policymaker Reasons Weighted Higher)"
    elif max_weight == cyclist_weight and cyclist_weight >= 0.5:
        return "Trajectory Scoring Based on Human Reasons \n (Cyclist Reasons Weighted Higher)"
    elif abs(driver_weight - policy_weight) < threshold and abs(driver_weight - cyclist_weight) < threshold:
        return "Trajectory Scoring Based on Human Reasons \n (All Reasons Weighted The Same)"
    else:
        return "Mixed Priority"

def format_weights(agent_weights):
    """
    Format agent weights as a concise string.

    Args:
        agent_weights: Dictionary containing weights for different agents (policymaker, driver, cyclist)

    Returns:
        str: A formatted string representation of the weights
    """
    # Extract weights from the dictionary
    policy_weight = agent_weights.get('policymaker', 0.0)
    driver_weight = agent_weights.get('driver', 0.0)
    cyclist_weight = agent_weights.get('cyclist', 0.0)

    # Format with D for driver, P for policy, C for cyclist
    return f"Driver:{driver_weight:.2f}; Policymaker:{policy_weight:.2f}; Cyclist:{cyclist_weight:.2f}"

def evaluate_trajectories_for_reasons(trajectories_full, moving_obstacles, state, car_dimensions, bicycle_dimensions,
                                      reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                                      time_elapsed_driver=0.0, time_passed_cyclist=0.0):
    """
    Evaluate multiple trajectories based on human-centered reasons.

    Args:
        trajectories_full: List of candidate trajectories
        moving_obstacles: List of moving obstacles (bicycle)
        state: Current state of the vehicle
        car_dimensions: Dimensions of the car
        bicycle_dimensions: Dimensions of the bicycle

    Returns:
        dict: Evaluation results with scores and best trajectory
    """
    trajectory_scores = []
    detailed_evaluations = []

    # For each trajectory option
    for i, trajectory in enumerate(trajectories_full):
        # 1. Calculate time needed to complete this trajectory
        # For the last trajectory, we need to calculate resampled_trajectory differently
        if i == len(trajectories_full) - 1:
            resampled_trajectory = compute_predicted_trajectory(state, trajectory[0],last_index=True)
            # set completion time as the real completion time from other trajectories
            # completion_time = calculate_trajectory_completion_time(resampled_trajectory, state, last_index=True)
        else:
            resampled_trajectory = compute_predicted_trajectory(state, trajectory[0])
            completion_time = calculate_trajectory_completion_time(resampled_trajectory, state)

        # 2. Predict bicycle movement for this duration
        bicycle_future_trajectory = np.vstack(MovingObstaclesPrediction(
            *moving_obstacles[0].get(),
            sample_time=ScenarioParameters.DT,
            car_dimensions=bicycle_dimensions
        ).state_prediction(completion_time)).T

        # 3. Sample points along both trajectories
        num_sample_points = len(resampled_trajectory)

        # Create evenly spaced indices
        ego_indices = np.linspace(0, len(resampled_trajectory) - 1, num_sample_points, dtype=int)
        # Delete the last point of bicycle_future_trajectory
        bicycle_indices = np.linspace(0, len(bicycle_future_trajectory[:-1]) - 1, num_sample_points, dtype=int)

        # 4. Evaluate reasons at each sample point
        current_time_elapsed_driver = time_elapsed_driver
        current_time_passed_cyclist = time_passed_cyclist

        policymaker_scores = []
        driver_scores = []
        cyclist_comfort_scores = []
        cyclist_time_scores = []
        cyclist_combined_scores = []

        # Width of the car for centerline evaluation
        car_width = car_dimensions.bounding_box_size[0]

        for j in range(num_sample_points):
            # Get ego vehicle state at this point
            ego_x = resampled_trajectory[ego_indices[j], 0]
            ego_y = resampled_trajectory[ego_indices[j], 1]
            ego_theta = resampled_trajectory[ego_indices[j], 2]

            # Get bicycle state at this point
            bicycle_x = bicycle_future_trajectory[bicycle_indices[j], 0]
            bicycle_y = bicycle_future_trajectory[bicycle_indices[j], 1]

            # Create a simulated state object for evaluation
            simulated_state = State(x=ego_x, y=ego_y, yaw=ego_theta, v=state.v)

            # Create a simulated obstacle object for evaluation
            simulated_obstacle = type('obj', (object,), {
                'get': lambda self=None: (bicycle_x, bicycle_y, 0, 0, CyclistParameters.SPEED, 0)
            })
            simulated_obstacles = [simulated_obstacle]

            # Evaluate policymaker (regulatory compliance)
            policymaker_score = evaluate_distance_to_centerline(
                ego_x, car_width, ScenarioParameters.CENTERLINE_LOCATION)
            policymaker_scores.append(policymaker_score)

            # Evaluate driver time efficiency
            driver_score, current_time_elapsed_driver = evaluate_time_following(
                'driver_reasons', ScenarioParameters.DT,
                DriverParameters.DISTANCE_BUFFER, DriverParameters.DISTANCE_REF,
                DriverParameters.TIME_THRESHOLD, simulated_obstacles,
                simulated_state, current_time_elapsed_driver)
            driver_scores.append(driver_score)

            # Evaluate cyclist comfort (distance)
            cyclist_comfort_score = evaluate_distance_to_obstacle(
                CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF,
                simulated_obstacles, simulated_state)
            cyclist_comfort_scores.append(cyclist_comfort_score)

            # Evaluate cyclist comfort (time)
            cyclist_time_score, current_time_passed_cyclist = evaluate_time_following(
                'cyclist_reasons', ScenarioParameters.DT,
                CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF,
                CyclistParameters.TIME_THRESHOLD, simulated_obstacles,
                simulated_state, current_time_passed_cyclist)
            cyclist_time_scores.append(cyclist_time_score)

            # Combined cyclist score
            cyclist_combined_score = cyclist_comfort_score * cyclist_time_score
            cyclist_combined_scores.append(cyclist_combined_score)

        #delete last value of scores because the ego vehicle is not moving but the bicycle is
        policymaker_scores = policymaker_scores[:-1]
        driver_scores = driver_scores[:-1]
        cyclist_combined_scores = cyclist_combined_scores[:-1]

        #replace first value of reasons scores with the first value of reasons evaluation
        policymaker_scores[0] = reasons_policymaker_reg_compliance
        driver_scores[0] = reasons_driver_time_eff
        cyclist_combined_scores[0] = reasons_cyclist_comfort

        # 5. Calculate average scores for each reason
        avg_policymaker = np.mean(policymaker_scores[:-1])
        avg_driver = np.mean(driver_scores)
        avg_cyclist = np.mean(cyclist_combined_scores)

        # 6. Calculate weighted total score
        # Define weights for different agents (matching your existing weights)
        agent_weights = {
            'policymaker': 0.33,  # Regulatory compliance
            'driver': 0.33,  # Driver patience/efficiency
            'cyclist': 0.34  # Cyclist comfort/safety
        }

        total_score = (
                agent_weights['policymaker'] * avg_policymaker +
                agent_weights['driver'] * avg_driver +
                agent_weights['cyclist'] * avg_cyclist
        )

        # Store results
        trajectory_scores.append(total_score)

        detailed_eval = {
            'trajectory_idx': i,
            'total_score': total_score,
            'completion_time': completion_time,
            'avg_scores': {
                'policymaker': avg_policymaker,
                'driver': avg_driver,
                'cyclist': avg_cyclist
            },
            'detailed_scores': {
                'policymaker': policymaker_scores,
                'driver': driver_scores,
                'cyclist_comfort': cyclist_comfort_scores,
                'cyclist_time': cyclist_time_scores,
                'cyclist_combined': cyclist_combined_scores
            }
        }

        detailed_evaluations.append(detailed_eval)

        logger.info(f"Trajectory {i}: Score {total_score:.3f} (P: {avg_policymaker:.2f}, "
                    f"D: {avg_driver:.2f}, C: {avg_cyclist:.2f}) "
                    f"Time: {completion_time:.2f}s")

    # 7. Find the best trajectory
    if trajectory_scores:
        best_idx = np.argmax(trajectory_scores)
        best_score = trajectory_scores[best_idx]
        best_trajectory = trajectories_full[best_idx]
        best_evaluation = detailed_evaluations[best_idx]

        logger.info(f"Selected trajectory {best_idx} with score {best_score:.3f}")
    else:
        best_idx = None
        best_trajectory = None
        best_evaluation = None
        logger.warning("No trajectories to evaluate")

    return agent_weights, {
        'scores': trajectory_scores,
        'best_idx': best_idx,
        'best_trajectory': best_trajectory,
        'best_evaluation': best_evaluation,
        'all_evaluations': detailed_evaluations
    }

def calculate_trajectory_completion_time(trajectory_res, state, last_index=None):
    """
    Calculate the estimated time to complete a trajectory based on vehicle dynamics.

    Args:
        trajectory_res: The trajectory to analyze
        state: The current state of the vehicle

    Returns:
        float: Estimated time to complete the trajectory in seconds
    """
    # Start with current velocity
    current_velocity = state.v

    # Initialize time counter
    total_time = 0.0

    # Calculate distance between consecutive points
    if len(trajectory_res) <= 1:
        return 0.0

    distances = []
    for i in range(1, len(trajectory_res)):
        # Calculate Euclidean distance between consecutive points
        distance = np.linalg.norm(trajectory_res[i, :2] - trajectory_res[i - 1, :2])
        distances.append(distance)

    # Simulate movement along the trajectory
    for distance in distances:
        # Update velocity based on acceleration
        if last_index == None:
            current_velocity = min(current_velocity + MAX_ACCEL, Simulation.MAX_SPEED)
        else:
            current_velocity = current_velocity
        # Calculate time needed for this segment
        segment_time = distance / current_velocity
        total_time += segment_time

    return total_time

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
    spawn_location_y = scenario_no_obstacles.start[1] + 9.8
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
    # ax.set_xlim((-10, 10))
    ax.set_ylim((-30, 30))

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
    # frame_files = sorted([f for f in os.listdir(results_folder) if f.startswith('frame_') and f.endswith('.jpg')])
    # for frame_file in frame_files[-30:]:
    #     os.remove(os.path.join(results_folder, frame_file))

    # Run the ffmpeg command to create a video from frames
    subprocess.run([
        'ffmpeg', '-framerate', '10', '-i', 'frame_%04d.jpg',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'output_video.mp4'
    ])

