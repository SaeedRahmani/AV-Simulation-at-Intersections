from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import sys
import numpy as np
from matplotlib.patches import Rectangle
sys.path.append('..')

from envs.intersection_multi_lanes import intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.helpers import measure_time
from lib.motion_primitive import load_motion_primitives
from lib.plotting import draw_scenario, draw_astar_search_points

# Import directly from the file instead of the lib folder
from lib.mp_search_reasoning import MotionPrimitiveSearch

if __name__ == '__main__':
    # Create figure with enough space for parameter table
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid spec to have main plot and parameter display
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0, :3])  # Main plot takes 3/4 of the width
    param_ax = fig.add_subplot(gs[0, 3])  # Parameter display takes 1/4 of the width
    param_ax.axis('off')  # Hide axes for parameter display
            
    # Scenario Parameters
    start_pos = 1
    turn_indicator = 2  # 0: straight, 1: left, 2: right
    
    # Load motion primitives
    version = 'bicycle_model'
    mps = load_motion_primitives(version=version)
    
    # Create intersection scenario
    scenario = intersection(
        turn_indicator=turn_indicator,
        start_pos=start_pos, 
        start_lane=1,
        goal_lane=2,
        number_of_lanes=3)
    
    # Create car dimensions
    car_dimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    # Define reasoning weight combinations to test - use smaller sets for debugging
    ego_weights = [1.0, 2.0]
    policy_weights = [2.7]  # Using the value that worked in the original
    rUser1_weights = [15.0]  # Using the value that worked in the original
    
    # Define low-level weights
    wh_dist2goal = 1.0
    wh_theta2goal = 2.7
    wh_steer2goal = 15.0
    wh_dist2obs = 0.1
    wh_dist2center = 0.0
    
    # Define cost weights
    wc_dist = 1.0
    wc_steering = 5.0
    wc_obstacle = 0.1
    wc_center = 0.0
    
    print("Creating search object...")
    # Create the reasoning-based motion primitive search
    search = MotionPrimitiveSearch(
        scenario, 
        car_dimensions, 
        mps, 
        margin=car_dimensions.radius,
        # High-level reasoning weights to test
        wh_ego=ego_weights,
        wh_policy=policy_weights,
        wh_rUser1=rUser1_weights,
        # Low-level weights - use the values that worked in the original
        wh_dist2goal=wh_dist2goal,
        wh_theta2goal=wh_theta2goal,
        wh_steer2goal=wh_steer2goal,
        wh_dist2obs=wh_dist2obs,
        wh_dist2center=wh_dist2center,
        # Cost weights
        wc_dist=wc_dist,
        wc_steering=wc_steering,
        wc_obstacle=wc_obstacle,
        wc_center=wc_center
    )

    print("Drawing scenario...")
    # Draw the scenario first
    draw_scenario(scenario, mps, car_dimensions, search, ax,
                  draw_obstacles=True, draw_goal=True, draw_car=True, draw_mps=False, 
                  draw_collision_checking=False, draw_car2=False, draw_mps2=False)

    # Run a single search first to verify it works
    print("Running single search for verification...")
    try:
        # Use the first set of weights
        search._current_wh_ego = ego_weights[0]
        search._current_wh_policy = policy_weights[0]
        search._current_wh_rUser1 = rUser1_weights[0]
        
        # Run the search with debug on
        cost, path, trajectory = search.run(debug=True)
        
        # Plot the single trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], color='b', linewidth=2, 
                label=f'Path (ego={ego_weights[0]}, policy={policy_weights[0]}, rUser1={rUser1_weights[0]})')
        
        # Draw search points to verify
        sc = draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)
        cbar = plt.colorbar(sc, ax=ax, label='Heuristic Cost', pad=0.01, fraction=0.046)
        
        print(f"Single search successful! Cost: {cost:.2f}")
        
        # Now run all combinations if the single search worked
        print("Running search with all weight combinations...")
        solutions = []
        
        # Manual implementation of running all combinations to better control the process
        for ego_w in ego_weights:
            for policy_w in policy_weights:
                for rUser1_w in rUser1_weights:
                    print(f"Testing weights: ego={ego_w}, policy={policy_w}, rUser1={rUser1_w}")
                    
                    # Set current weights
                    search._current_wh_ego = ego_w
                    search._current_wh_policy = policy_w
                    search._current_wh_rUser1 = rUser1_w
                    
                    # Run search with these weights
                    try:
                        cost, path, trajectory = search.run(debug=False)
                        solutions.append((trajectory, (ego_w, policy_w, rUser1_w, 0, 0), cost))
                        print(f"  -> Success! Cost: {cost:.2f}")
                    except Exception as e:
                        print(f"  -> Failed with error: {e}")
                        continue
        
        # Plot all trajectories if we found multiple solutions
        if len(solutions) > 1:
            # Plot each trajectory with a different color
            colors = plt.cm.viridis(np.linspace(0, 1, len(solutions)))
            
            for i, (trajectory, weights, cost) in enumerate(solutions):
                if i == 0:  # Skip the first one as we've already plotted it
                    continue
                    
                ego_w, policy_w, rUser1_w, _, _ = weights
                label = f'Path (ego={ego_w}, policy={policy_w}, rUser1={rUser1_w}), cost={cost:.2f}'
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], linewidth=2, label=label)
        
        print(f"Found {len(solutions)} different paths with various weight combinations")
    
    except KeyboardInterrupt:
        print("Search interrupted by user")
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()

    # Create legend for the main plot
    marker_size = 10
    handles = []
    
    # Add custom legend elements
    handles.append(mlines.Line2D([], [], color=(1, 0.8, 0.8), marker='s', ls='', label='Goal area', markersize=marker_size))
    handles.append(mlines.Line2D([], [], color='r', marker='$\u279C$', ls='', label='Goal direction', markersize=marker_size))
    handles.append(mlines.Line2D([], [], color='b', marker='s', ls='', label='Obstacles', markersize=marker_size))
    handles.append(mlines.Line2D([], [], color='g', marker='s', ls='', label='Ego vehicle', markersize=marker_size, fillstyle='none'))
    handles.append(mlines.Line2D([], [], color=(47/255, 108/255, 144/255), marker='.', ls='', label='Visited points A*', markersize=marker_size))
    
    # Add the legend - keep it smaller to not crowd the figure
    ax.legend(handles=handles, loc='best')

    # Display parameter info on the right side
    param_text = "Motion Primitive Search Parameters\n"
    param_text += "=" * 35 + "\n\n"
    
    # High-level weights 
    param_text += "HIGH-LEVEL WEIGHTS:\n"
    param_text += "-" * 25 + "\n"
    param_text += f"Ego weights:     {ego_weights}\n"
    param_text += f"Policy weights:  {policy_weights}\n"
    param_text += f"rUser1 weights:  {rUser1_weights}\n\n"
    
    # Low-level weights
    param_text += "LOW-LEVEL WEIGHTS:\n"
    param_text += "-" * 25 + "\n"
    param_text += f"Distance to goal:     {wh_dist2goal}\n"
    param_text += f"Theta to goal:        {wh_theta2goal}\n"
    param_text += f"Steering to goal:     {wh_steer2goal}\n"
    param_text += f"Distance to obstacle: {wh_dist2obs}\n"
    param_text += f"Distance to center:   {wh_dist2center}\n\n"
    
    # Cost weights
    param_text += "COST WEIGHTS:\n"
    param_text += "-" * 25 + "\n"
    param_text += f"Distance:        {wc_dist}\n"
    param_text += f"Steering:        {wc_steering}\n"
    param_text += f"Obstacle:        {wc_obstacle}\n"
    param_text += f"Center:          {wc_center}\n\n"
    
    # Execution info
    param_text += "EXECUTION INFO:\n"
    param_text += "-" * 25 + "\n"
    param_text += f"Scenario:        {'Right' if turn_indicator==2 else 'Left' if turn_indicator==1 else 'Straight'} Turn\n"
    param_text += f"Start position:  {start_pos}\n"
    param_text += f"Car model:       {version}\n"
    param_text += f"Solutions found: {len(solutions)}\n"
    
    # Add the parameter text
    param_ax.text(0.05, 0.95, param_text, fontfamily='monospace', 
                 verticalalignment='top', horizontalalignment='left',
                 transform=param_ax.transAxes, fontsize=10)

    # Add a background box for the parameter text
    param_ax.add_patch(Rectangle((0.03, 0.03), 0.94, 0.94, fill=True, 
                               color='whitesmoke', alpha=0.8, transform=param_ax.transAxes))
    param_ax.text(0.05, 0.95, param_text, fontfamily='monospace', 
                 verticalalignment='top', horizontalalignment='left',
                 transform=param_ax.transAxes, fontsize=10)

    ax.set_title("Motion Primitive Search with Reasoning Weights")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.axis('equal')
    plt.tight_layout()
    plt.show()