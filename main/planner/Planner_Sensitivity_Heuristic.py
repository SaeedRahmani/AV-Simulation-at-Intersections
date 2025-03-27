from matplotlib import pyplot as plt
import sys
import time
from itertools import product
sys.path.append('..')

from envs.intersection_multi_lanes import intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.motion_primitive import load_motion_primitives
from lib.mp_search_ww_generic import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points

def measure_time(func):
    """Decorator to measure runtime of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result
    return wrapper

@measure_time
def run_search(search):
    return search.run(debug=True)

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(6, 6))

    start_pos = 1
    turn_indicator = 1
    scenario = intersection(turn_indicator=turn_indicator, start_pos=start_pos, 
                           start_lane=1, goal_lane=1, number_of_lanes=2)
    
    mps = load_motion_primitives(version='bicycle_model')
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    # Parameter values
    wh_dist = [1, 2] 
    wh_theta = [0, 2.7]
    wh_steering = [0, 15]
    parameter_combinations = list(product(wh_dist, wh_theta, wh_steering))

    # Define colors and line styles
    #colors = {1: 'red', 2: 'blue'}
    #line_styles = {2.7: '-', 3.1: '--', 3.5: '-.'}

    for index, (wh_dist, wh_theta, wh_steering) in enumerate(parameter_combinations):
        search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius,
                                       wh_dist=wh_dist, wh_theta=wh_theta, wh_steering=wh_steering)

        runtime, (cost, path, trajectory) = run_search(search)
        cx = trajectory[:, 0]
        cy = trajectory[:, 1]
        line_style = '-'
        line_width = 1
        line_color = 'red'
        # Determine line style and color
        #color = colors[wh_dist]
        #line_style = line_styles[wh_theta]
        
        if wh_dist == 1:
            line_width = 1
            if wh_theta == 0:
                line_width = 2
                if wh_steering == 0:
                    line_width = 3
                    line_color = 'blue'
                else:
                    line_width = 2
                    line_color = 'red'
            elif wh_steering == 0:
                line_width = 1
                line_color = 'green'
            else:
                line_width = 1
                line_color = 'orange'
                line_style = ':'
        else:
            line_width = 1
            if wh_theta == 0:
                line_color = 'black'
                line_width = 2
                if wh_steering == 0:
                    line_style = '--'
            elif wh_steering == 0:
                line_color = 'blue'
            else:
                line_color = 'orange'
                line_style = '-.'
            
        
        # Line color
        # if wh_dist == 1:
        #     line_width = 1
        #     if wh_theta == 0:
        #         line_width = 2
        #         if wh_steering == 0:
        #             line_width = 3
        # else:
        #     line_width = 0.5
        
        # # Line width
        # if wh_theta == 0:
        #     line_color = 'blue'
        # else:
        #     line_color = 'red'
        
        # # Line style
        # if wh_dist != 1 & wh_steering == 0:
        #     line_style = '-.'
        # elif wh_dist != 1 & wh_steering != 0:
        #     line_style = '--'
        
        label_text = f'$W_{{h,d}}: {wh_dist}, W_{{h,\\theta}}: {wh_theta}, W_{{h,\\phi}}: {wh_steering}, \\mathrm{{Time}}: {runtime:.2f}\\,\\mathrm{{s}}$'
        #label_text = f'W_Distance: {wh_dist}, W_Theta: {wh_theta}, W_Steering: {wh_steering}, Time: {runtime:.2f}s'
        ax.plot(cx, cy, color=line_color, label=label_text, linestyle=line_style, linewidth=line_width)

    draw_scenario(scenario, mps, car_dimensions, search, ax, draw_obstacles=False, draw_goal=True, 
                  draw_car=True, draw_mps=False, draw_collision_checking=False, draw_car2=False, draw_mps2=False, 
                  mp_name='right1')

    plt.legend(loc='lower left')
    ax.axis('equal')
    plt.tight_layout()
    plt.show()