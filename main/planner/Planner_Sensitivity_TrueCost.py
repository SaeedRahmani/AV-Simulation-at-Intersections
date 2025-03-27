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
    wc_dist = [0, 1] 
    wc_steering = [0, 10]

    parameter_combinations = list(product(wc_dist, wc_steering))

    for index, (wc_dist, wc_steering) in enumerate(parameter_combinations):
        search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius,
                                       wc_dist=wc_dist, wc_steering=wc_steering)

        runtime, (cost, path, trajectory) = run_search(search)
        cx = trajectory[:, 0]
        cy = trajectory[:, 1]

        # Determine line style and color
        line_style = '--' if wc_steering == 0 else '-'
        color = 'red' if wc_dist == 0 else 'blue'
        
        label_text = f'$W_{{c,d}}: {wc_dist}, W_{{c,\\phi}}: {wc_steering}$'
        ax.plot(cx, cy, label=label_text, linestyle=line_style, color=color)

    draw_scenario(scenario, mps, car_dimensions, search, ax, draw_obstacles=False, draw_goal=True, 
                  draw_car=True, draw_mps=False, draw_collision_checking=False, draw_car2=False, draw_mps2=False, 
                  mp_name='right1')

    plt.legend(loc='lower left')
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
