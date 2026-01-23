import sys

import numpy as np

from lib.obstacles import BoxObstacle, CircleObstacle
from lib.scenario import Scenario


def t_intersection(no_obstacles=False, turn_left=True) -> Scenario:
    """ Function the build a T-intersection. Takes the gym environment as input. """
    # The intersection was originally build in urdf_envs which applies a scale factor on the pruis.urdf model
    # so to make the intersection usable for a normally sized prius, we should divide by the scale factor
    width_road = 4  # https://www.sddc.army.mil/sites/TEA/Functions/SpecialAssistant/TrafficEngineeringBranch/BMTE/calcIntersections/intersectionsTurtorials/intersectionDesign/Pages/standardLaneWidth.aspx
    width_traffic_island = 2  # arbitrary
    width_pavement = 5  # arbitrary
    length = 30  # arbitrary
    height = 0.5  # irrelevant, height of obst in pybullet
    corner_radius = 6  # https://streetsillustrated.seattle.gov/design-standards/intersections/
    distance_center = corner_radius + width_road + width_traffic_island
    flip_start_position = False
    flip_goal_position = False
    allowed_goal_theta_difference = np.pi / 16

    if flip_goal_position and not flip_start_position:
        print("The goal position is flipped and start is not. This will probably take a bit longer to compute.",
              file=sys.stderr)
    elif flip_start_position and not flip_goal_position:
        print("The start position is flipped and goal is not. "
              "This will take REALLY REALLY REALLY long to compute.\n"
              "It is not worth it.",
              file=sys.stderr)

    start = (width_traffic_island / 2 + width_road / 2, -30, -0.5 * np.pi if flip_start_position else 0.5 * np.pi)

    if turn_left:
        flip_goal_position = False
        goal = (
            -(distance_center + length * 0.6), (width_traffic_island + width_road) / 2,
            -(1 - flip_goal_position) * np.pi)
    else:
        flip_goal_position = True
        goal = (
            (distance_center + length * 0.6), -(width_traffic_island + width_road) / 2,
            -(1 - flip_goal_position) * np.pi)

    goal_area = BoxObstacle(xy_width=(width_road * 1.8, width_road), height=height,
                            xy_center=(goal[0], goal[1]))

    if no_obstacles:
        obstacles = []
    else:
        obstacles = [
            # T-intersection
            # Leg of T
            BoxObstacle(xy_width=(width_traffic_island, length), height=height,
                        xy_center=(0, -(length / 2 + distance_center))),
            BoxObstacle(xy_width=(width_pavement, length), height=height,
                        xy_center=(
                            (width_traffic_island / 2 + width_road + width_pavement / 2),
                            -(length / 2 + distance_center))),
            BoxObstacle(xy_width=(width_pavement, length), height=height,
                        xy_center=(
                            -(width_traffic_island / 2 + width_road + width_pavement / 2),
                            -(length / 2 + distance_center))),
            CircleObstacle(radius=(width_traffic_island / 2), height=height,
                           xy_center=(0, -distance_center)),

            # left part of T
            BoxObstacle(xy_width=(length, width_traffic_island), height=height,
                        xy_center=(-(length / 2 + distance_center), 0.)),
            BoxObstacle(xy_width=(length, width_pavement), height=height,
                        xy_center=(
                            -(length / 2 + distance_center),
                            -(width_traffic_island / 2 + width_road + width_pavement / 2))),
            CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                           xy_center=(-distance_center, -distance_center)),
            CircleObstacle(radius=(width_traffic_island / 2), height=height,
                           xy_center=(-distance_center, 0)),

            # right part of T
            BoxObstacle(xy_width=(length, width_traffic_island), height=height,
                        xy_center=((length / 2 + distance_center), 0)),
            BoxObstacle(xy_width=(length, width_pavement), height=height,
                        xy_center=(
                            (length / 2 + distance_center),
                            -(width_traffic_island / 2 + width_road + width_pavement / 2))),
            CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                           xy_center=(distance_center, -distance_center)),
            CircleObstacle(radius=(width_traffic_island / 2), height=height,
                           xy_center=(distance_center, 0)),

            # upper part of T
            BoxObstacle(xy_width=((2 * length + 2 * distance_center), width_pavement), height=height,
                        xy_center=(0, (width_traffic_island / 2 + width_road + width_pavement / 2))),

            # Prevent the MP from looking in the wrong lanes / violating traffic rules
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=(-(length / 2 + distance_center), - (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=((length / 2 + distance_center), (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=(-(width_road + width_traffic_island) / 2, -(length / 2 + distance_center)),
                        hidden=True),
        ]

    return Scenario(
        start=start,
        goal_point=goal,
        goal_area=goal_area,
        allowed_goal_theta_difference=allowed_goal_theta_difference,
        obstacles=obstacles,
    )
