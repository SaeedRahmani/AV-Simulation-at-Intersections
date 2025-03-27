#!/usr/bin/env python3
'''
This is a more advanced version of the mp_search_ww_generic.py, which allows for 
searching over multiple weights for the heuristic/true cost functions. 
'''
import sys
sys.path.append('..')

from itertools import product
from typing import Dict, Tuple, List, Iterable
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

from envs.intersection_multi_lanes import intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.helpers import measure_time
from lib.motion_primitive import load_motion_primitives
from lib.plotting import draw_scenario
from lib.a_star import AStar
from lib.linalg import create_2d_transform_mtx, transform_2d_pts
from lib.maths import normalize_angle
from lib.motion_primitive import MotionPrimitive
from lib.obstacles import check_collision
from lib.scenario import Scenario
from lib.trajectories import car_trajectory_to_collision_point_trajectories, resample_curve

NodeType = Tuple[float, float, float]

# Your modified class (heuristic weights as lists)
class MotionPrimitiveSearch:
    def __init__(self, scenario: Scenario, car_dimensions: CarDimensions, mps: Dict[str, MotionPrimitive],
                 margin: float,
                 wh_dist: List[float] = None, wh_theta: List[float] = None, wh_steering: List[float] = None,
                 wh_obstacle: List[float] = None, wh_center: List[float] = None,
                 wc_dist: float = 1.0, wc_steering: float = 5.0, wc_obstacle: float = 0.1, wc_center: float = 0.0):

        self._mps = mps
        self._car_dimensions = car_dimensions
        self._points_to_mp_names: Dict[Tuple[NodeType, NodeType], str] = {}

        self._start = scenario.start
        self._goal_area = scenario.goal_area
        self._goal_point = scenario.goal_point
        self._allowed_goal_theta_difference = scenario.allowed_goal_theta_difference
        self._obstacles_hp: List[np.ndarray] = [o.to_convex(margin=margin) for o in scenario.obstacles]
        self._gx, self._gy, self._gtheta = scenario.goal_point

        self._a_star: AStar[NodeType] = AStar(neighbor_function=self.neighbor_function)

        self._wh_dist_list = wh_dist if wh_dist else [1.0]
        self._wh_theta_list = wh_theta if wh_theta else [2.7]
        self._wh_steering_list = wh_steering if wh_steering else [15.0]
        self._wh_obstacle_list = wh_obstacle if wh_obstacle else [0.0]
        self._wh_center_list = wh_center if wh_center else [0.0]

        self._wc_dist = wc_dist
        self._wc_steering = wc_steering
        self._wc_obstacle = wc_obstacle
        self._wc_center = wc_center

        self._mp_collision_points: Dict[str, np.ndarray] = self._create_collision_points()

        self._current_wh_dist = None
        self._current_wh_theta = None
        self._current_wh_steering = None
        self._current_wh_obstacle = None
        self._current_wh_center = None

    def _create_collision_points(self) -> Dict[str, np.ndarray]:
        MIN_DISTANCE_BETWEEN_POINTS = self._car_dimensions.radius
        out: Dict[str, np.ndarray] = {}
        for mp_name, mp in self._mps.items():
            points = mp.points.copy()
            points = resample_curve(points, dl=MIN_DISTANCE_BETWEEN_POINTS, keep_last_point=True)
            cc_trajectories = car_trajectory_to_collision_point_trajectories(points, self._car_dimensions)
            out[mp_name] = np.concatenate(cc_trajectories, axis=0)
        return out

    def calculate_steering_change_cost(self, current_node: NodeType, next_node: NodeType, steering_angle_weight: float = 1.0) -> float:
        _, _, current_theta = current_node
        _, _, next_theta = next_node
        orientation_change = next_theta - current_theta
        orientation_change = (orientation_change + np.pi) % (2 * np.pi) - np.pi
        steering_change_cost = abs(orientation_change) * steering_angle_weight
        return steering_change_cost

    def calculate_distance_point_to_halfplane(self, point: Tuple[float, float], half_planes: np.ndarray) -> float:
        x0, y0 = point
        distances = []
        for a, b, c in half_planes:
            distance = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
            distances.append(distance)
        return min(distances)

    def distance_to_nearest_obstacle(self, node: NodeType) -> float:
        x, y, _ = node
        return min(self.calculate_distance_point_to_halfplane((x, y), hp) for hp in self._obstacles_hp)

    def run(self, debug=False):
        cost, path = self._a_star.run(self._start, is_goal_function=self.is_goal,
                                      heuristic_function=self.distance_to_goal, debug=debug)
        trajectory = self.path_to_full_trajectory(path)
        return cost, path, trajectory

    def run_all(self, debug=False):
        trajectories = []
        combinations = product(self._wh_dist_list, self._wh_theta_list, self._wh_steering_list,
                               self._wh_obstacle_list, self._wh_center_list)
        for wh_dist, wh_theta, wh_steering, wh_obstacle, wh_center in combinations:
            self._current_wh_dist = wh_dist
            self._current_wh_theta = wh_theta
            self._current_wh_steering = wh_steering
            self._current_wh_obstacle = wh_obstacle
            self._current_wh_center = wh_center
            cost, path, trajectory = self.run(debug=debug)
            trajectories.append((trajectory, (wh_dist, wh_theta, wh_steering, wh_obstacle, wh_center)))
        return trajectories

    def is_goal(self, node: NodeType) -> bool:
        _, _, theta = node
        return (self._goal_area.distance_to_point(node[:2]) <= 1e-5 and
                abs(theta - self._gtheta) <= self._allowed_goal_theta_difference)

    def distance_to_goal(self, node: NodeType) -> float:
        x, y, theta = node
        goal_x, goal_y, goal_orientation = self._goal_point
        distance_xy = np.hypot(x - goal_x, y - goal_y)
        distance_theta = abs(theta - goal_orientation)
        steering_change_cost = self.calculate_steering_change_cost(node, self._goal_point)
        heuristic_cost = (self._current_wh_dist * distance_xy +
                          self._current_wh_theta * distance_theta +
                          self._current_wh_steering * steering_change_cost)
        return heuristic_cost

    def neighbor_function(self, node: NodeType) -> Iterable[Tuple[float, NodeType]]:
        node_mtx = create_2d_transform_mtx(*node)
        for mp_name, mp in self._mps.items():
            pts = transform_2d_pts(node[2], node_mtx, self._mp_collision_points[mp_name])
            if any(check_collision(o, pts[:, :2].T) for o in self._obstacles_hp):
                continue
            x, y, theta = transform_2d_pts(node[2], node_mtx, [mp.points[-1]])[0]
            neighbor = (x, y, normalize_angle(theta))
            cost = self._wc_dist * mp.total_length
            yield cost, neighbor

    def path_to_full_trajectory(self, path: List[NodeType]) -> np.ndarray:
        points = []
        for p1, p2 in zip(path[:-1], path[1:]):
            mp_name = self._points_to_mp_names[p1, p2]
            points.append(self.motion_primitive_at(mp_name, p1)[:-1])
        return np.concatenate(points)

    def motion_primitive_at(self, mp_name: str, configuration: NodeType) -> np.ndarray:
        mtx = create_2d_transform_mtx(*configuration)
        return transform_2d_pts(configuration[2], mtx, self._mps[mp_name].points)


# Main scenario and plotting code
if __name__ == '__main__':
    fig, ax = plt.subplots()
    mps = load_motion_primitives(version='bicycle_model')
    scenario = intersection(turn_indicator=2, start_pos=1, start_lane=1, goal_lane=2, number_of_lanes=3)
    car_dimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    search = MotionPrimitiveSearch(
        scenario, car_dimensions, mps, margin=car_dimensions.radius,
        wh_dist=[1.0, 3.0], wh_theta=[2.7], wh_steering=[15]
    )

    draw_scenario(scenario, mps, car_dimensions, search, ax, draw_obstacles=True, draw_goal=True, draw_car=True)

    solutions = search.run_all(debug=True)

    for traj, weights in solutions:
        label = f'dist={weights[0]}'
        ax.plot(traj[:, 0], traj[:, 1], label=label)

    ax.legend()
    ax.axis('equal')
    plt.show()
