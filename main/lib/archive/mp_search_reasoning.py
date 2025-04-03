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
                 wh_ego: List[float] = None, wh_policy: List[float] = None, wh_rUser1: List[float] = None,
                 wh_rUser2: List[float] = None, wh_rUser3: List[float] = None,
                 wh_dist2goal: float = 1.0, wh_theta2goal: float = 2.7, wh_steer2goal: float = 15.0,
                 wh_dist2obs: float = 0.1, wh_dist2center: float = 0.0,
                 # wh_dist2goal: List[float] = None, wh_theta2goal: List[float] = None, wh_steer2goal: List[float] = None,
                 # wh_dist2obs: List[float] = None, wh_dist2center: List[float] = None,
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
        
        # Reasoning weights
        self._wh_ego_list = wh_ego if wh_ego else [0.25]
        self._wh_policy_list = wh_policy if wh_policy else [0.31]
        self._wh_rUser1_list = wh_rUser1 if wh_rUser1 else [0.44]
        self._wh_rUser2_list = wh_rUser2 if wh_rUser2 else [0.0]
        self._wh_rUser3_list = wh_rUser3 if wh_rUser3 else [0.0]
        
        # heuristic weights
        self._wh_dist2goal = wh_dist2goal
        self._wh_theta2goal = wh_theta2goal
        self._wh_steer2goal = wh_steer2goal
        self._wh_dist2obs = wh_dist2obs
        self._wh_dist2center = wh_dist2center

        # cost weights
        self._wc_dist = wc_dist
        self._wc_steering = wc_steering
        self._wc_obstacle = wc_obstacle
        self._wc_center = wc_center

        self._mp_collision_points: Dict[str, np.ndarray] = self._create_collision_points()

        self._current_wh_ego = self._wh_ego_list[0]
        self._current_wh_policy = self._wh_policy_list[0]
        self._current_wh_rUser1 = self._wh_rUser1_list[0]
        self._current_wh_rUser2 = self._wh_rUser2_list[0]
        self._current_wh_rUser3 = self._wh_rUser3_list[0]
        
        # Just in case needed
        self._current_wh_dist2goal = self._wh_dist2goal
        self._current_wh_theta2goal = self._wh_theta2goal
        self._current_wh_steer2goal = self._wh_steer2goal
        self._current_wh_dist2obs = self._wh_dist2obs
        self._current_wh_dist2center = self._wh_dist2center

    def _create_collision_points(self) -> Dict[str, np.ndarray]:
        MIN_DISTANCE_BETWEEN_POINTS = self._car_dimensions.radius

        out: Dict[str, np.ndarray] = {}

        # for each motion primitive
        for mp_name, mp in self._mps.items():
            points = mp.points.copy()

            # filter the points because we most likely don't need all of them
            points = resample_curve(points,
                                    dl=MIN_DISTANCE_BETWEEN_POINTS,
                                    keep_last_point=True)

            cc_trajectories = car_trajectory_to_collision_point_trajectories(points, self._car_dimensions)
            out[mp_name] = np.concatenate(cc_trajectories, axis=0)

        return out

    def calculate_steering_change_cost(self, current_node: NodeType, next_node: NodeType, steering_angle_weight: float = 1.0) -> float:
        """
        Calculates the cost associated with the change in steering angle required to transition from the current node to the next node.
        :param current_node: The current node (state) represented as a tuple (x, y, theta).
        :param next_node: The next node (state) represented as a tuple (x, y, theta).
        :param steering_angle_weight: A weighting factor for the steering angle change cost.
        :return: The calculated steering change cost.
        """
        # Extract the orientation (theta) from the current and next nodes
        _, _, current_theta = current_node
        _, _, next_theta = next_node

        # Calculate the change in orientation, which we'll use as a proxy for steering angle change
        # Normalize the angle difference to the range [-pi, pi] to handle the wrap-around case
        orientation_change = next_theta - current_theta
        orientation_change = (orientation_change + np.pi) % (2 * np.pi) - np.pi

        # Calculate the cost associated with this change in orientation
        steering_change_cost = abs(orientation_change) * steering_angle_weight
        
        return steering_change_cost
    
    def calculate_distance_point_to_halfplane(self, point: Tuple[float, float], half_planes: np.ndarray) -> float:
        """
        Calculate the minimum distance from a 2D point to a rectangular obstacle represented by half-planes.
        
        :param point: A tuple representing the (x, y) coordinates of the point.
        :param half_planes: A numpy array of shape (n, 3) where each row represents the coefficients (a, b, c) of a half-plane equation.
        :return: The minimum distance from the point to the obstacle's boundary represented by half-planes.
        """
        x0, y0 = point
        distances = []
        
        for a, b, c in half_planes:
            distance = abs(a*x0 + b*y0 + c) / (a**2 + b**2)**0.5
            distances.append(distance)
        
        return min(distances)

    def distance_to_nearest_obstacle(self, node: NodeType) -> float:
        """
        Calculate the minimum distance from a node to the nearest obstacle, considering the obstacle's half-planes.
        :param node: The node (state) for which to calculate the distance to the nearest obstacle.
        :return: The minimum distance to the nearest obstacle.
        """
        x, y, _ = node  # Extract the position of the node. Ignore orientation.
        min_distance = float('inf')
            
        for obstacle_half_planes in self._obstacles_hp:
            distance = self.calculate_distance_point_to_halfplane((x, y), obstacle_half_planes)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def collision_checking_points_at(self, mp_name: str, configuration: Tuple[float, float, float]) -> np.ndarray:
        cc_points = self._mp_collision_points[mp_name]
        mtx = create_2d_transform_mtx(*configuration)
        return transform_2d_pts(configuration[2], mtx, cc_points)

    def motion_primitive_at(self, mp_name: str, configuration: Tuple[float, float, float]) -> np.ndarray:
        points = self._mps[mp_name].points
        mtx = create_2d_transform_mtx(*configuration)
        return transform_2d_pts(configuration[2], mtx, points)

    @property
    def debug_data(self):
        return self._a_star.debug_data

    def run(self, debug=False):
        cost, path = self._a_star.run(self._start, is_goal_function=self.is_goal,
                                      heuristic_function=self.heuristicCost, debug=debug)
        trajectory = self.path_to_full_trajectory(path)
        return cost, path, trajectory

    def run_all(self, debug=False):
        trajectories = []
        combinations = product(self._wh_ego_list, self._wh_policy_list, self._wh_rUser1_list,
                               self._wh_rUser2_list, self._wh_rUser3_list)
        for wh_ego, wh_policy, wh_rUser1, wh_rUser2, wh_rUser3 in combinations:
            self._current_wh_ego = wh_ego
            self._current_wh_policy = wh_policy
            self._current_wh_rUser1 = wh_rUser1
            self._current_wh_rUser2 = wh_rUser2
            self._current_wh_rUser3 = wh_rUser3
            cost, path, trajectory = self.run(debug=debug)
            trajectories.append((trajectory, (wh_ego, wh_policy, wh_rUser1, wh_rUser2, wh_rUser3)))
        return trajectories

    def is_goal(self, node: NodeType) -> bool:
        _, _, theta = node
        result = self._goal_area.distance_to_point(node[:2]) <= 1e-5 \
                 and abs(theta - self._gtheta) <= self._allowed_goal_theta_difference

        return result

    def heuristicCost(self, node: NodeType) -> float:
        x, y, theta = node
        goal_x, goal_y, goal_orientation = self._goal_point
        
        # Calculate basic components
        distance_xy = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        distance_theta = min(abs(theta - goal_orientation), abs(theta - goal_orientation) - self._allowed_goal_theta_difference/2)
        steering_change_cost = self.calculate_steering_change_cost(node, self._goal_point, steering_angle_weight=1.0)
        
        # Calculate obstacle and center components
        obstacle_avoidance_cost = 0.0
        distance_from_center = 0.0
        
        if self._wh_dist2obs != 0.0:
            distance_to_obstacle = self.distance_to_nearest_obstacle(node)
            obstacle_avoidance_cost = 1 / distance_to_obstacle if distance_to_obstacle > 0 else float('inf')
            
        if self._wh_dist2center != 0.0:
            distance_from_center = np.sqrt(x**2 + y**2)
        
        # Ego-related costs (distance to goal, orientation difference, steering)
        ego_cost = (self._wh_dist2goal * distance_xy + 
                   self._wh_theta2goal * distance_theta + 
                   self._wh_steer2goal * steering_change_cost)
        
        # Policy-related costs (obstacle avoidance)
        policy_cost = self._wh_dist2obs * obstacle_avoidance_cost
        
        # User1-related costs (distance from center)
        rUser1_cost = self._wh_dist2center * distance_from_center
        
        # User2 and User3 costs (can be expanded in the future)
        rUser2_cost = 0.0
        rUser3_cost = 0.0
        
        # Combine with high-level weights
        heuristic_cost = (self._current_wh_ego * ego_cost +
                         self._current_wh_policy * policy_cost +
                         self._current_wh_rUser1 * rUser1_cost +
                         self._current_wh_rUser2 * rUser2_cost +
                         self._current_wh_rUser3 * rUser3_cost)
        
        return heuristic_cost

    def neighbor_function(self, node: NodeType) -> Iterable[Tuple[float, NodeType]]:
        node_rel_to_world_mtx = create_2d_transform_mtx(*node)
        
        for mp_name, mp in self._mps.items():
            # Transform Collision Checking Points Given Existing Matrix
            collision_checking_points = transform_2d_pts(node[2], node_rel_to_world_mtx,
                                                        self._mp_collision_points[mp_name])
            collision_checking_points_xy = collision_checking_points[:, :2].T

            # Check Collision With Every Obstacle
            # since it is not a list but an iterator, it doesn't check all of them if it doesn't have to,
            # but breaks once the first colliding obstacle is found.
            # Important: If the () parentheses were replaced with [] in the line below, then it would check all,
            # which we don't want.
            #
            # If no colliding obstacles are found, then it will have checked all of them.
            collides = any((check_collision(o, collision_checking_points_xy) for o in self._obstacles_hp))

            if not collides:
                # we can yield this obstacle because it has now been properly collision-checked

                # first, transform just the last point of the trajectory (because this may be the wrong path, and
                # we may not need this trajectory in the end at all, so there is no point in computing it right now)
                x, y, theta = tuple(
                    np.squeeze(transform_2d_pts(node[2], node_rel_to_world_mtx, np.atleast_2d(mp.points[-1]))).tolist())

                # normalize its angle
                neighbor = x, y, normalize_angle(theta)

                # store the motion primitive name
                self._points_to_mp_names[node, neighbor] = mp_name

                # Calculate costs
                steering_change_cost = self.calculate_steering_change_cost(node, neighbor, steering_angle_weight=1.0)
                
                obstacle_avoidance_cost = 0.0
                distance_from_center = 0.0
                
                if self._wc_obstacle != 0.0:
                    distance_to_obstacle = self.distance_to_nearest_obstacle(neighbor)
                    obstacle_avoidance_cost = 1 / distance_to_obstacle if distance_to_obstacle > 0 else float('inf')
                    
                if self._wc_center != 0.0: 
                    distance_from_center = np.linalg.norm([x, y])
                
                # Combine costs with weights
                cost = (self._wc_dist * mp.total_length + 
                       self._wc_steering * steering_change_cost + 
                       self._wc_obstacle * obstacle_avoidance_cost + 
                       self._wc_center * distance_from_center)
                
                yield cost, neighbor

    def path_to_full_trajectory(self, path: List[NodeType]) -> np.ndarray:
        points: List[np.ndarray] = []

        for p1, p2 in zip(path[:-1], path[1:]):
            # get the name of the motion primitive that leads from p1 to p2
            mp_name = self._points_to_mp_names[p1, p2]

            # transform trajectory (relative to p1) to world space
            points_this = self.motion_primitive_at(mp_name=mp_name, configuration=p1)[:-1]
            points.append(points_this)

        # get the whole trajectory 
        return np.concatenate(points, axis=0)


# Main scenario and plotting code
if __name__ == '__main__':
    fig, ax = plt.subplots()
    mps = load_motion_primitives(version='bicycle_model')
    scenario = intersection(turn_indicator=2, start_pos=1, start_lane=1, goal_lane=2, number_of_lanes=3)
    car_dimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    search = MotionPrimitiveSearch(
        scenario, car_dimensions, mps, margin=car_dimensions.radius,
        wh_ego=[1.0, 3.0], wh_policy=[2.7], wh_rUser1=[15]
    )

    draw_scenario(scenario, mps, car_dimensions, search, ax, draw_obstacles=True, draw_goal=True, draw_car=True, draw_mps= False, draw_mps2=False, draw_collision_checking=False, draw_car2=False)

    solutions = search.run_all(debug=True)

    for traj, weights in solutions:
        label = f'ego={weights[0]}, policy={weights[1]}, rUser1={weights[2]}'
        ax.plot(traj[:, 0], traj[:, 1], label=label)

    ax.legend()
    ax.axis('equal')
    plt.show()