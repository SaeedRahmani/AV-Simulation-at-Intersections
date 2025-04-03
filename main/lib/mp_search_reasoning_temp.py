#!/usr/bin/env python3
'''
This is a comprehensive version of motion primitive search that combines:
1. The multi-weight reasoning from mp_search_reasoning.py
2. The road user behavior models from mp_search_ww_generic_reasons.py
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
from lib.parameters import CyclistParameters, DriverParameters

NodeType = Tuple[float, float, float]

class MotionPrimitiveSearch:
    def __init__(self, scenario: Scenario, car_dimensions: CarDimensions, mps: Dict[str, MotionPrimitive],
                 margin: float,
                 moving_obstacles_state: np.ndarray = None,
                 centerline: float = 0.0,
                 # Reasoning weights (as lists to enable multiple trajectory generation)
                 wh_ego: List[float] = None, 
                 wh_policy: List[float] = None, 
                 wh_rUser1: List[float] = None,
                 wh_rUser2: List[float] = None, 
                 wh_rUser3: List[float] = None,
                 # Original heuristic weights
                 wh_dist2goal: float = 0.25, 
                 wh_theta2goal: float = 2.7, 
                 wh_steer2goal: float = 15.0,
                 wh_dist2obs: float = 0.0, 
                 wh_dist2center: float = 0.0,
                 # Reasoning-specific weights (from mp_search_ww_generic_reasons.py)
                 wh_policymaker_rightlane_reason: float = 0.25, 
                 wh_cyclist_comfort_reason: float = 0.25, 
                 wh_driver_patience_reason: float = 0.25,
                 # Weights for the real cost function
                 wc_dist: float = 1.0, 
                 wc_steering: float = 5.0, 
                 wc_obstacle: float = 0.1, 
                 wc_center: float = 0.0):

        self.CENTERLINE_LOCATION = centerline
        self._mps = mps
        self._car_dimensions = car_dimensions
        self._points_to_mp_names: Dict[Tuple[NodeType, NodeType], str] = {}
        self._moving_obstacles_state = moving_obstacles_state

        self._start = scenario.start
        self._goal_area = scenario.goal_area
        self._goal_point = scenario.goal_point
        self._allowed_goal_theta_difference = scenario.allowed_goal_theta_difference
        self._obstacles_hp: List[np.ndarray] = [o.to_convex(margin=margin) for o in scenario.obstacles]
        self._gx, self._gy, self._gtheta = scenario.goal_point

        self._a_star: AStar[NodeType] = AStar(neighbor_function=self.neighbor_function)
        
        # Initialize high-level reasoning weights (as lists for multiple trajectory generation)
        self._wh_ego_list = wh_ego if wh_ego else [0.25]
        self._wh_policy_list = wh_policy if wh_policy else [0.31]
        self._wh_rUser1_list = wh_rUser1 if wh_rUser1 else [0.44]
        self._wh_rUser2_list = wh_rUser2 if wh_rUser2 else [0.0]
        self._wh_rUser3_list = wh_rUser3 if wh_rUser3 else [0.0]
        
        # Initialize the current weights (will be updated during run_all)
        self._current_wh_ego = self._wh_ego_list[0]
        self._current_wh_policy = self._wh_policy_list[0]
        self._current_wh_rUser1 = self._wh_rUser1_list[0]
        self._current_wh_rUser2 = self._wh_rUser2_list[0]
        self._current_wh_rUser3 = self._wh_rUser3_list[0]
        
        # Original heuristic weights
        self._wh_dist2goal = wh_dist2goal
        self._wh_theta2goal = wh_theta2goal
        self._wh_steer2goal = wh_steer2goal
        self._wh_dist2obs = wh_dist2obs
        self._wh_dist2center = wh_dist2center
        
        # Reasoning-specific weights
        self._wh_policymaker_rightlane_reason = wh_policymaker_rightlane_reason
        self._wh_cyclist_comfort_reason = wh_cyclist_comfort_reason
        self._wh_driver_patience_reason = wh_driver_patience_reason

        # Weights for the real cost function
        self._wc_dist = wc_dist
        self._wc_steering = wc_steering
        self._wc_obstacle = wc_obstacle
        self._wc_center = wc_center
        
        # For each motion primitive, create collision points
        self._mp_collision_points: Dict[str, np.ndarray] = self._create_collision_points()
        
        # Initialize proximity tracking (for time-based reasoning)
        self._close_proximity_time = 0.0

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
            # Reset proximity tracking for each run
            self._close_proximity_time = 0.0
            
            # Update current weights
            self._current_wh_ego = wh_ego
            self._current_wh_policy = wh_policy
            self._current_wh_rUser1 = wh_rUser1
            self._current_wh_rUser2 = wh_rUser2
            self._current_wh_rUser3 = wh_rUser3
            
            # Run A* with current weights
            cost, path, trajectory = self.run(debug=debug)
            trajectories.append((trajectory, (wh_ego, wh_policy, wh_rUser1, wh_rUser2, wh_rUser3)))
        return trajectories

    def is_goal(self, node: Tuple[float, float, float]) -> bool:
        _, _, theta = node
        result = self._goal_area.distance_to_point(node[:2]) <= 1e-5 \
                 and abs(theta - self._gtheta) <= self._allowed_goal_theta_difference

        return result

    def normalize_distance_to_goal(self, x, y, goal_x, goal_y):
        """
        Calculate and normalize the Euclidean distance to the goal.
        
        Args:
            x, y: Current position coordinates
            goal_x, goal_y: Goal position coordinates
            
        Returns:
            float: Normalized distance (0.0 = at goal, 1.0 = at or beyond maximum distance)
        """
        # Calculate raw Euclidean distance
        raw_distance_xy = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        
        # Normalize to 0-1 range where 0 = at goal, 1 = at or beyond 80 meters
        MAX_DISTANCE = 80.0  # meters
        normalized_distance = min(raw_distance_xy / MAX_DISTANCE, 1.0)
        
        return normalized_distance

    def compute_centerline_deviation_cost(self, x):
        """
        Calculate the cost for deviating from the centerline based on specific road rules:
        - If x < 0: Cost ranges from 0 (at centerline) to 1 (at -3 meters)
        - If x ≥ 0: Cost is always 0
        
        Args:
            x: Current x-coordinate of the vehicle
            
        Returns:
            float: Normalized cost (0-1) based on specified rules
        """
        # Road parameters
        MAX_DEVIATION = 3.0  # meters from centerline, width of one lane
        
        # Only apply cost when x is negative (left of centerline)
        if x < 0:
            # Calculate deviation (how far left of centerline)
            deviation = abs(x - self.CENTERLINE_LOCATION)
            
            # Normalize to 0-1 range, capped at 1.0
            return min(deviation / MAX_DEVIATION, 1.0)
        else:
            # No cost if vehicle is on or to the right of centerline
            return 0.0

    def compute_bicycle_time_cost(self, distance_to_moving_obstacles):
        """
        Calculate time-based cyclist comfort factor based on time spent in close proximity.
        
        Args:
            distance_to_moving_obstacles: Distance between vehicle and cyclist (meters)
            
        Returns:
            float: Time-based comfort cost (0.0 = comfortable, 1.0 = maximum discomfort)
        """
        # Define threshold for "too close"
        SAFETY_DISTANCE = CyclistParameters.DISTANCE_REF  # meters
        
        # If we're too close, increment the time counter
        if distance_to_moving_obstacles < SAFETY_DISTANCE:
            self._close_proximity_time += self._mps['straight'].n_seconds  # Add time for each node call
        else:
            # Reset the counter if we're not too close anymore
            self._close_proximity_time = 0.0
        
        # Define maximum allowed time in close proximity
        MAX_ALLOWED_TIME = CyclistParameters.TIME_THRESHOLD  # seconds
        
        # Define saturation time (when discomfort reaches maximum)
        SATURATION_TIME = MAX_ALLOWED_TIME * 2.0  # seconds
        
        # Calculate time-based penalty
        if self._close_proximity_time <= MAX_ALLOWED_TIME:
            # No discomfort if within allowed time
            return 0.0
        elif self._close_proximity_time >= SATURATION_TIME:
            # Maximum discomfort if beyond saturation time
            return 1.0
        else:
            # Linear interpolation between allowed time and saturation time
            normalized_excess_time = (self._close_proximity_time - MAX_ALLOWED_TIME) / (SATURATION_TIME - MAX_ALLOWED_TIME)
            return normalized_excess_time

    def compute_bicycle_distance_cost(self, distance_to_moving_obstacles: float) -> float:
        """
        Calculate the cyclist's comfort based on distance to the vehicle:
        - If distance > MIN_SAFE_DISTANCE: Cost is 0.0 (cyclist is comfortable)
        - If distance <= MIN_SAFE_DISTANCE: Cost increases exponentially from 0.0 to 1.0
        
        Args:
            distance_to_moving_obstacles: Distance between vehicle and cyclist (meters)
            
        Returns:
            float: Comfort cost, where 0.0 is comfortable and 1.0 is very uncomfortable
        """
        # Minimum safe distance threshold
        MIN_SAFE_DISTANCE = CyclistParameters.DISTANCE_REF  # meters
        
        # If we're beyond the minimum safe distance, cyclist is comfortable (cost = 0)
        if distance_to_moving_obstacles >= MIN_SAFE_DISTANCE:
            return 0.0
        
        # Calculate how much we've encroached below the minimum safe distance
        encroachment = MIN_SAFE_DISTANCE - distance_to_moving_obstacles
        
        # Define how quickly discomfort grows as distance decreases
        # Higher values make the cost grow more rapidly
        growth_rate = 0.5
        
        # Calculate exponential cost
        # This gives us 0 when at MIN_SAFE_DISTANCE and approaches 1 as distance decreases
        cost = 1.0 - np.exp(-growth_rate * encroachment)
        
        # Determine at what encroachment we want to reach a cost of 1.0
        # We'll set it to reach 1.0 when distance is 0 (collision)
        max_encroachment = MIN_SAFE_DISTANCE  # When distance is 0
        max_cost = 1.0 - np.exp(-growth_rate * max_encroachment)
        
        # Scale to ensure we reach exactly 1.0 at maximum encroachment
        scaled_cost = cost / max_cost if max_cost > 0 else cost
        
        return min(scaled_cost, 1.0)  # Cap at 1.0 to be safe

    def compute_driver_patience(self, distance_to_moving_obstacles):
        """
        Calculate driver impatience based on time spent too close to cyclist.
        Normalized to return values between 0 and 1.
        
        Args:
            distance_to_moving_obstacles: Distance between vehicle and cyclist (meters)
            
        Returns:
            float: Driver impatience (0.0 = patient, 1.0 = completely impatient)
        """
        # Define threshold for "too close"
        SAFETY_DISTANCE = DriverParameters.DISTANCE_REF  # meters
        
        # If we're too close, increment the time counter
        if distance_to_moving_obstacles < SAFETY_DISTANCE:
            self._close_proximity_time += self._mps['straight'].n_seconds  # Add time for each node call
        else:
            # Reset the counter if we're not too close anymore
            self._close_proximity_time = 0.0
            
        # Define maximum allowed time in close proximity
        MAX_ALLOWED_TIME = DriverParameters.TIME_THRESHOLD  # seconds
        
        # Define saturation time (when patience fully expires)
        SATURATION_TIME = MAX_ALLOWED_TIME * 1.5  # seconds after threshold when impatience reaches maximum
        
        # Calculate time-based impatience
        if self._close_proximity_time <= MAX_ALLOWED_TIME:
            # No impatience if within allowed time
            return 0.0
        elif self._close_proximity_time >= (MAX_ALLOWED_TIME + SATURATION_TIME):
            # Maximum impatience if beyond saturation point
            return 1.0
        else:
            # Exponential growth of impatience between threshold and saturation
            excess_time = self._close_proximity_time - MAX_ALLOWED_TIME
            
            # Exponential function that grows from 0 to 1
            raw_impatience = 1.0 - np.exp(-3.0 * excess_time / SATURATION_TIME)
            
            # Ensure the function reaches exactly 1.0 at saturation time
            max_possible_value = 1.0 - np.exp(-3.0)
            normalized_impatience = raw_impatience / max_possible_value
            
            return min(normalized_impatience, 1.0)  # Cap at 1.0 for safety

    def heuristicCost(self, node: NodeType) -> float:
        x, y, theta = node
        goal_x, goal_y, goal_orientation = self._goal_point
        
        # Calculate basic components
        distance_xy = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        normalized_distance_xy = self.normalize_distance_to_goal(x, y, goal_x, goal_y)
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
        
        if self._moving_obstacles_state is not None:
            # Update bicycle position (project forward)
            # Projected bicycle forward, s = v * t
            projected_bicycle_x = self._moving_obstacles_state[0]
            projected_bicycle_y = self._moving_obstacles_state[1] + self._moving_obstacles_state[2] * self._mps['straight'].n_seconds
            
            # Calculate the distance between the current node and the moving obstacles
            distance_to_moving_obstacles = np.linalg.norm(
                [x - projected_bicycle_x, y - projected_bicycle_y])

            # === Reasoning Components ===
            
            # Ego-related costs (distance to goal, orientation difference, steering)
            ego_cost = (self._wh_dist2goal * normalized_distance_xy + 
                       self._wh_theta2goal * distance_theta + 
                       self._wh_steer2goal * steering_change_cost)
            
            # Policy-related costs (obstacle avoidance, centerline deviation)
            centerline_deviation = self.compute_centerline_deviation_cost(x)
            policy_cost = (self._wh_dist2obs * obstacle_avoidance_cost +
                          self._wh_policymaker_rightlane_reason * centerline_deviation)
            
            # Road User 1 (Cyclist) costs - comfort based on distance and time
            cyclist_distance_comfort = self.compute_bicycle_distance_cost(distance_to_moving_obstacles)
            cyclist_time_comfort = self.compute_bicycle_time_cost(distance_to_moving_obstacles)
            cyclist_total_comfort = cyclist_distance_comfort * cyclist_time_comfort
            rUser1_cost = self._wh_cyclist_comfort_reason * cyclist_total_comfort
            
            # Road User 2 (Driver) costs - patience
            driver_patience = self.compute_driver_patience(distance_to_moving_obstacles)
            rUser2_cost = self._wh_driver_patience_reason * driver_patience
            
            # Road User 3 costs (can be expanded in the future)
            rUser3_cost = 0.0
        else:
            # If no moving obstacles, use simpler cost structure
            ego_cost = (self._wh_dist2goal * normalized_distance_xy + 
                       self._wh_theta2goal * distance_theta + 
                       self._wh_steer2goal * steering_change_cost)
            
            policy_cost = self._wh_dist2obs * obstacle_avoidance_cost
            
            rUser1_cost = self._wh_dist2center * distance_from_center
            
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
    fig, ax = plt.subplots(figsize=(12, 8))
    mps = load_motion_primitives(version='bicycle_model')
    scenario = intersection(turn_indicator=2, start_pos=1, start_lane=1, goal_lane=2, number_of_lanes=3)
    car_dimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    # Example of moving obstacle (bicycle) [x, y, velocity]
    moving_obstacle = np.array([5.0, 10.0, 1.5])  # position (5,10) with 1.5 m/s velocity

    search = MotionPrimitiveSearch(
        scenario, car_dimensions, mps, margin=car_dimensions.radius,
        moving_obstacles_state=moving_obstacle,
        centerline=0.0,
        # Different weight combinations to explore
        wh_ego=[0.25, 0.5], 
        wh_policy=[0.25, 0.5], 
        wh_rUser1=[0.5, 0.25]
    )

    draw_scenario(scenario, mps, car_dimensions, search, ax, draw_obstacles=True, 
                 draw_goal=True, draw_car=True, draw_mps=False, draw_mps2=False, 
                 draw_collision_checking=False, draw_car2=False)

    solutions = search.run_all(debug=True)

    # Plot all trajectories with detailed labels
    for i, (traj, weights) in enumerate(solutions):
        label = f'ego={weights[0]:.2f}, policy={weights[1]:.2f}, cyclist={weights[2]:.2f}'
        ax.plot(traj[:, 0], traj[:, 1], label=label, linewidth=2)
        
        # Optionally, add markers at key points for better visualization
        if i == 0:  # Only for the first trajectory to avoid clutter
            # Mark points every 10 steps
            for j in range(0, len(traj), 10):
                ax.plot(traj[j, 0], traj[j, 1], 'o', markersize=4)

    # Add a marker for moving obstacle (bicycle)
    if search._moving_obstacles_state is not None:
        bicycle_x, bicycle_y = search._moving_obstacles_state[0], search._moving_obstacles_state[1]
        ax.plot(bicycle_x, bicycle_y, 'cs', markersize=10, label='Bicycle')

    # Add clear legend
    ax.legend(loc='best', fontsize=10)
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('X position (m)', fontsize=12)
    ax.set_ylabel('Y position (m)', fontsize=12)
    ax.set_title('Motion Primitive Search with Multi-Agent Reasoning', fontsize=14)
    
    # Ensure equal aspect ratio
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()