import sys
sys.path.append('..')

from typing import Dict, Tuple, List, Iterable
import numpy as np

from lib.a_star import AStar
from lib.car_dimensions import CarDimensions
from lib.linalg import create_2d_transform_mtx, transform_2d_pts
from lib.maths import normalize_angle
from lib.motion_primitive import MotionPrimitive
from lib.obstacles import check_collision
from lib.scenario import Scenario
from lib.trajectories import car_trajectory_to_collision_point_trajectories, resample_curve

NodeType = Tuple[float, float, float]

class MotionPrimitiveSearch:
    def __init__(self, 
                 scenario: Scenario, 
                 car_dimensions: CarDimensions, 
                 mps: Dict[str, MotionPrimitive],
                 margin: float,
                 # Three lists of weights for the *heuristic*
                 wh_ego: List[float] = None,
                 wh_policy: List[float] = None,
                 wh_other: List[float] = None,
                 # Real cost weights remain unchanged
                 wc_dist: float = 1.0, 
                 wc_steering: float = 5.0, 
                 wc_obstacle: float = 0.1, 
                 wc_center: float = 0.0):
        """
        Constructor for MotionPrimitiveSearch.

        :param scenario: Scenario object containing start, goal, obstacles, etc.
        :param car_dimensions: CarDimensions object with radius, etc.
        :param mps: Dictionary of motion primitives, each identified by a string key.
        :param margin: Obstacle margin for collision checks.
        :param wh_ego: List of heuristic weights for the 'ego vehicle'. May be empty or of any length.
        :param wh_policy: List of heuristic weights for the 'policy maker'. May be empty or of any length.
        :param wh_other: List of heuristic weights for 'other user'. May be empty or of any length.
        :param wc_dist: Weight for distance in the *real* cost.
        :param wc_steering: Weight for steering angle change in the *real* cost.
        :param wc_obstacle: Weight for obstacle cost in the *real* cost.
        :param wc_center: Weight for distance-from-center cost in the *real* cost.
        """
        self._mps = mps
        self._car_dimensions = car_dimensions

        # Keep a mapping from (prev_node, next_node) -> mp_name
        self._points_to_mp_names: Dict[Tuple[NodeType, NodeType], str] = {}

        # Scenario references
        self._start = scenario.start
        self._goal_area = scenario.goal_area
        self._goal_point = scenario.goal_point
        self._allowed_goal_theta_difference = scenario.allowed_goal_theta_difference
        self._gx, self._gy, self._gtheta = scenario.goal_point

        # Half-plane obstacles
        self._obstacles_hp: List[np.ndarray] = [o.to_convex(margin=margin) for o in scenario.obstacles]

        # A* instance
        self._a_star: AStar[NodeType] = AStar(neighbor_function=self.neighbor_function)

        # Store the three lists for the heuristic
        self._wh_ego = wh_ego if wh_ego is not None else []
        self._wh_policy = wh_policy if wh_policy is not None else []
        self._wh_other = wh_other if wh_other is not None else []

        # Real cost function weights
        self._wc_dist = wc_dist
        self._wc_steering = wc_steering
        self._wc_obstacle = wc_obstacle
        self._wc_center = wc_center

        # For convenience, we will hold "active" sums of each list
        # to be used by distance_to_goal() in each run. 
        # We'll overwrite these before each run in run_all().
        self._sum_ego = sum(self._wh_ego)  # default if needed
        self._sum_policy = sum(self._wh_policy)
        self._sum_other = sum(self._wh_other)

        # Precompute collision points for each motion primitive
        self._mp_collision_points: Dict[str, np.ndarray] = self._create_collision_points()

    def _create_collision_points(self) -> Dict[str, np.ndarray]:
        """
        Creates sets of collision-check points along each motion primitive,
        spaced at least by the car radius (MIN_DISTANCE_BETWEEN_POINTS).
        """
        MIN_DISTANCE_BETWEEN_POINTS = self._car_dimensions.radius
        out: Dict[str, np.ndarray] = {}
        for mp_name, mp in self._mps.items():
            points = mp.points.copy()
            # Resample to ensure consistent spacing
            points = resample_curve(points, dl=MIN_DISTANCE_BETWEEN_POINTS, keep_last_point=True)
            cc_trajectories = car_trajectory_to_collision_point_trajectories(points, self._car_dimensions)
            out[mp_name] = np.concatenate(cc_trajectories, axis=0)
        return out

    def calculate_steering_change_cost(self, current_node: NodeType, next_node: NodeType, steering_angle_weight: float = 1.0) -> float:
        """
        Calculates the cost associated with the change in steering angle required
        to transition from the current node to the next node.
        """
        _, _, current_theta = current_node
        _, _, next_theta = next_node

        orientation_change = next_theta - current_theta
        # Normalize angle difference to [-pi, pi]
        orientation_change = (orientation_change + np.pi) % (2 * np.pi) - np.pi

        steering_change_cost = abs(orientation_change) * steering_angle_weight
        return steering_change_cost

    def calculate_distance_point_to_halfplane(self, point: Tuple[float, float], half_planes: np.ndarray) -> float:
        """
        Calculates the minimum distance from a 2D point to the boundary 
        of a rectangular obstacle represented by half-planes.
        """
        x0, y0 = point
        distances = []
        for a, b, c in half_planes:
            distance = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
            distances.append(distance)
        return min(distances) if distances else 0.0

    def distance_to_nearest_obstacle(self, node: NodeType) -> float:
        """
        Returns the minimum distance from 'node' to the nearest obstacle.
        """
        x, y, _ = node
        min_distance = float('inf')
        for obstacle_half_planes in self._obstacles_hp:
            distance = self.calculate_distance_point_to_halfplane((x, y), obstacle_half_planes)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def neighbor_function(self, node: NodeType) -> Iterable[Tuple[float, NodeType]]:
        """
        Provides neighbors of the given node (i.e., reachable states) along 
        with their real costs, determined by wc_dist, wc_steering, wc_obstacle, wc_center.
        """
        node_rel_to_world_mtx = create_2d_transform_mtx(*node)

        for mp_name, mp in self._mps.items():
            # Transform collision checking points
            collision_checking_points = transform_2d_pts(node[2], node_rel_to_world_mtx,
                                                         self._mp_collision_points[mp_name])
            collision_checking_points_xy = collision_checking_points[:, :2].T

            # Collision check
            collides = any(check_collision(o, collision_checking_points_xy) for o in self._obstacles_hp)
            if not collides:
                # Transform just the final point of this MP
                final_pt = transform_2d_pts(node[2], node_rel_to_world_mtx, 
                                            np.atleast_2d(mp.points[-1]))
                x, y, theta = tuple(np.squeeze(final_pt).tolist())
                neighbor = (x, y, normalize_angle(theta))

                # Record the MP used
                self._points_to_mp_names[(node, neighbor)] = mp_name

                # Steering angle cost
                steering_change_cost = self.calculate_steering_change_cost(node, neighbor, 1.0)

                # Obstacle proximity cost
                obstacle_avoidance_cost = 0.0
                if self._wc_obstacle != 0.0:
                    dist_to_obs = self.distance_to_nearest_obstacle(neighbor)
                    obstacle_avoidance_cost = (1.0 / dist_to_obs) if dist_to_obs > 0.0 else float('inf')

                # Distance from center
                distance_from_center = 0.0
                if self._wc_center != 0.0:
                    distance_from_center = np.linalg.norm([x, y])

                # Real cost function
                cost = (self._wc_dist * mp.total_length
                        + self._wc_steering * steering_change_cost
                        + self._wc_obstacle * obstacle_avoidance_cost
                        + self._wc_center * distance_from_center)
                
                yield cost, neighbor

    def is_goal(self, node: NodeType) -> bool:
        """
        Checks if 'node' lies in the goal region with an acceptable heading.
        """
        _, _, theta = node
        close_to_center = (self._goal_area.distance_to_point(node[:2]) <= 1e-5)
        heading_ok = (abs(theta - self._gtheta) <= self._allowed_goal_theta_difference)
        return close_to_center and heading_ok

    def distance_to_goal(self, node: NodeType) -> float:
        """
        Heuristic used by A*: exclusively uses the three new lists 
        (wh_ego, wh_policy, wh_other). We sum them and multiply
        by some key terms (distance_xy, orientation difference, steering proxy).

        Modify as needed to incorporate more advanced usage of the lists.
        """
        x, y, theta = node
        gx, gy, gtheta = self._goal_point

        # Basic geometric terms
        distance_xy = np.hypot(x - gx, y - gy)
        # Orientation difference
        orientation_diff = abs(((theta - gtheta) + np.pi) % (2*np.pi) - np.pi)
        # Steering proxy from node -> goal (optional)
        steering_change_cost = self.calculate_steering_change_cost(node, self._goal_point, 1.0)

        # Example heuristic formula
        # Using the "active" sums of wh_ego, wh_policy, wh_other
        heuristic_cost = (self._sum_ego    * distance_xy
                          + self._sum_policy * orientation_diff
                          + self._sum_other  * steering_change_cost)

        return heuristic_cost

    def motion_primitive_at(self, mp_name: str, configuration: NodeType) -> np.ndarray:
        """
        Retrieve the trajectory of the named motion primitive transformed into world coordinates 
        based on 'configuration' (x, y, theta).
        """
        points = self._mps[mp_name].points
        mtx = create_2d_transform_mtx(*configuration)
        return transform_2d_pts(configuration[2], mtx, points)

    def path_to_full_trajectory(self, path: List[NodeType]) -> np.ndarray:
        """
        Reconstructs the full continuous trajectory (as an array of XYTheta) 
        by applying the relevant motion primitive transformations along 'path'.
        """
        if len(path) < 2:
            return np.array([])

        segments = []
        for p1, p2 in zip(path[:-1], path[1:]):
            mp_name = self._points_to_mp_names[(p1, p2)]
            transformed = self.motion_primitive_at(mp_name, p1)[:-1]
            segments.append(transformed)
        if segments:
            return np.concatenate(segments, axis=0)
        return np.array([])

    @property
    def debug_data(self):
        """
        If you need debug information from the A* routine, 
        it is accessible here.
        """
        return self._a_star.debug_data

    # -------------------------------------------------------------------
    #  Provide a method to run once for a single set of sums 
    #  (Optional convenience if you only want one run).
    # -------------------------------------------------------------------
    def run(self, debug: bool = False) -> Tuple[float, List[NodeType], np.ndarray]:
        """
        Runs A* with whatever the current sums of wh_ego, wh_policy, wh_other are.
        If you called this directly, ensure you set self._sum_ego, etc. beforehand 
        (or provide non-empty lists to the constructor).
        """
        cost, path = self._a_star.run(
            start_node=self._start,
            is_goal_function=self.is_goal,
            heuristic_function=self.distance_to_goal,
            debug=debug
        )
        trajectory = self.path_to_full_trajectory(path)
        return cost, path, trajectory

    # -------------------------------------------------------------------
    #  Return multiple solutions by iterating over all combinations
    #  of wh_ego, wh_policy, wh_other. 
    # -------------------------------------------------------------------
    def run_all(self, debug=False) -> List[Tuple[float, List[NodeType], np.ndarray, float, float, float]]:
        """
        Runs A* for every combination of (wh_ego, wh_policy, wh_other),
        returning a list of (cost, path, trajectory, e, p, o).

        If any list is empty, we skip and return an empty list.
        """
        solutions = []

        if (len(self._wh_ego) == 0) or (len(self._wh_policy) == 0) or (len(self._wh_other) == 0):
            print("One (or more) of the weight lists is empty - no solutions returned.")
            return solutions

        for e in self._wh_ego:
            for p in self._wh_policy:
                for o in self._wh_other:
                    # Temporarily set the "active" sums
                    self._sum_ego = e
                    self._sum_policy = p
                    self._sum_other = o

                    cost, path = self._a_star.run(
                        start_node=self._start,
                        is_goal_function=self.is_goal,
                        heuristic_function=self.distance_to_goal,
                        debug=debug
                    )
                    trajectory = self.path_to_full_trajectory(path)

                    solutions.append((cost, path, trajectory, e, p, o))

        return solutions
