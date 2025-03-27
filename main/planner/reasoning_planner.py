#!/usr/bin/env python3

"""
Example script demonstrating the updated MotionPrimitiveSearch with
three lists of heuristic weights (wh_ego, wh_policy, wh_other),
while keeping the real-cost weights (wc_*) unchanged.

It uses the intersection_multi_lanes scenario, loads motion primitives,
and then runs and plots the search results.

Replace or rename as you wish (e.g. multi_trajectory_planner.py),
and adjust imports if your directory structure differs.
"""

import sys
sys.path.append('..')

from typing import Dict, Tuple, List, Iterable
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

# -- Imports from your codebase (paths may need adjusting) --
from envs.arterial_multi_lanes import intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.helpers import measure_time
from lib.motion_primitive import load_motion_primitives
from lib.plotting import draw_scenario, draw_astar_search_points

# AStar and other classes used by MotionPrimitiveSearch
from lib.a_star import AStar
from lib.linalg import create_2d_transform_mtx, transform_2d_pts
from lib.maths import normalize_angle
from lib.motion_primitive import MotionPrimitive
from lib.obstacles import check_collision
from lib.scenario import Scenario
from lib.trajectories import car_trajectory_to_collision_point_trajectories, resample_curve

# --------------------------------
# New MotionPrimitiveSearch class
# --------------------------------
NodeType = Tuple[float, float, float]

class MotionPrimitiveSearch:
    def __init__(self, 
                 scenario: Scenario, 
                 car_dimensions: CarDimensions, 
                 mps: Dict[str, MotionPrimitive],
                 margin: float,
                 # Heuristic weights split into three lists:
                 wh_ego: List[float] = None,
                 wh_policy: List[float] = None,
                 wh_other: List[float] = None,
                 # Real cost weights remain unchanged:
                 wc_dist: float = 1.0, 
                 wc_steering: float = 5.0, 
                 wc_obstacle: float = 0.1, 
                 wc_center: float = 0.0):
        """
        Constructor for MotionPrimitiveSearch with three-list heuristic weighting
        and unchanged real cost weighting.
        """
        self._mps = mps
        self._car_dimensions = car_dimensions

        # Scenario references
        self._start = scenario.start
        self._goal_area = scenario.goal_area
        self._goal_point = scenario.goal_point
        self._allowed_goal_theta_difference = scenario.allowed_goal_theta_difference
        self._gx, self._gy, self._gtheta = scenario.goal_point

        # Obstacles as half-planes
        self._obstacles_hp: List[np.ndarray] = [o.to_convex(margin=margin) for o in scenario.obstacles]

        # A* instance (same code you had, just storing it)
        self._a_star: AStar[NodeType] = AStar(neighbor_function=self._neighbor_function)

        # Keep the three lists for the heuristic
        self._wh_ego = wh_ego if wh_ego is not None else []
        self._wh_policy = wh_policy if wh_policy is not None else []
        self._wh_other = wh_other if wh_other is not None else []

        # Real cost weights
        self._wc_dist = wc_dist
        self._wc_steering = wc_steering
        self._wc_obstacle = wc_obstacle
        self._wc_center = wc_center

        # We'll store a dictionary mapping (old_node, new_node) -> mp_name
        self._points_to_mp_names: Dict[Tuple[NodeType, NodeType], str] = {}

        # For convenience, define "active" sums for the lists, used in the heuristic
        self._sum_ego = sum(self._wh_ego)
        self._sum_policy = sum(self._wh_policy)
        self._sum_other = sum(self._wh_other)

        # Precompute collision points for each MP
        self._mp_collision_points: Dict[str, np.ndarray] = self._create_collision_points()

    def _create_collision_points(self) -> Dict[str, np.ndarray]:
        MIN_DISTANCE_BETWEEN_POINTS = self._car_dimensions.radius
        out: Dict[str, np.ndarray] = {}
        for mp_name, mp in self._mps.items():
            pts = mp.points.copy()
            # Resample so each segment is at least radius in length
            pts = resample_curve(pts, dl=MIN_DISTANCE_BETWEEN_POINTS, keep_last_point=True)
            cc_trajectories = car_trajectory_to_collision_point_trajectories(pts, self._car_dimensions)
            out[mp_name] = np.concatenate(cc_trajectories, axis=0)
        return out

    def _neighbor_function(self, node: NodeType) -> Iterable[Tuple[float, NodeType]]:
        """
        Same neighbor logic as your original code, using wc_* for real cost.
        """
        node_rel_to_world_mtx = create_2d_transform_mtx(*node)

        for mp_name, mp in self._mps.items():
            # Transform collision checking points
            collision_points = transform_2d_pts(node[2], node_rel_to_world_mtx, self._mp_collision_points[mp_name])
            collision_points_xy = collision_points[:, :2].T

            # Check collision with obstacles
            collides = any(check_collision(o, collision_points_xy) for o in self._obstacles_hp)
            if not collides:
                # Transform final MP point
                final_pt = transform_2d_pts(node[2], node_rel_to_world_mtx, np.atleast_2d(mp.points[-1]))
                x, y, theta = tuple(np.squeeze(final_pt).tolist())
                neighbor = (x, y, normalize_angle(theta))

                # Record motion primitive used
                self._points_to_mp_names[(node, neighbor)] = mp_name

                # Steering angle cost (proxy)
                steering_cost = self._calculate_steering_change_cost(node, neighbor, 1.0)

                # Obstacle "cost"
                obs_cost = 0.0
                if self._wc_obstacle != 0.0:
                    d = self._distance_to_nearest_obstacle(neighbor)
                    obs_cost = (1.0 / d) if d > 0.0 else float('inf')

                # Distance from center
                center_dist = 0.0
                if self._wc_center != 0.0:
                    center_dist = np.linalg.norm([x, y])

                # Final real cost
                cost = (self._wc_dist * mp.total_length
                        + self._wc_steering * steering_cost
                        + self._wc_obstacle * obs_cost
                        + self._wc_center * center_dist)
                yield cost, neighbor

    def _calculate_steering_change_cost(self, current_node: NodeType, next_node: NodeType, steering_angle_weight: float = 1.0) -> float:
        _, _, cur_theta = current_node
        _, _, nxt_theta = next_node
        dtheta = nxt_theta - cur_theta
        dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi  # normalize
        return abs(dtheta) * steering_angle_weight

    def _distance_to_nearest_obstacle(self, node: NodeType) -> float:
        x, y, _ = node
        min_dist = float('inf')
        for hp in self._obstacles_hp:
            dist = self._distance_point_to_halfplane((x, y), hp)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _distance_point_to_halfplane(self, point: Tuple[float, float], half_planes: np.ndarray) -> float:
        x0, y0 = point
        distances = []
        for a, b, c in half_planes:
            dist = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
            distances.append(dist)
        return min(distances) if distances else 0.0

    def is_goal(self, node: NodeType) -> bool:
        _, _, theta = node
        close_to_center = (self._goal_area.distance_to_point(node[:2]) <= 1e-5)
        heading_ok = (abs(theta - self._gtheta) <= self._allowed_goal_theta_difference)
        return close_to_center and heading_ok

    def distance_to_goal(self, node: NodeType) -> float:
        """
        Heuristic that uses wh_ego, wh_policy, wh_other. Feel free to adjust 
        how these sums weigh distance, orientation, steering, etc.
        """
        x, y, theta = node
        gx, gy, gtheta = self._goal_point

        dist_xy = np.hypot(x - gx, y - gy)
        dtheta = abs(((theta - gtheta) + np.pi) % (2*np.pi) - np.pi)
        steer_proxy = self._calculate_steering_change_cost(node, (gx, gy, gtheta), 1.0)

        # Combine them with the "active" sums
        cost = (self._sum_ego * dist_xy
                + self._sum_policy * dtheta
                + self._sum_other * steer_proxy)
        return cost

    def path_to_full_trajectory(self, path: List[NodeType]) -> np.ndarray:
        """
        Reconstruct a continuous XY(theta) trajectory from a list of discrete nodes.
        """
        if len(path) < 2:
            return np.array([])
        segments = []
        for p1, p2 in zip(path[:-1], path[1:]):
            mp_name = self._points_to_mp_names[(p1, p2)]
            transformed = self.motion_primitive_at(mp_name, p1)[:-1]
            segments.append(transformed)
        return np.concatenate(segments, axis=0) if segments else np.array([])

    def motion_primitive_at(self, mp_name: str, configuration: NodeType) -> np.ndarray:
        """
        Transform the MP's points from local to world coordinates.
        """
        pts = self._mps[mp_name].points
        mtx = create_2d_transform_mtx(*configuration)
        return transform_2d_pts(configuration[2], mtx, pts)

    @property
    def debug_data(self):
        return self._a_star.debug_data

    def run(self, debug: bool = False):
        """
        Executes a single A* run with the *current* sums of wh_ego, wh_policy, wh_other.
        Note: We pass self._start as the *first* argument to avoid TypeError in your AStar.
        """
        cost, path = self._a_star.run(
            self._start,
            is_goal_function=self.is_goal,
            heuristic_function=self.distance_to_goal,
            debug=debug
        )
        trajectory = self.path_to_full_trajectory(path)
        return cost, path, trajectory

    def run_all(self, debug: bool = False) -> List[Tuple[float, List[NodeType], np.ndarray, float, float, float]]:
        """
        If you have multiple values in each of wh_ego, wh_policy, wh_other,
        run A* for every combination. Returns a list of solutions.
        """
        solutions = []

        if not self._wh_ego or not self._wh_policy or not self._wh_other:
            print("One or more weight lists are empty; no solutions returned.")
            return solutions

        for e in self._wh_ego:
            for p in self._wh_policy:
                for o in self._wh_other:
                    self._sum_ego = e
                    self._sum_policy = p
                    self._sum_other = o

                    cost, path = self._a_star.run(
                        self._start,
                        is_goal_function=self.is_goal,
                        heuristic_function=self.distance_to_goal,
                        debug=debug
                    )
                    trajectory = self.path_to_full_trajectory(path)
                    solutions.append((cost, path, trajectory, e, p, o))

        return solutions


# -------------------------
#  Main usage (demo)
# -------------------------
if __name__ == '__main__':
    fig, ax = plt.subplots()

    # Example scenario parameters
    start_pos = 1
    turn_indicator = 2

    # Just do a single 'bicycle_model' version for demonstration
    for version in ['bicycle_model']:
        # Load motion primitives
        mps = load_motion_primitives(version=version)

        # Create an intersection scenario
        scenario = intersection(
            turn_indicator=turn_indicator,
            start_pos=start_pos,
            start_lane=1,
            goal_lane=2,
            number_of_lanes=3
        )
        car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

        # Instantiate our new search class with some example heuristic lists
        # and real-cost weights
        search = MotionPrimitiveSearch(
            scenario,
            car_dimensions,
            mps,
            margin=car_dimensions.radius,
            wh_ego=[1.0, 1.5, 10.0],
            wh_policy=[2.7],
            wh_other=[15],
            wc_dist=1.0,
            wc_steering=5.0,
            wc_obstacle=0.1,
            wc_center=0.0
        )

        # Draw the scenario (same calls as your snippet)
        draw_scenario(scenario, mps, car_dimensions, search, ax,
                      draw_obstacles=True, draw_goal=True, draw_car=True, 
                      draw_mps=False, draw_collision_checking=False,
                      draw_car2=False, draw_mps2=False, mp_name='right1')

        # # Run the search (single run) with the chosen sums
        # @measure_time
        # def run_search():
        #     return search.run(debug=True)

        # try:
        #     cost, path, trajectory = run_search()
        #     print(f"Found path with cost={cost}, # of nodes in path={len(path)}")

        #     # Plot the resulting trajectory in blue
        #     if trajectory.shape[0] > 0:
        #         cx = trajectory[:, 0]
        #         cy = trajectory[:, 1]
        #         ax.plot(cx, cy, color='b')
        # except KeyboardInterrupt:
        #     print("Search interrupted by user.")

        # If you want to test multiple combos from the lists, do:
        solutions = search.run_all(debug=True)
        for c, pth, traj, e, pol, o in solutions:
            print(f"Combo e={e}, p={pol}, o={o} => cost={c}, path length={len(pth)}")
            ax.plot(traj[:,0], traj[:,1], '--')

        # Draw expansions
        # sc = draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)
        # plt.colorbar(sc)

    # Example legend
    marker_size = 10
    line_trajectory = mlines.Line2D([], [], color='b', marker='', ls='-',
                                    label='Path from MP', markeredgewidth=3, markersize=marker_size)
    goal_area = mlines.Line2D([], [], color=(1, 0.8, 0.8), marker='s', ls='',
                              label='Goal area', markersize=marker_size)
    goal_arrow = mlines.Line2D([], [], color='r', marker='$\u279C$', ls='',
                               label='Goal direction', markersize=marker_size)
    obstacles_legend = mlines.Line2D([], [], color='b', marker='s', ls='',
                                     label='Obstacles', markersize=marker_size)
    forbidden = mlines.Line2D([], [], color=(0.8, 0.8, 0.8), marker='s', ls='',
                              label='Forbidden by traffic rules', markersize=marker_size)
    ego_vehicle = mlines.Line2D([], [], color='g', marker='s', ls='',
                                label='Ego vehicle', markersize=marker_size, fillstyle='none')
    mp_points = mlines.Line2D([], [], color=(47/255, 108/255, 144/255), marker='.',
                              ls='', label='Visited points A*', markersize=marker_size)

    plt.legend(handles=[line_trajectory, goal_area, goal_arrow, 
                        obstacles_legend, ego_vehicle, forbidden, mp_points])

    ax.axis('equal')
    plt.show()
