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
    def __init__(self, scenario: Scenario, car_dimensions: CarDimensions, mps: Dict[str, MotionPrimitive],
                 margin: float):
        self._mps = mps
        self._car_dimensions = car_dimensions
        self._points_to_mp_names: Dict[Tuple[NodeType, NodeType], str] = {}

        self._start = scenario.start
        self._goal_area = scenario.goal_area
        self._allowed_goal_theta_difference = scenario.allowed_goal_theta_difference
        self._obstacles_hp: List[np.ndarray] = [o.to_convex(margin=margin) for o in scenario.obstacles]
        self._gx, self._gy, self._gtheta = scenario.goal_point

        self._a_star: AStar[NodeType] = AStar(neighbor_function=self.neighbor_function)

        # for each motion primitive, create collision points
        self._mp_collision_points: Dict[str, np.ndarray] = self._create_collision_points()

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

    def run(self, debug=False) -> Tuple[float, List[NodeType], np.ndarray]:
        cost, path = self._a_star.run(self._start, is_goal_function=self.is_goal,
                                      heuristic_function=self.distance_to_goal, debug=debug)
        trajectory = self.path_to_full_trajectory(path)
        return cost, path, trajectory

    @property
    def debug_data(self):
        return self._a_star.debug_data

    def is_goal(self, node: Tuple[float, float, float]) -> bool:
        _, _, theta = node
        result = self._goal_area.distance_to_point(node[:2]) <= 1e-5 \
                 and abs(theta - self._gtheta) <= self._allowed_goal_theta_difference

        return result

    def distance_to_goal(self, node: Tuple[float, float, float]) -> float:
        x, y, theta = node
        distance_xy = self._goal_area.distance_to_point(node[:2])
        distance_theta = max(0., abs(theta - self._gtheta) - self._allowed_goal_theta_difference)
        return distance_xy + 2.7 * distance_theta 

    def collision_checking_points_at(self, mp_name: str, configuration: Tuple[float, float, float]) -> np.ndarray:
        cc_points = self._mp_collision_points[mp_name]
        mtx = create_2d_transform_mtx(*configuration)
        return transform_2d_pts(configuration[2], mtx, cc_points)

    def motion_primitive_at(self, mp_name: str, configuration: Tuple[float, float, float]) -> np.ndarray:
        points = self._mps[mp_name].points
        mtx = create_2d_transform_mtx(*configuration)
        return transform_2d_pts(configuration[2], mtx, points)

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

                # yield
                cost = mp.total_length
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
