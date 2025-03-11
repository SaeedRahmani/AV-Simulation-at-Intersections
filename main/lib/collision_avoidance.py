from typing import List, Tuple, Union, Optional

import numpy as np

from lib.car_dimensions import CarDimensions
from lib.helpers import measure_time
from lib.trajectories import car_trajectory_to_collision_point_trajectories


def _combine_rowwise_repeat(arrays_2d: List[np.ndarray], repeats=1) -> Tuple[int, np.ndarray]:
    n = len(arrays_2d) * repeats
    arr = np.concatenate([*arrays_2d] * repeats, axis=1)
    a, b = arr.shape
    arr = arr.reshape((a * n, b // n))
    return n, arr


def _pad_trajectory(traj: np.ndarray, n_iterations: int) -> np.ndarray:
    if len(traj) < n_iterations:
        return np.vstack([traj, np.repeat(traj[-1:, :], n_iterations - len(traj), axis=0)])
    else:
        return traj[:n_iterations]


def _pad_trajectories(traj_agent: np.ndarray, trajs_o: List[np.ndarray]):
    n_iterations = max(len(traj_agent), max((len(tr) for tr in trajs_o)))
    traj_agent = _pad_trajectory(traj_agent, n_iterations)
    trajs_o = [_pad_trajectory(tr, n_iterations) for tr in trajs_o]
    return traj_agent, trajs_o


def _get_rowwise_diffs(car_dimensions, traj_agent: np.ndarray, traj_obstacles: List[np.ndarray]):
    n_circle_centers = len(car_dimensions.circle_centers)

    traj_agent, traj_obstacles = _pad_trajectories(traj_agent, traj_obstacles)

    rows_per_frame_ag, cc_pts_ag = _combine_rowwise_repeat(
        [tr[:, :2] for tr in car_trajectory_to_collision_point_trajectories(traj_agent, car_dimensions)], repeats=1)
    rows_per_frame_ag *= n_circle_centers * len(traj_obstacles)
    cc_pts_ag = np.repeat(cc_pts_ag, len(traj_obstacles) * n_circle_centers, axis=0)
    rows_per_frame_obs, cc_pts_obs = _combine_rowwise_repeat(
        [cc_tr[:, :2] for tr in traj_obstacles for cc_tr in
         car_trajectory_to_collision_point_trajectories(tr, car_dimensions)], repeats=n_circle_centers)
    assert rows_per_frame_ag == rows_per_frame_obs
    # print("Number of point pairs:", len(cc_pts_ag))
    return rows_per_frame_ag, cc_pts_ag, cc_pts_obs

def _get_rowwise_agent_obstacle_diffs_(car_dimensions, bicycle_dimensions, traj_agent: np.ndarray, traj_obstacles: List[np.ndarray]):
    n_circle_centers_agent = len(car_dimensions.circle_centers)
    n_circle_centers_obstacle = len(bicycle_dimensions.circle_centers)

    traj_agent, traj_obstacles = _pad_trajectories(traj_agent, traj_obstacles)

    rows_per_frame_ag, cc_pts_ag = _combine_rowwise_repeat(
        [tr[:, :2] for tr in car_trajectory_to_collision_point_trajectories(traj_agent, car_dimensions)], repeats=1)
    rows_per_frame_ag *= n_circle_centers_obstacle * len(traj_obstacles)
    cc_pts_ag = np.repeat(cc_pts_ag, len(traj_obstacles) * n_circle_centers_obstacle, axis=0)

    rows_per_frame_obs, cc_pts_obs = _combine_rowwise_repeat(
        [cc_tr[:, :2] for tr in traj_obstacles for cc_tr in
         car_trajectory_to_collision_point_trajectories(tr, bicycle_dimensions)], repeats=n_circle_centers_agent)

    assert rows_per_frame_ag == rows_per_frame_obs, (
        f"Mismatch in rows per frame: agent ({rows_per_frame_ag}) vs obstacles ({rows_per_frame_obs})")

    return rows_per_frame_ag, cc_pts_ag, cc_pts_obs

def _offset_trajectories_by_frames(trajs: List[np.ndarray], offsets: Union[List[int], np.ndarray]) -> List[np.ndarray]:
    out = []

    for traj in trajs:
        for idx_offset in offsets:
            if idx_offset < 0:
                obst2 = np.concatenate([traj[-idx_offset:], np.repeat(traj[-1:, :], repeats=-idx_offset, axis=0)],
                                       axis=0)
            elif idx_offset > 0:
                obst2 = np.concatenate([np.repeat(traj[0:1], repeats=idx_offset, axis=0), traj[:-idx_offset]], axis=0)
            else:
                obst2 = traj
            out.append(obst2)

    return out


def check_collision_moving_cars(car_dimensions: CarDimensions, traj_agent: np.ndarray, path_agent_detailed: np.ndarray,
                                traj_obstacles: List[np.ndarray], frame_window: int = 0) -> Optional[
    Tuple[float, float, float]]:
    if len(traj_obstacles) == 0:
        return None

    offsets = np.array(range(-frame_window, frame_window + 1, 1))
    traj_obstacles = _offset_trajectories_by_frames(traj_obstacles, offsets=offsets)

    # min_distance calculation below applies if it is assumed that both vehicle are cars
    min_distance = 2 * car_dimensions.radius

    rows_per_frame, cc_pts_ag, cc_pts_obs = _get_rowwise_diffs(car_dimensions, traj_agent, traj_obstacles)

    mask = np.linalg.norm(cc_pts_ag - cc_pts_obs, axis=1) <= min_distance
    # print("Total point pairs to collision-check:", len(diff_pts))
    first_row_idx = np.argmax(mask)

    if not mask[first_row_idx]:
        # no collision
        return None

    # first_frame_idx = first_row_idx // rows_per_frame

    # find the first position where the collision occurs :
    # get the circle position from the collision
    obstacle_position = cc_pts_obs[first_row_idx]
    agent_ccs = np.concatenate([tr[:, :2] for tr in car_trajectory_to_collision_point_trajectories(path_agent_detailed, car_dimensions)])

    # compute difference of entire agent trajectory with the obstacle position
    mask = np.linalg.norm(obstacle_position - agent_ccs, axis=1) <= min_distance

    # get the earliest
    first_frame_idx = np.argmax(mask) % len(path_agent_detailed)

    if first_frame_idx >= len(path_agent_detailed):
        return None

    x, y = path_agent_detailed[first_frame_idx, :2]
    return x, y, first_frame_idx

def check_collision_moving_bicycle(car_dimensions: CarDimensions, bicycle_dimensions: CarDimensions, traj_agent: np.ndarray, path_agent_detailed: np.ndarray,
                                traj_obstacles: List[np.ndarray], frame_window: int = 0) -> Optional[
    Tuple[float, float, float]]:
    if len(traj_obstacles) == 0:
        return None

    offsets = np.array(range(-frame_window, frame_window + 1, 1))
    traj_obstacles = _offset_trajectories_by_frames(traj_obstacles, offsets=offsets)

    # Instead of 2 times car_dimensions.raidus, we add the radius of car_dimensions + radius of bicycle_dimensions
    # min_distance = 2 * car_dimensions.radius
    min_distance = car_dimensions.radius + bicycle_dimensions.radius

    rows_per_frame, cc_pts_ag, cc_pts_obs = _get_rowwise_agent_obstacle_diffs_(car_dimensions, bicycle_dimensions, traj_agent, traj_obstacles)

    mask = np.linalg.norm(cc_pts_ag - cc_pts_obs, axis=1) <= min_distance
    # print("Total point pairs to collision-check:", len(diff_pts))
    first_row_idx = np.argmax(mask)

    if not mask[first_row_idx]:
        # no collision
        return None

    # first_frame_idx = first_row_idx // rows_per_frame

    # find the first position where the collision occurs :
    # get the circle position from the collision
    obstacle_position = cc_pts_obs[first_row_idx]
    agent_ccs = np.concatenate([tr[:, :2] for tr in car_trajectory_to_collision_point_trajectories(path_agent_detailed, car_dimensions)])

    # compute difference of entire agent trajectory with the obstacle position
    mask = np.linalg.norm(obstacle_position - agent_ccs, axis=1) <= min_distance

    # get the earliest
    first_frame_idx = np.argmax(mask) % len(path_agent_detailed)

    if first_frame_idx >= len(path_agent_detailed):
        return None

    x, y = path_agent_detailed[first_frame_idx, :2]
    return x, y, first_frame_idx

def get_cutoff_curve_by_position_idx(points: np.ndarray, x: float, y: float, radius: float = 0.001) -> int:
    points_diff = points[:, :2].copy()
    points_diff[:, 0] -= x
    points_diff[:, 1] -= y

    points_dist = np.linalg.norm(points_diff, axis=1) <= radius
    first_idx = np.argmax(points_dist)

    if not points_dist[first_idx]:
        # no cutoff
        return points

    return first_idx
