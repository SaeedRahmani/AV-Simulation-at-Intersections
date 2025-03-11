import math
from typing import Tuple, Dict, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Rectangle

from lib.a_star import AStar
from lib.car_dimensions import CarDimensions
from lib.linalg import create_2d_transform_mtx, transform_2d_pts
from lib.motion_primitive import MotionPrimitive
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.scenario import Scenario


def draw_scenario(scenario: Scenario, mps: Dict[str, MotionPrimitive], car_dimensions: CarDimensions,
                  search: MotionPrimitiveSearch, ax,
                  draw_obstacles=True, draw_goal=True, draw_car=True, draw_mps=True, draw_collision_checking=True,
                  draw_car2=True, draw_mps2=True, mp_name='right1'):
    if draw_obstacles:
        for obstacle in scenario.obstacles:
            # obstacle.draw(ax, color=(0.5, 0.5, 0.5))
            obstacle.draw(ax, color='b', hidden_color=(1.0, 1.0, 1.0, 0.0))

    if draw_goal:
        scenario.goal_area.draw(ax, color=(1, 0.8, 0.8))
        draw_point_arrow(scenario.goal_point, ax, color='r')

    # For Start Point:

    draw_car_func = globals()['draw_car']

    if draw_car:
        draw_car_func(start=scenario.start, car_dimensions=car_dimensions, ax=ax, color='g',
                      draw_collision_circles=not draw_collision_checking)

    # Draw Motion Primitives
    if draw_mps:
        for mp_name in mps.keys():
            mp_points = search.motion_primitive_at(mp_name, configuration=scenario.start)
            ax.plot(mp_points[:, 0], mp_points[:, 1], color='g')
            draw_point_arrow(mp_points[-1], ax=ax, color='g')

    # For One Random Neighbor Of The Start Point:
    if draw_car2 or draw_mps2:
        neighbor = tuple(search.motion_primitive_at(mp_name, configuration=scenario.start)[-1].tolist())

        # Draw Car:
        if draw_car2:
            draw_car_func(start=neighbor, car_dimensions=car_dimensions, ax=ax, color='b', draw_collision_circles=False)

        # Draw Motion Primitives
        if draw_mps2:
            for mp_name in mps.keys():
                mp_points = search.motion_primitive_at(mp_name, configuration=neighbor)
                ax.plot(mp_points[:, 0], mp_points[:, 1], color='b')
                draw_point_arrow(mp_points[-1], ax=ax, color='b')

    # Draw Collision Checking Points Of Random Motion Primitive As Circles With Radius
    if draw_collision_checking:
        cc_points = search.collision_checking_points_at(mp_name, configuration=scenario.start)
        colormap = plt.get_cmap('Set2')
        for i, cc_points_this in enumerate(np.array_split(cc_points, len(car_dimensions.circle_centers))):
            draw_circles(cc_points_this, radius=car_dimensions.radius, ax=ax, color=colormap(i))


def draw_point_arrow(point: Tuple[float, float, float], ax, color=None):
    x, y, theta = point
    u = np.cos(theta)
    v = np.sin(theta)
    ax.quiver(x, y, u, v, color=color)


def draw_circles(points: np.ndarray, radius: float, ax, color=None, marker=None):
    ax.add_collection(
        EllipseCollection(widths=radius * 2, heights=radius * 2, angles=0, units='xy', edgecolors=color,
                          facecolors='none', offsets=points[:, :2],
                          offset_transform=ax.transData))
    ax.scatter(points[:, 0], points[:, 1], color=color, marker=marker)


def draw_car(start: Tuple[float, float, float], car_dimensions: CarDimensions, ax, steer=0.0, color='b',
             draw_collision_circles=False):
    # Vehicle parameters
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]

    x, y, yaw = start

    center_offset_y, center_offset_x = car_dimensions.center_point_offset

    width, length = car_dimensions.bounding_box_size

    outline = np.array(
        [[center_offset_y - length / 2, center_offset_y + length / 2, center_offset_y + length / 2,
          center_offset_y - length / 2, center_offset_y - length / 2],
         [width / 2 + center_offset_x, width / 2 + center_offset_x, - width / 2 + center_offset_x,
          -width / 2 + center_offset_x, width / 2 + center_offset_x]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += car_dimensions.distance_back_to_front_wheel
    fl_wheel[0, :] += car_dimensions.distance_back_to_front_wheel

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    ax.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), color)
    ax.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), color)
    ax.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), color)
    ax.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), color)
    ax.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), color)
    ax.plot(x, y, "*", color=color)

    if draw_collision_circles:
        circle_center_mtx = create_2d_transform_mtx(*start)
        circle_centers = transform_2d_pts(start[2], circle_center_mtx, car_dimensions.circle_centers)

        draw_circles(circle_centers, radius=car_dimensions.radius, ax=ax, color=color)
        ax.scatter(circle_centers[:, 0], circle_centers[:, 1], color=color)

def draw_bicycle(start: Tuple[float, float, float], car_dimensions: CarDimensions, ax, steer=0.0, color='b',
             draw_collision_circles=False):
    # Vehicle parameters
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.015  # [m]
    TREAD = 0  # [m]

    x, y, yaw = start

    center_offset_y, center_offset_x = car_dimensions.center_point_offset

    width, length = car_dimensions.bounding_box_size

    outline = np.array(
        [[center_offset_y - length / 2, center_offset_y + length / 2, center_offset_y + length / 2,
          center_offset_y - length / 2, center_offset_y - length / 2],
         [width / 2 + center_offset_x, width / 2 + center_offset_x, - width / 2 + center_offset_x,
          -width / 2 + center_offset_x, width / 2 + center_offset_x]])

    # create fr_wheel: assumed to be front_wheel instead of front_right_wheel in draw_car
    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    # create rr_wheel: assumed to be rear_wheel instead of rear_right_wheel in draw_car
    rr_wheel = np.copy(fr_wheel)


    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += car_dimensions.distance_back_to_front_wheel

    fr_wheel = (fr_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y

    ax.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), color)
    ax.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), color)
    ax.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), color)
    ax.plot(x, y, "*", color=color)

    if draw_collision_circles:
        circle_center_mtx = create_2d_transform_mtx(*start)
        circle_centers = transform_2d_pts(start[2], circle_center_mtx, car_dimensions.circle_centers)

        draw_circles(circle_centers, radius=car_dimensions.radius, ax=ax, color=color)
        ax.scatter(circle_centers[:, 0], circle_centers[:, 1], color=color)

def draw_astar_search_points(search: Union[MotionPrimitiveSearch, AStar], ax, visualize_heuristic: bool,
                             visualize_cost_to_come: bool):
    debug_points = np.array([p.node for p in search.debug_data])

    if len(search.debug_data) == 0:
        return

    if visualize_heuristic or visualize_cost_to_come:
        c = np.zeros((len(search.debug_data)))

        if visualize_heuristic:
            c += np.array([p.h for p in search.debug_data])

        if visualize_cost_to_come:
            c += np.array([p.g for p in search.debug_data])

    else:
        c = None

    return ax.scatter(debug_points[:, 0], debug_points[:, 1], c=c, cmap='viridis_r')
