import sys

from cvxpy import length

sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
from main.lib.obstacles import BoxObstacle
from main.lib.scenario import Scenario
from main.lib.plot_obstacles import plot_intersection
from main.lib.car_dimensions import CarDimensions, BicycleModelDimensions

class ArterialMultiLanes:
    def __init__(self, num_lanes=2, goal_lane=1):
        self.num_lanes = num_lanes
        self.goal_lane = goal_lane
        self.width_road = 4
        self.width_pavement = 5
        self.length = 50
        self.allowed_goal_theta_difference = np.pi / 16
        self.goal_lane_adjustment = goal_lane - 1

    def validate_lanes(self):
        if self.num_lanes < 1:
            print("Number of lanes should be at least 1")
            return False
        if self.goal_lane > self.num_lanes:
            print("Goal lane should be less than or equal to the number of lanes")
            return False
        return True

    def calculate_offsets(self):
        # to determine the location of the pavements
        left_pavement = - (self.num_lanes * self.width_road / 2) - (self.width_pavement / 2)
        right_pavement = (self.num_lanes * self.width_road / 2) + (self.width_pavement / 2)
        lane_offset = (self.num_lanes // 2 - 0.5) * self.width_road - self.goal_lane_adjustment * self.width_road
        if self.num_lanes % 2 != 0:
            lane_offset += self.width_road / 2
        return left_pavement, right_pavement, lane_offset

    def create_scenario(self, moving_obstacles=False, moving_obstacles_trajectory=None,
                        spawn_location_x=None, spawn_location_y=None,
                        av_location_x=None, av_location_y=None):
        if not self.validate_lanes():
            return None

        left_pavement, right_pavement, lane_offset = self.calculate_offsets()
        start = (self.width_road * (self.num_lanes / 2 - 0.5), -self.length / 2, np.pi/2)
        goal = (lane_offset, self.length / 2, np.pi/2)

        car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)
        goal_area = BoxObstacle(xy_width=(car_dimensions.bounding_box_size[0], car_dimensions.bounding_box_size[1]), height=1, xy_center=(goal[0], goal[1]))

        # TO SET GOAL AREA AS A BOX
        # goal_area = BoxObstacle(xy_width=(self.width_road, self.width_road), height=1, xy_center=(goal[0], goal[1]))

        if moving_obstacles is True:
            start = (av_location_x, av_location_y, np.pi / 2)
            lower_side_obstacle = moving_obstacles_trajectory[0][0][1]
            upper_side_obstacle = moving_obstacles_trajectory[0][-1][1]
            # to set the length of the obstacle as the sum of the length of the obstacle and the predicted trajectory
            length_obstacle_plus_predicted_trajectory = upper_side_obstacle - lower_side_obstacle
            obstacles = [
                BoxObstacle(xy_width=(self.width_pavement, self.length), height=1, xy_center=(left_pavement, 0)),
                BoxObstacle(xy_width=(self.width_pavement, self.length), height=0.1, xy_center=(right_pavement, 0)),
                BoxObstacle(xy_width=(1.64, length_obstacle_plus_predicted_trajectory), height=0.1, xy_center=(spawn_location_x, spawn_location_y))
            ]
        else:
            obstacles = [
                BoxObstacle(xy_width=(self.width_pavement, self.length), height=1, xy_center=(left_pavement, 0)),
                BoxObstacle(xy_width=(self.width_pavement, self.length), height=0.1, xy_center=(right_pavement, 0))]
        return Scenario(
            start=start,
            goal_point=goal,
            goal_area=goal_area,
            allowed_goal_theta_difference=self.allowed_goal_theta_difference,
            obstacles=obstacles
        )

if __name__ == '__main__':
    arterial = ArterialMultiLanes(num_lanes=4, goal_lane=4)
    scenario = arterial.create_scenario()
    if scenario:
        plot_intersection(scenario)