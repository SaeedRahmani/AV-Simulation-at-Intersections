import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
from main.lib.obstacles import BoxObstacle
from main.lib.scenario import Scenario
from main.lib.plot_obstacles import plot_intersection

class ArterialMultiLanes:
    def __init__(self, num_lanes=2, goal_lane=1):
        self.num_lanes = num_lanes
        self.goal_lane = goal_lane
        self.width_road = 4
        self.width_pavement = 5
        self.length = 100
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
        left_pavement = - (self.num_lanes * self.width_road / 2) - (self.width_pavement / 2)
        right_pavement = (self.num_lanes * self.width_road / 2) + (self.width_pavement / 2)
        lane_offset = (self.num_lanes // 2 - 0.5) * self.width_road - self.goal_lane_adjustment * self.width_road
        if self.num_lanes % 2 != 0:
            lane_offset += self.width_road / 2
        return left_pavement, right_pavement, lane_offset

    def create_scenario(self):
        if not self.validate_lanes():
            return None

        left_pavement, right_pavement, lane_offset = self.calculate_offsets()
        start = (lane_offset, -self.length / 2, 0)
        goal = (lane_offset, self.length / 2, 0)
        goal_area = BoxObstacle(xy_width=(self.width_road, self.width_road), height=1, xy_center=(goal[0], goal[1]))

        obstacles = [
            BoxObstacle(xy_width=(self.width_pavement, self.length), height=1, xy_center=(left_pavement, 0)),
            BoxObstacle(xy_width=(self.width_pavement, self.length), height=0.1, xy_center=(right_pavement, 0))
        ]

        return Scenario(
            start=start,
            goal_point=goal,
            goal_area=goal_area,
            allowed_goal_theta_difference=self.allowed_goal_theta_difference,
            obstacles=obstacles,
        )

if __name__ == '__main__':
    arterial = ArterialMultiLanes(num_lanes=3, goal_lane=2)
    scenario = arterial.create_scenario()
    if scenario:
        plot_intersection(scenario)