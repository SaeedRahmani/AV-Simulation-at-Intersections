from dataclasses import dataclass, field
from typing import Tuple, List
from main.lib.obstacles import Obstacle


@dataclass
class Scenario:
    start: Tuple[float, float, float]
    goal_point: Tuple[float, float, float]
    goal_area: Obstacle
    allowed_goal_theta_difference: float
    obstacles: List[Obstacle]

# The Scenario class is defined as a Python data class. Data classes are a way of creating classes which primarily exist to hold values and don't contain much functionality. They're a useful way of grouping related data together without having to write boilerplate code for things like initialization and representation.
# When you create an instance of a data class, you pass in values for the attributes as arguments to the constructor.
# Example: 
# obstacle = Obstacle()  # assuming Obstacle can be instantiated without parameters
# start = (0.0, 0.0, 0.0)
# goal = (1.0, 1.0, 1.0)
# goal_area = Obstacle()
# allowed_difference = 0.5
# scenario = Scenario(start, [obstacle], goal_area, goal, allowed_difference)
# print(scenario.start)
# OR
# scenario.start = (1.0, 1.0, 1.0)

# The class does not return a value because it's not meant to perform a computation that would result in a meaningful output. Instead, it's meant to encapsulate related data into a single object for easier management and manipulation. However, when an instance of a class is created, it's "returned" from the constructor in the sense that the new object is the result of the constructor. But we don't usually think of this as the class "returning" a value.