# parameters.py
from dataclasses import dataclass

@dataclass
class ScenarioParameters:
    DT = 0.1  # Time step
    CENTERLINE_LOCATION = 0.0  # Centerline location for evaluation
    LENGTH = 44.0 # Length of the scenario
    WIDTH_ROAD = 4.0 # Width of a single road lane
    X_LOC_GOAL = 2.0  # x location of the goal, right side of the middle of the road
    Y_LOC_GOAL = 22.0  # y location of the goal
    X_LOC_EGO = 2.0  # Initial x location of the ego vehicle (AV), right side of the middle of the road
    X_LOC_CYCLIST_BUFFER = 0.0  # Initial x location of the cyclist
    Y_LOC_CYCLIST_BUFFER  = 9.7  # Initial x location of the cyclist

@dataclass
class ReasonParameters:
    REASONS_THRESHOLD = 0.7  # Threshold for reasons to trigger replan

@dataclass
class MPCParameters:
    TIME_HORIZON = 7.0  # Time horizon for predictions
    FRAME_WINDOW = 10  # Frame window for collision checking
    MAX_SPEED_FREEWAY = 30 / 3.6  # Maximum speed for freeway

@dataclass
class DriverParameters:
    DISTANCE_REF = 10.0  # Reference distance for driver patience
    DISTANCE_BUFFER = 2.0  # Buffer distance for driver patience
    TIME_THRESHOLD = 8.0  # Time threshold for driver patience

@dataclass
class CyclistParameters:
    DISTANCE_REF = 8.0  # Reference distance for cyclist patience
    DISTANCE_BUFFER = 2.0  # Buffer distance for cyclist patience
    TIME_THRESHOLD = 5.0  # Time threshold for cyclist patience
    SPEED = 5 / 3.6 # Cyclist speed [km/h]