from lib.car_dimensions import BicycleModelDimensions  # Import the car_dimensions from the appropriate module
from lib.motion_primitive import load_motion_primitives  # Import load_motion_primitives from the appropriate module
#from lib.state import State  # Import the State class from the appropriate module
from lib.simulation import Simulation, History, HistorySimulation  # Import the classes from the simulation module
from lib.mpc import MPC, MAX_ACCEL
from lib.collision_avoidance import check_collision_moving_cars, get_cutoff_curve_by_position_idx  # Import the collision functions
from envs.intersection import intersection  # Import the intersection function from the intersection module
from lib.mp_search import MotionPrimitiveSearch  # Import MotionPrimitiveSearch from the appropriate module
from lib.other_agents_prediction import OtherAgentsPrediction  # Import MovingObstaclesPrediction from the appropriate module
from envs.intersection import intersection

class Ego_instance:
    def __init__(self, start_position: int, turn_indicator: int):
        self.start_position = start_position
        self.turn_indicator = turn_indicator
        self.scenario = intersection(self.start_position, self.turn_indicator)
        
        # self.car_dimensions = BicycleModelDimensions()  # Initialize the car_dimensions
        # self.state = None  # Initialize the state to None
        # self.delta = None  # Initialize the steering angle to None
        # self.acceleration = None  # Initialize the acceleration to None
        # self.Simulation = Simulation()  # Initialize the Simulation
        # self.History = History()  # Initialize the History
        # self.HistorySimulation = HistorySimulation()  # Initialize the HistorySimulation
        # self.traj_agent_idx = 0  # Initialize the traj_agent_idx
        # self.tmp_trajectory = None  # Initialize the tmp_trajectory
        # self.loop_runtimes = []  # Initialize the loop_runtimes
        # self.EXTRA_CUTOFF_MARGIN = 5  # Initialize the EXTRA_CUTOFF_MARGIN
        # self.FRAME_WINDOW = 10  # Initialize the FRAME_WINDOW


    # def create_scenario(self):
    #     self.scenario = intersection(self.start_position, self.turn_indicator)
    #     return self.scenario

    # def generate_path(self):
    #     __, __, self.trajectory_full = MotionPrimitiveSearch(self.scenario).run(debug=False)  # Assume MotionPrimitiveSearch takes scenario as input
    #     # Initialize the state based on the first point in the trajectory
    #     # self.state = State(x=self.trajectory_full[0, 0], y=self.trajectory_full[0, 1], yaw=self.trajectory_full[0, 2], v=0.0)
    #     return self.trajectory_full

    def predict_obstacles(self, moving_obstacles, DT, TIME_HORIZON):
        # Predict the trajectories of the moving obstacles
        self.trajs_moving_obstacles = [np.vstack(OtherAgentsPrediction(*o.get(), sample_time=DT, 
                                                                           car_dimensions=self.car_dimensions)
                                                 .state_prediction(TIME_HORIZON)).T for o in moving_obstacles]

    def cut_trajectory(self, trajectory_res, trajectory_full):
        # Check for collision and cutoff the trajectory if necessary
        collision_xy = check_collision_moving_cars(self.car_dimensions, trajectory_res, self.trajectory_full, 
                                                   self.trajs_moving_obstacles, frame_window=self.FRAME_WINDOW)

        # cutoff the curve such that it ends right before the collision (and some margin)
        if collision_xy is not None:
            cutoff_idx = get_cutoff_curve_by_position_idx(trajectory_full, collision_xy[0], 
                                                          collision_xy[1]) - self.EXTRA_CUTOFF_MARGIN
            cutoff_idx = max(self.traj_agent_idx + 1, cutoff_idx)
            self.tmp_trajectory = trajectory_full[:cutoff_idx]
        else:
            self.tmp_trajectory = trajectory_full

        # pass the cut trajectory to the MPC
        self.MPC.set_trajectory_fromarray(self.tmp_trajectory)

    def calculate_optimal_control(self):
        # Calculate the optimal steering angle and acceleration
        self.delta, self.acceleration = self.MPC.step(self.state)


# Examples:
# ego_vehicle_1 = MPCInstance(start_position=1, turn_indicator=2)
# car_dimensions = ego_vehicle_1.car_dimensions
# motion_primitives = ego_vehicle_1.motion_primitives
# mpc = ego_vehicle_1.MPC
# simulation = ego_vehicle_1.Simulation
# history = ego_vehicle_1.History
# history_simulation = ego_vehicle_1.HistorySimulation
# traj_agent_idx = ego_vehicle_1.traj_agent_idx
# tmp_trajectory = ego_vehicle_1.tmp_trajectory
# loop_runtimes = ego_vehicle_1.loop_runtimes
# ego_vehicle_1.create_scenario()
# ego_vehicle_1.create_trajectory()
# ego_vehicle_1.predict_obstacles(moving_obstacles, DT, TIME_HORIZON)
# ego_vehicle_1.cut_trajectory(trajectory_res, trajectory_full)
# ego_vehicle_1.calculate_optimal_control()
# state = ego_vehicle_1.state

# trajectory = ego_vehicle_1.trajectory
# trajs_moving_obstacles = ego_vehicle_1.trajs_moving_obstacles
# tmp_trajectory = ego_vehicle_1.tmp_trajectory
# delta = ego_vehicle_1.delta
# acceleration = ego_vehicle_1.acceleration

# IMports:
# BicycleModelDimensions from lib.car_dimensions
# load_motion_primitives from lib.motion_primitive
# State from lib.state
# MPC, Simulation, History, HistorySimulation from lib.simulation
# check_collision_moving_cars, get_cutoff_curve_by_position_idx from lib.collision
# MovingObstaclesPrediction from lib.moving_obstacles
# MotionPrimitiveSearch from lib.motion_primitive
# intersection from intersection

