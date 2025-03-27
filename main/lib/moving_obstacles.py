import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
import numpy as np
from typing import Tuple

import sys
sys.path.append('..')

from bicycle.main import Bicycle
import warnings

from lib.car_dimensions import CarDimensions, BicycleModelDimensions

def calculate_steering_angle_for_radius(radius, L=2.86) -> float:
    """
    Calculate the required constant steering angle for a vehicle to drive in a circle of a given radius.
    
    :param radius: The radius of the circle in meters
    :return: Required steering angle in radians
    """
    curvature = 1 / radius  # Curvature is the inverse of the radius
    steering_angle = np.arctan(curvature * L)  # Steering angle calculation
    return steering_angle


class MovingObstacleRoundabout:
    def __init__(self, car_dimensions: CarDimensions, direction: int, turning: bool, speed: float, offset=None,
                 dt=10e-3, start_pos=2, end_pos=4):
        """
        Function that creates moving obstacles
        :param car_dimensions:
        :param direction: positive for moving to the right, negative for moving to the left
        :param turning: True for turning, False for going straight
        :param speed: sets the forward speed and is not bounded
        :param offset: sets the time in seconds it should start moving after the start of the simulation. None or 0 for no offset
        :param dt: the dt used in the simulator. !WARNING! is 10e-3 in the Bicycle model
        """
        self.direction = 1 if direction >= 0 else -1
        self.turning = turning
        self.speed = speed
        self.model = Bicycle(car_dimensions=car_dimensions, sample_time=dt)
        self.offset = None if offset is None else offset if offset > 0 else None  # None except if offset > 0
        self.dt = dt = 0.2
        self.start_pos = start_pos
        self.end_pos = end_pos
        # if abs(dt - 10e-3) > 1e-6:
        #     warnings.warn("The dt is most likely not compatible with the bicycle model!")
        self.counter = 0
        if self.direction == 1:
            self.model.xc = -30
            self.model.yc = -3
            self.model.theta = 0
            self.x_turn = -10
        else:
            self.model.xc = 30
            self.model.yc = 3
            self.model.theta = np.pi
            self.x_turn = 12

    # We are defining this function as a property because it basically returns 
    # steering angle as an attribute but we need to check some conditions and 
    # make some changes if necessary before returning it
    @property 
    def steering_angle(self) -> float:
        steering_angle = 0.0

        if self.turning is not True:
            return steering_angle
        
        elif self.direction == 1: #left to right
            if -7 <= self.model.xc <= -4 and self.model.yc < 0:
                steering_angle = -calculate_steering_angle_for_radius(5) 
                print(self.model.xc, self.model.yc) 
            if -3 < self.model.xc:
                steering_angle = calculate_steering_angle_for_radius(5)
            if self.model.yc > 0 and -5 <= self.model.xc <= -3:
                steering_angle = -calculate_steering_angle_for_radius(5)
            if self.model.xc <= -3 and self.model.yc > 0:
                self.model.theta = -np.pi
                steering_angle = 0
            
            
            # if ((self.x_turn + 5) < self.model.xc < (self.x_turn + 6)) and (self.model.yc < 0):
            #     steering_angle = -calculate_steering_angle_for_radius(4)
            # elif self.model.xc > (self.x_turn + 6):
            #     steering_angle = calculate_steering_angle_for_radius(4)
            # elif (self.x_turn +5) <self.model.xc < (self.x_turn + 6) and self.model.yc >0:
            #     steering_angle = -calculate_steering_angle_for_radius(4)
            # else:
            #     steering_angle = 0


        else:
            if 4 <= self.model.xc <= 7 and self.model.yc > 0:
                steering_angle = -calculate_steering_angle_for_radius(5) 
                print(self.model.xc, self.model.yc) 
            if  self.model.xc < 3:
                steering_angle = calculate_steering_angle_for_radius(5)
            if self.model.yc < 0 and 3 <= self.model.xc <= 5:
                steering_angle = -calculate_steering_angle_for_radius(5)
            if 3 <= self.model.xc and self.model.yc < 0:
                self.model.theta = 0
                steering_angle = 0

        return steering_angle

    @property
    def forward_velocity(self):
        if self.offset is None or self.counter > (self.offset / self.dt):
            forward_velocity = self.speed
        else:
            forward_velocity = 0
        return forward_velocity

    def step(self):
        steering_angle = self.steering_angle
        self.model.step(self.forward_velocity, steering_angle)
        self.counter += 1

    def get(self) -> Tuple[float, float, float, float, float, float]:
        acceleration = 0.0
        return self.model.xc, self.model.yc, self.forward_velocity, self.model.theta, acceleration, self.steering_angle

class MovingObstacleArterial:
    def __init__(self, car_dimensions: CarDimensions, x_init: float, y_init: float, speed: float, initial_speed: float, offset=None, dt=10e-3):
        """
        Function that creates moving obstacles that only go up
        :param car_dimensions:
        :param speed: sets the forward speed and is not bounded
        :param offset: sets the time in seconds it should start moving after the start of the simulation. None or 0 for no offset
        :param dt: the dt used in the simulator. !WARNING! is 10e-3 in the Bicycle model
        """
        self.speed = speed
        self.initial_speed = initial_speed
        self.model = Bicycle(car_dimensions=car_dimensions, sample_time=dt)
        self.offset = None if offset is None else offset if offset > 0 else None  # None except if offset > 0
        self.dt = dt
        self.counter = 0
        self.model.xc = x_init
        self.model.yc = y_init
        self.model.theta = np.pi / 2  # Facing up

    @property
    def steering_angle(self) -> float:
        return 0.0  # No steering, goes straight up

    @property
    def forward_velocity(self):
        if self.offset is None or self.counter > (self.offset / self.dt):
            forward_velocity = self.speed
        else:
            forward_velocity = self.initial_speed
        return forward_velocity

    def step(self):
        steering_angle = self.steering_angle
        self.model.step(self.forward_velocity, steering_angle)
        self.counter += 1

    def get(self) -> Tuple[float, float, float, float, float, float]:
        acceleration = 0.0
        return self.model.xc, self.model.yc, self.forward_velocity, self.model.theta, acceleration, self.steering_angle

class MovingObstacleTIntersection:
    def __init__(self, car_dimensions: CarDimensions, direction: int, turning: bool, speed: float, offset=None,
                 dt=10e-3):
        """
        Function that creates moving obstacles
        :param car_dimensions:
        :param direction: positive for moving to the right, negative for moving to the left
        :param turning: True for turning, False for going straight
        :param speed: sets the forward speed and is not bounded
        :param offset: sets the time in seconds it should start moving after the start of the simulation. None or 0 for no offset
        :param dt: the dt used in the simulator. !WARNING! is 10e-3 in the Bicycle model
        """
        self.direction = 1 if direction >= 0 else -1
        self.turning = turning
        self.speed = speed
        self.model = Bicycle(car_dimensions=car_dimensions, sample_time=dt)
        self.offset = None if offset is None else offset if offset > 0 else None  # None except if offset > 0
        self.dt = dt
        # if abs(dt - 10e-3) > 1e-6:
        #     warnings.warn("The dt is most likely not compatible with the bicycle model!")
        self.counter = 0
        if self.direction == 1:
            self.model.xc = -30
            self.model.yc = -3
            self.model.theta = 0
            self.x_turn = -10
        else:
            self.model.xc = 30
            self.model.yc = 3
            self.model.theta = np.pi
            self.x_turn = 12

    # We are defining this function as a property because it basically returns 
    # steering angle as an attribute but we need to check some conditions and 
    # make some changes if necessary before returning it
    @property 
    def steering_angle(self) -> float:
        steering_angle = 0.  # rad

        if self.turning is not True:
            return steering_angle

        if self.direction == 1:
            if self.model.xc >= self.x_turn and self.model.theta > (-np.pi / 2):
                steering_angle = -0.38  # steering angle right (short turn)
        else:
            if self.model.xc <= self.x_turn and self.model.theta < (3 * np.pi / 2):
                steering_angle = 0.19  # steering angle left (long turn)

        return steering_angle

    @property
    def forward_velocity(self):
        if self.offset is None or self.counter > (self.offset / self.dt):
            forward_velocity = self.speed
        else:
            forward_velocity = 0
        return forward_velocity

    def step(self):
        steering_angle = self.steering_angle
        self.model.step(self.forward_velocity, steering_angle)
        self.counter += 1

    def get(self) -> Tuple[float, float, float, float, float, float]:
        acceleration = 0.0
        return self.model.xc, self.model.yc, self.forward_velocity, self.model.theta, acceleration, self.steering_angle


if __name__ == "__main__":
    car_dimensions: CarDimensions = BicycleModelDimensions()
    obstacles = [MovingObstacleRoundabout(car_dimensions, 1, True, 25),
                 MovingObstacleRoundabout(car_dimensions, 1, False, 25),
                 MovingObstacleRoundabout(car_dimensions, -1, False, 25),
                 MovingObstacleRoundabout(car_dimensions, -1, True, 25)]

    length = 1000  # steps in the simulation
    colors = np.array([np.zeros(length), np.linspace(0, 1, length), np.ones(length)]).T.astype(float)
    positions = np.zeros((len(obstacles), length, 6))

    # Run the simulation
    for t in range(length):  # 5 seconds, because dt of bicycle model is 10e-3
        for i_obs, obstacle in enumerate(obstacles):
            obstacle.step()
            positions[i_obs, t] = obstacle.get()

    print(positions.shape)

    ###################### Create scatter animation with changing colors
    fig, ax = plt.subplots()

    # Create the initial plot
    x0 = positions[:, 0, 0]
    y0 = positions[:, 0, 1]
    scat = ax.scatter(x0, y0)
    mat, = ax.plot(x0, y0, 'o')

    # different color arrays
    colors_array = []
    combinations = [(0, 0), (0, 1), (1, 1)]
    for i in range(6):
        colors_array.append(
            np.array([np.linspace(*combinations[i % 3], length), np.linspace(*combinations[(i + 1) % 3], length),
                      np.linspace(*combinations[(i + 2) % 3], length)]).T.astype(float))

    # Setting to True gives every obstacle a different color ranges, but it becomes really messy
    # and somehow there are only 3 unique combinations
    different_colors = False


    def animate(i):
        """ Function that is called for every animation frame """
        # Set x and y data...
        interval = 5
        if different_colors:
            x = positions[:, :i + 1:interval, 0].flatten('C')
            y = positions[:, :i + 1:interval, 1].flatten('C')
        else:
            x = positions[:, :i + 1:interval, 0].flatten('F')
            y = positions[:, :i + 1:interval, 1].flatten('F')
            # x_car = ((i+1)%5) + 1
            # y_car = 0
            # mat.set_data(x_car, y_car)
            # mat.set_color((1, 0, 0))

        scat.set_offsets(np.array([x, y]).T)
        if different_colors:
            scat.set_facecolors(np.vstack((colors_array[0][:i + 1:interval], colors_array[1][:i + 1:interval],
                                           colors_array[2][:i + 1:interval], colors_array[5][:i + 1:interval])))
        else:
            scat.set_facecolors(np.repeat(colors[:i + 1:interval], positions.shape[0], axis=0))


    # Set the axis values
    ax.axis([-20, 20, -25, 5])
    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=0.2) # old 10e-3

    start = mlines.Line2D([], [], color=colors[0], marker='o', ls='', label='Start point')
    end = mlines.Line2D([], [], color=colors[-1], marker='o', ls='', label='End point')
    plt.legend(handles=[start, end])

    # Code to save the animation
    # FFwriter = animation.FFMpegWriter(fps=100)
    # ani.save('moving_obstacles_trajectory.mp4', writer=FFwriter)

    #plt.show()

    ################# Create animation: Not possible with changing colors
    # fig, ax = plt.subplots()
    #
    # # Create the initial plot
    # x0 = positions[:, 0, 0]
    # y0 = positions[:, 0, 1]
    # mat, = ax.plot(x0, y0, 'o')
    #
    # def animate(i):
    #     """ Function that is called for every animation frame """
    #     interval = 5
    #     x = positions[:, :i+1:interval, 0]
    #     y = positions[:, :i+1:interval, 1]
    #     mat.set_data(x, y)
    #     mat.set_color((0, 0, 1))
    #     return mat,
    #
    # # Set the axis values
    # ax.axis([-20, 20, -25, 5])
    # ani = animation.FuncAnimation(fig, animate, frames=600, interval=10e-3, repeat=True)

    # Code to save the animation
    # FFwriter = animation.FFMpegWriter(fps=100)
    # ani.save('moving_obstacles_trajectory.mp4', writer=FFwriter)

    # plt.show()

    #################3 Create plot
    plt.figure()
    for position_obstacle in positions:
        plt.scatter(position_obstacle[:, 0], position_obstacle[:, 1], c=colors)
    start = mlines.Line2D([], [], color=colors[0], marker='o', ls='', label='Start point')
    end = mlines.Line2D([], [], color=colors[-1], marker='o', ls='', label='End point')
    plt.legend(handles=[start, end])

    plt.axis('equal')

    # Code to save the plot
    # plt.savefig('moving_obstacles_trajectory.png')

    plt.show()
