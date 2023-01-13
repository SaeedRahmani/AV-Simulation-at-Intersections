import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Reference paper:
# https://uwspace.uwaterloo.ca/bitstream/handle/10012/16847/Ahmadi_Behnaz.pdf?sequence=1
# see page 29/80 to see the bicycle model used

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

        # TO DO: Tune this parameters
        self.scale_factor = 0.3
        self.L = (1.45 + 1.41) * self.scale_factor

    def reset(self):
        self.xc = 0 # vehicle x position
        self.yc = 0 # vehicle y position
        self.theta = 0 # angle between vehicle frame and x-axis
        self.delta = 0 # angle between vehicle steering wheel and vehicle frame

class Bicycle(Bicycle):
    def step(self, v, delta):
        # ==================================
        #  Implement kinematic model here
        # ==================================

        # sampling time
        t_sample = 10e-3

        # implementing the differential equations
        xc_dot = v * np.cos(self.theta)
        yc_dot = v * np.sin(self.theta)
        theta_dot = (v / self.L) * np.tan(delta)

        # update equations using the sampling time
        self.xc += xc_dot * t_sample
        self.yc += yc_dot * t_sample
        self.theta += theta_dot * t_sample

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sample_time = 0.01
    time_end = 1
    model = Bicycle()
    model1 = Bicycle()
    model2 = Bicycle()

    t_data = np.arange(0, time_end, sample_time)
    x_data = np.zeros_like(t_data)
    y_data = np.zeros_like(t_data)
    x_data1 = np.zeros_like(t_data)
    y_data1 = np.zeros_like(t_data)
    x_data2 = np.zeros_like(t_data)
    y_data2 = np.zeros_like(t_data)

    for i in range(t_data.shape[0]):
        x_data[i] = model.xc
        y_data[i] = model.yc
        x_data1[i] = model1.xc
        y_data1[i] = model1.yc
        x_data2[i] = model2.xc
        y_data2[i] = model2.yc
        model.step(1.5, 2.35)
        model1.step(1.5, 0)
        model2.step(1.5, 0.78)

    plt.axis('equal')
    plt.scatter(x_data, y_data, label='Delta = 135 degree')
    plt.scatter(x_data1, y_data1, label='Delta = 0 degree')
    plt.scatter(x_data2, y_data2, label='Delta = 45 degree')
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
