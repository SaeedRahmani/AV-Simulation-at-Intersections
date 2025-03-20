import json
import math
import sys
from typing import Tuple, List, Optional

import cvxpy
import numpy as np

from lib.car_dimensions import CarDimensions
from lib.simulation import State, Simulation
from lib.trajectories import calc_nearest_index, calc_nearest_index_in_direction

with open('../config/mpc_config.json', 'r') as f:
    config = json.load(f)

# Extract parameters from config
NX = config['NX']
NU = config['NU']
T = config['T']
w_perp = config['w_perp']
w_para = config['w_para']
R = np.diag(config['R'])  # input cost matrix
Rd = np.diag(config['Rd'])  # input difference cost matrix
Q_v_yaw = np.diag(config['Q_v_yaw'])  # state cost matrix [v, yaw]
Qf = np.diag(config['Qf']) * T  # state final matrix [x, y, v, yaw]
GOAL_DIS = config['GOAL_DIS']  # goal distance
STOP_SPEED = config['STOP_SPEED']  # stop speed
MAX_TIME = config['MAX_TIME']  # max simulation time

# iterative parameter
MAX_ITER = config['MAX_ITER']  # Max iteration
DU_TH = config['DU_TH']  # iteration finish param

MAX_DSTEER = np.deg2rad(config['MAX_DSTEER'])  # maximum steering speed [rad/s]
MAX_ACCEL = config['MAX_ACCEL']  # maximum accel [m/ss]
MAX_DECEL = config['MAX_DECEL']  # maximum deceleration [m/ss]


class MPCSolutionNotFoundException(Exception):
    pass


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def _get_linear_model_matrix(v, phi, delta, dt, L):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = dt * math.cos(phi)
    A[0, 3] = - dt * v * math.sin(phi)
    A[1, 2] = dt * math.sin(phi)
    A[1, 3] = dt * v * math.cos(phi)
    A[3, 2] = dt * math.tan(delta) / L

    B = np.zeros((NX, NU))
    B[2, 0] = dt
    B[3, 1] = dt * v / (L * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = dt * v * math.sin(phi) * phi
    C[1] = - dt * v * math.cos(phi) * phi
    C[3] = - dt * v * delta / (L * math.cos(delta) ** 2)

    return A, B, C


def _get_nparray_from_matrix(x):
    return np.array(x).flatten()


def _calc_ref_trajectory(state, cx, cy, cyaw, dl, dt, start_idx, ov):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    start_idx = calc_nearest_index_in_direction(state, cx, cy, start_index=start_idx, forward=True)

    dref[:, :T + 1] = 0.0  # steer operational point should be 0.0 for indices 0 to T

    if ov is None:
        ov = np.ones((T + 1,)) * max(state.v, 10 / 3.6)

    travel = np.cumsum(np.abs(ov) * dt)
    idx = np.rint(travel / dl).astype(int)
    idx = np.minimum(idx + start_idx, ncourse - 1)

    xref[0, :T + 1] = cx[idx]
    xref[1, :T + 1] = cy[idx]
    # we skip the velocity - we don't need it (Q must have a 0 for it)
    xref[3, :T + 1] = cyaw[idx]

    reaches_end = idx == ncourse - 1

    return xref, start_idx, dref, reaches_end


def _predict_motion(x0, oa, od, xref, car_dimensions: CarDimensions, dt):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    simulation = Simulation(initial_state=state, car_dimensions=car_dimensions, sample_time=dt)
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = simulation.step(ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def _get_xy_cost_mtx_for_orientation(angle: float):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c ** 2, c * s],
        [c * s, s ** 2]
    ])


def _linear_mpc_control(xref, xbar, x0, dref, reaches_end, dt, car_dimensions: CarDimensions, speed):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    :param reaches_end:
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    L = car_dimensions.distance_back_to_front_wheel
    ## Add a cost for colliding the road boundries for cases we cannot follow the ref traj
    for t in range(T + 1):
        if t > 0:
            if not reaches_end[t]:
                # penalize difference from reference perpendicular to yaw strongly
                ref_yaw_perp = xref[3, t] + 0.5 * np.pi
                cost += cvxpy.quad_form(xref[:2, t] - x[:2, t], _get_xy_cost_mtx_for_orientation(ref_yaw_perp) * w_perp)

                # penalize difference from reference parallel to yaw weakly
                ref_yaw = xref[3, t]
                cost += cvxpy.quad_form(xref[:2, t] - x[:2, t], _get_xy_cost_mtx_for_orientation(ref_yaw) * w_para)

                # penalize velocity and yaw itself
                cost += cvxpy.quad_form(xref[2:, t] - x[2:, t], Q_v_yaw)
            else:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Qf)

        if t < T:
            A, B, C = _get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t], dt=dt, L=L)
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if reaches_end[t]:
                cost += cvxpy.quad_form(u[:, t], np.diag([10., 10.]))
            else:
                cost += cvxpy.quad_form(u[:, t], R)

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * dt]

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= speed]
    constraints += [x[2, :] >= Simulation.MIN_SPEED]
    constraints += [u[0, :] <= MAX_ACCEL]
    constraints += [u[0, :] >= MAX_DECEL]
    constraints += [cvxpy.abs(u[1, :]) <= Simulation.MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = _get_nparray_from_matrix(x.value[0, :])
        oy = _get_nparray_from_matrix(x.value[1, :])
        ov = _get_nparray_from_matrix(x.value[2, :])
        oyaw = _get_nparray_from_matrix(x.value[3, :])
        oa = _get_nparray_from_matrix(u.value[0, :])
        odelta = _get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc...", file=sys.stderr)
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def _iterative_linear_mpc_control(x0, oa, od, state, cx, cy, cyaw, dl, dt, target_ind, car_dimensions: CarDimensions, speed):
    """
    MPC contorl with updating operational point iteraitvely
    :param state:
    :param cx:
    :param cy:
    :param cyaw:
    :param dl:
    :param target_ind:
    """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    ov = None

    for _ in range(MAX_ITER):
        xref, target_ind, dref, reaches_end = _calc_ref_trajectory(state, cx, cy, cyaw, dl, dt, target_ind, ov)
        xbar = _predict_motion(x0, oa, od, xref, car_dimensions=car_dimensions, dt=dt)
        # poa, pod = oa, od
        oa, od, ox, oy, oyaw, ov = _linear_mpc_control(xref, xbar, x0, dref, reaches_end, dt, car_dimensions, speed)
        # du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        # if du <= DU_TH:
        #     break
    # else:
    #     print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov, xref, target_ind


class MPC:
    def __init__(self, cx: np.ndarray, cy: np.ndarray, cyaw: np.ndarray, dl: float, speed: float, car_dimensions: CarDimensions,
                 dt: float = 0.2):
        """
        Simulation
        cx: course x position list
        cy: course y position list
        cyaw: course yaw position list
        dl: course tick [m]
        dt: delta time [s]
        """

        self.cx = cx
        self.cy = cy

        cyaw = smooth_yaw(cyaw)
        self.cyaw = cyaw
        self.dl = dl
        self.dt = dt
        self.car_dimensions = car_dimensions
        self.speed = speed

        self.goal: Tuple[float, float] = cx[-1], cy[-1]

        self.target_ind: int = 0

        self.odelta: Optional[List[float]] = None
        self.oa: Optional[List[float]] = None

        self.di: float = 0.0
        self.ai: float = 0.0

        # cyaw = smooth_yaw(cyaw)

    def set_trajectory_fromarray(self, trajectory: np.ndarray):
        self.cx = trajectory[:, 0]
        self.cy = trajectory[:, 1]
        self.cyaw = trajectory[:, 2]

    def step(self, state: State) -> Tuple[float, float]:
        # initial yaw compensation
        # if state.yaw - cyaw[0] >= math.pi:
        #     state.yaw -= math.pi * 2.0
        # elif state.yaw - cyaw[0] <= -math.pi:
        #     state.yaw += math.pi * 2.0

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        self.oa, self.odelta, self.ox, self.oy, self.oyaw, self.ov, self.xref, self.target_ind = \
            _iterative_linear_mpc_control(x0, self.oa, self.odelta, state, self.cx, self.cy,
                                          self.cyaw, self.dl, self.dt, self.target_ind,
                                          car_dimensions=self.car_dimensions, speed=self.speed)

        if self.odelta is not None:
            self.di, self.ai = self.odelta[0], self.oa[0]
        else:
            self.ai = MAX_DECEL

        return self.di, self.ai

    def get_current_xref_deviation(self):
        ref_point = np.array([self.cx[self.target_ind], self.cy[self.target_ind]])
        true_point = np.array([self.ox[0], self.oy[0]])

        ref_yaw_perp = self.cyaw[self.target_ind] + np.pi / 2
        diff_vect = ref_point - true_point
        ref_dir_normal = np.array([np.cos(ref_yaw_perp) * diff_vect[0], np.sin(ref_yaw_perp) * diff_vect[1]])
        return np.linalg.norm(ref_dir_normal)

    def is_goal(self, state: State) -> bool:
        # check goal
        dx = state.x - self.goal[0]
        dy = state.y - self.goal[1]
        d = math.hypot(dx, dy)

        isgoal = (d <= GOAL_DIS)

        if abs(self.target_ind - len(self.cx)) >= 5:
            isgoal = False

        isstop = (abs(state.v) <= STOP_SPEED)

        if isgoal and isstop:
            return True

        return False
