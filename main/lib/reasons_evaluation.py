import numpy as np
import matplotlib.pyplot as plt

def evaluate_distance_to_obstacle(distance_buffer, distance_threshold, moving_obstacles,state):
    distance_to_car = np.linalg.norm(
        [moving_obstacles[0].get()[0] - state.x, moving_obstacles[0].get()[1] - state.y])
    if distance_to_car < (distance_threshold + distance_buffer):
        reasons_cyclist = np.exp(0.2 * (distance_to_car - (distance_threshold + distance_buffer)))
        return reasons_cyclist
    else:
        reasons_cyclist = 1
        return reasons_cyclist

def evaluate_time_following(reasons, DT, distance_buffer, distance_threshold, time_threshold, moving_obstacles,state, time_passed):
    # reasons = 'driver_reasons' or 'cyclist_reasons', depending on the evaluation
    if reasons == 'driver_reasons':
        distance_to_obstacle = np.linalg.norm(
            [moving_obstacles[0].get()[0] - state.x, moving_obstacles[0].get()[1] - state.y])
        if distance_to_obstacle < (distance_threshold + distance_buffer):
            time_passed += DT
            # print(time_passed)
            # print('time passed driver is: {}'.format(time_passed))
            # print('distance to bicycle is: {}'.format(distance_to_obstacle))
            if time_passed >= time_threshold:
                reasons_driver = 1 / np.exp(0.2 * (time_passed - time_threshold))
                return reasons_driver, time_passed
        else:
            reasons_driver = 1
            return reasons_driver, time_passed
        # Ensure the function always returns something
        return 1, time_passed  # Default return (e.g., full compliance)

    elif reasons == 'cyclist_reasons':
        distance_to_car = np.linalg.norm(
            [moving_obstacles[0].get()[0] - state.x, moving_obstacles[0].get()[1] - state.y])
        if distance_to_car < (distance_threshold + distance_buffer):
            time_passed += DT
            print('time passed cyclist is: {}'.format(time_passed))
            # print('distance to bicycle is: {}'.format(distance_to_obstacle))
            if time_passed >= time_threshold:
                reasons_cyclist = 1 / np.exp(0.2 * (time_passed - time_threshold))
                return reasons_cyclist, time_passed
        else:
            reasons_cyclist = 1
            return reasons_cyclist, time_passed

        # Ensure the function always returns something
        return 1, time_passed  # Default return (e.g., full compliance)

def evaluate_distance_to_centerline(av_position, av_width, centerline_location):
    # Calculate the distance from the left edge of the vehicle to the centerline
    left_edge_position = av_position - av_width / 2
    distance = left_edge_position - centerline_location

    if distance >= 0:
        reasons_policymaker = 1
    else:
        reasons_policymaker = np.exp(0.2 * (distance))

    return reasons_policymaker