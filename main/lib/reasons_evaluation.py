import numpy as np

def evaluate_distance_to_centerline(av_position, av_width, centerline_location, constant):
    # Calculate the distance from the left edge of the vehicle to the centerline
    left_edge_position = av_position - av_width / 2
    distance = left_edge_position - centerline_location

    if distance >= 0:
        reasons_policymaker = 1
    else:
        reasons_policymaker = np.exp(constant * (distance ** -2))

    return reasons_policymaker