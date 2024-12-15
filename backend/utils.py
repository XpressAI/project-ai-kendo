import numpy as np
import cv2

def calculate_shinai_endpoints(left_palm, right_palm):
    """
    Calculate the endpoints of the shinai based on the two palm centers.
    Extend the line 2x the distance between the palm centers.
    """
    dx = right_palm[0] - left_palm[0]
    dy = right_palm[1] - left_palm[1]
    palm_distance = np.sqrt(dx**2 + dy**2)

    # Direction vector (normalized)
    direction = np.array([dx, dy]) / (palm_distance + 1e-6)

    # Extend the shinai line
    start = np.array(left_palm) - direction * palm_distance
    end = np.array(right_palm) + direction * palm_distance * 2

    return tuple(start.astype(int)), tuple(end.astype(int))


def draw_shinai(frame, start_point, end_point):
    """
    Draw the shinai on the frame as a line.
    """
    cv2.line(frame, start_point, end_point, (0, 255, 0), thickness=4)
