import numpy as np
import cv2
import math

def calculate_shinai_endpoints(left_palm, right_palm):
    dx = right_palm[0] - left_palm[0]
    dy = right_palm[1] - left_palm[1]
    palm_distance = np.sqrt(dx**2 + dy**2)
    direction = np.array([dx, dy]) / (palm_distance + 1e-6)
    start = np.array(left_palm) - direction * palm_distance
    end = np.array(right_palm) + direction * palm_distance * 2
    return tuple(start.astype(int)), tuple(end.astype(int))

def draw_shinai(frame, start_point, end_point):
    cv2.line(frame, start_point, end_point, (0, 255, 0), thickness=4)

def draw_transparent_line(image, start_point, end_point, color, thickness=2, alpha=0.5):
    overlay = image.copy()
    cv2.line(overlay, start_point, end_point, color, thickness)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_angle_text(image, text, position, color=(255,255,255)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def save_frame_with_overlay(frame, shinai_start, shinai_end, left_palm, right_palm,
                            body_angle, shinai_angle, save_path,
                            draw_vertical_line=False, draw_arc=False):
    draw_transparent_line(frame, tuple(map(int, shinai_start)), tuple(map(int, shinai_end)), (0, 255, 0), thickness=4, alpha=0.6)
    horizontal_ref_end = (int(shinai_start[0] + 100), int(shinai_start[1]))
    draw_transparent_line(frame, tuple(map(int, shinai_start)), horizontal_ref_end, (255, 255, 255), thickness=2, alpha=0.6)

    draw_angle_text(frame, f"Shinai Angle: {shinai_angle:.2f} deg", (20, 40))
    draw_angle_text(frame, f"Body Angle: {body_angle:.2f} deg", (20, 80))

    if draw_vertical_line:
        left_palm_int = (int(left_palm[0]), int(left_palm[1]))
        vertical_line_end = (left_palm_int[0], left_palm_int[1] + 100)
        draw_transparent_line(frame, left_palm_int, vertical_line_end, (0, 0, 255), thickness=2, alpha=0.6)

    if draw_arc:
        center = (int(left_palm[0]), int(left_palm[1]))
        radius = 80
        start_angle = 0
        end_angle = shinai_angle
        cv2.ellipse(frame, center, (radius, radius), 90, start_angle, end_angle, (255, 0, 0), 2)

    cv2.imwrite(save_path, frame)
    return frame

def fit_principal_line(mask_path):
    """
    Fits a principal line to the shinai segmentation mask and returns the line parameters.

    Parameters:
        mask_path (str): Path to the mask image.

    Returns:
        (vx, vy, x, y) from cv2.fitLine or None if no valid line is found.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, thresholded = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
    vx, vy, x, y = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
    return vx, vy, x, y

def compute_angle_from_line(vx, vy):
    """
    Compute the angle relative to the vertical axis from the line direction (vx, vy).
    Parameters:
        vx, vy (float): Directional vectors of the line.
    Returns:
        float: Angle in degrees relative to the vertical axis (0° is vertical).
    """
    length = math.sqrt(vx * vx + vy * vy)
    if length < 1e-6:
        return 0.0
    vy_norm = vy / length
    return math.degrees(math.acos(vy_norm))



def line_points(x, y, vx, vy, length=200):
    """
    Computes two points along the line defined by (vx, vy) direction starting from (x, y).

    Parameters:
        x, y (float): A point on the line.
        vx, vy (float): Directional vector of the line.
        length (int): Length of the line segment to compute.

    Returns:
        tuple: Two points as ((x1, y1), (x2, y2)).
    """
    x1 = int(x - length * vx)
    y1 = int(y - length * vy)
    x2 = int(x + length * vx)
    y2 = int(y + length * vy)
    return (x1, y1), (x2, y2)
