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

def draw_transparent_line(image, start_point, end_point, color, thickness=2, alpha=0.5):
    """
    Draw a semi-transparent line on the image.
    """
    overlay = image.copy()
    cv2.line(overlay, start_point, end_point, color, thickness)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_angle_text(image, text, position, color=(255,255,255)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def save_frame_with_overlay(frame, shinai_start, shinai_end, left_palm, right_palm,
                            body_angle, shinai_angle, save_path,
                            draw_vertical_line=False, draw_arc=False):
    """
    Draw semi-transparent lines for shinai and show angles.
    Optionally draw a vertical line from the left_palm downward and an arc to visualize the angle.
    
    Parameters:
    - frame: The image frame to draw on.
    - shinai_start, shinai_end: Endpoints of the shinai line.
    - left_palm, right_palm: Coordinates of palms.
    - body_angle, shinai_angle: Computed angles.
    - save_path: Path to save the image.
    - draw_vertical_line (bool): If True, draw a vertical reference line from left palm downward.
    - draw_arc (bool): If True, draw an arc showing angle between vertical and the shinai line.
    """

    # Shinai line (green, semi-transparent)
    draw_transparent_line(frame, tuple(map(int, shinai_start)), tuple(map(int, shinai_end)), (0, 255, 0), thickness=4, alpha=0.6)

    # Draw horizontal reference line (optional, for legacy visualization)
    horizontal_ref_end = (int(shinai_start[0] + 100), int(shinai_start[1]))
    draw_transparent_line(frame, tuple(map(int, shinai_start)), horizontal_ref_end, (255, 255, 255), thickness=2, alpha=0.6)

    # Annotate angles
    draw_angle_text(frame, f"Shinai Angle: {shinai_angle:.2f} deg", (20, 40))
    draw_angle_text(frame, f"Body Angle: {body_angle:.2f} deg", (20, 80))

    if draw_vertical_line:
        # Ensure the left_palm and vertical_line_end points are integers
        left_palm_int = (int(left_palm[0]), int(left_palm[1]))
        vertical_line_end = (left_palm_int[0], left_palm_int[1] + 100)  # 100 pixels down
        draw_transparent_line(frame, left_palm_int, vertical_line_end, (0, 0, 255), thickness=2, alpha=0.6)

    if draw_arc:
        # Draw an arc to represent the angle between vertical and the shinai line.
        center = (int(left_palm[0]), int(left_palm[1]))
        radius = 80  # radius of the arc
        # Arc angles
        start_angle = 0
        end_angle = shinai_angle
        cv2.ellipse(frame, center, (radius, radius), 90, start_angle, end_angle, (255, 0, 0), 2)

    cv2.imwrite(save_path, frame)
    return frame
