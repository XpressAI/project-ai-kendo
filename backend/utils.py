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

def classify_cut(relative_angle):
    # If final angle > 0 => small cut (pointing upwards)
    # If final angle < 0 => big cut (pointing downwards)
    if relative_angle > 0:
        return "small_cut"
    else:
        return "big_cut"

def fit_principal_line(mask_path):
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

def angle_from_vector(vx, vy):
    """
    Compute angle from vector vx, vy relative to +x axis, counterclockwise.
    Range: (-180,180)
    """
    length = math.sqrt(vx*vx + vy*vy)
    if length < 1e-6:
        return 0.0
    vx /= length
    vy /= length
    # Flip vy to match a mathematical coordinate system (y up)
    angle = math.degrees(math.atan2(-vy, vx))
    return angle

def angle_difference(base_angle, target_angle):
    """
    Compute difference (target_angle - base_angle) normalized to (-180,180)
    """
    diff = target_angle - base_angle
    while diff > 180:
        diff -= 360
    while diff <= -180:
        diff += 360
    return diff

def draw_text_with_outline(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale=0.8, text_color=(255,255,255), thickness=2, 
                           outline_color=(0,0,0), outline_thickness=4):
    sanitized_text = text.replace("°", " deg")
    x, y = position
    cv2.putText(image, sanitized_text, (x, y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(image, sanitized_text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_angle_arc(frame, center, baseline_angle, current_angle, radius=80, color=(255, 0, 0), thickness=2):
    relative_angle = angle_difference(baseline_angle, current_angle) * -1
    # We compute difference as baseline->current, but cv2.ellipse requires startAngle=0 to endAngle=relative_angle
    # angle_difference gives (target-base), we want arc from baseline to current:
    # If relative_angle is positive, arc goes that direction; if negative, it goes the other way.
    # We'll just use relative_angle as computed from base to current.
    # Actually, angle_difference(base, current) = current - base, so if we want an arc from base line,
    # rotate ellipse by baseline_angle, start at 0°, end at that difference.
    diff = angle_difference(baseline_angle, current_angle)
    cv2.ellipse(frame, center, (radius, radius), baseline_angle, 0, diff, color, thickness)

def save_frame_with_overlay_perp(frame, vx, vy, x, y, baseline_angle, current_angle, save_path, draw_arc=True):
    """
    Draw from a perpendicular baseline line to shinai line:
     - The baseline line is defined by baseline_angle.
     - The shinai line is defined by current_angle.
    """
    # We'll construct line for baseline:
    # A vector from angle:
    rad = math.radians(baseline_angle)
    bvx = math.cos(rad)
    bvy = -math.sin(rad)  # because we flipped y in angle_from_vector

    # Construct line for shinai:
    rad_s = math.radians(current_angle)
    svx = math.cos(rad_s)
    svy = -math.sin(rad_s)

    def line_points(x, y, vx, vy, length=200):
        x1 = int(x - length * vx)
        y1 = int(y - length * vy)
        x2 = int(x + length * vx)
        y2 = int(y + length * vy)
        return (x1, y1), (x2, y2)

    # Baseline line
    baseline_p1, baseline_p2 = line_points(x, y, bvx, bvy)
    # Current shinai line
    shinai_p1, shinai_p2 = line_points(x, y, svx, svy)

    overlay = frame.copy()
    # Draw baseline line (white)
    cv2.line(overlay, baseline_p1, baseline_p2, (255, 255, 255), thickness=2)
    # Draw shinai line (green)
    cv2.line(overlay, shinai_p1, shinai_p2, (0, 255, 0), thickness=4)

    if draw_arc:
        center = (int(x), int(y))
        draw_angle_arc(overlay, center, baseline_angle, current_angle, radius=80, color=(0, 0, 255), thickness=2)

    rel_angle = angle_difference(baseline_angle, current_angle)
    draw_text_with_outline(overlay, f"Shinai Angle (rel): {rel_angle:.2f}°", (20, 40), text_color=(255,255,0))
    draw_text_with_outline(overlay, f"Baseline Angle: {baseline_angle:.2f}°", (20, 80), text_color=(255,255,0))

    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imwrite(save_path, frame)
    return frame
