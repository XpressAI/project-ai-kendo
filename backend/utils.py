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
    # If relative_angle > 0 => shinai above perpendicular baseline = small_cut
    # If relative_angle < 0 => shinai below perpendicular baseline = big_cut
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

def angle_from_vector(vx, vy, orientation):
    """
    Compute angle from vector vx, vy relative to +x axis (0 degrees), going counter-clockwise.
    Adjusts calculation based on body orientation.
    """
    length = math.sqrt(vx*vx + vy*vy)
    if length < 1e-6:
        return 0.0
    vx /= length
    vy /= length
    
    if orientation == "facing_left":
        # Flip the x-component for left-facing stance
        angle = math.degrees(math.atan2(-vy, -vx))
    else:
        angle = math.degrees(math.atan2(-vy, vx))
    return angle

def angle_difference(base_angle, target_angle):
    """
    Returns (target_angle - base_angle) normalized to (-180,180).
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
    """
    Draw text with an outline for better visibility. Replace ° with 'deg'.
    """
    sanitized_text = text.replace("°", " deg")
    x, y = position
    cv2.putText(image, sanitized_text, (x, y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(image, sanitized_text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_angle_arc(frame, center, baseline_angle, current_angle, radius=80, color=(255, 0, 0), thickness=2):
    """
    Draw an arc from the baseline line (0° after rotation) to the current line.
    The arc is drawn by rotating the ellipse by baseline_angle, then drawing
    from min(0, diff) to max(0, diff), where diff = current_angle - baseline_angle.
    """
    diff = angle_difference(baseline_angle, current_angle)
    start_angle = min(0, diff)
    end_angle = max(0, diff)
    cv2.ellipse(frame, center, (radius, radius), baseline_angle, start_angle, end_angle, color, thickness)

def save_frame_with_overlay_perp(frame, baseline_angle, current_angle, x, y, 
                                save_path, draw_arc=True, orientation="facing_right"):
    """
    Draws the overlay with orientation awareness
    """
    # Convert angles to vectors
    rad_b = math.radians(baseline_angle)
    rad_s = math.radians(current_angle)
    
    if orientation == "facing_left":
        # Adjust vector calculations for left-facing stance
        bvx = -math.cos(rad_b)
        bvy = -math.sin(rad_b)
        svx = -math.cos(rad_s)
        svy = -math.sin(rad_s)
    else:
        bvx = math.cos(rad_b)
        bvy = -math.sin(rad_b)
        svx = math.cos(rad_s)
        svy = -math.sin(rad_s)

    def line_points(xc, yc, vx, vy, length=200):
        x1 = int(xc - length * vx)
        y1 = int(yc - length * vy)
        x2 = int(xc + length * vx)
        y2 = int(yc + length * vy)
        return (x1, y1), (x2, y2)

    overlay = frame.copy()
    baseline_p1, baseline_p2 = line_points(x, y, bvx, bvy)
    shinai_p1, shinai_p2 = line_points(x, y, svx, svy)

    # Draw baseline (white)
    cv2.line(overlay, baseline_p1, baseline_p2, (255, 255, 255), thickness=2)
    # Draw shinai (green)
    cv2.line(overlay, shinai_p1, shinai_p2, (0, 255, 0), thickness=4)

    if draw_arc:
        center = (int(x), int(y))
        draw_angle_arc(overlay, center, baseline_angle, current_angle, 
                      radius=80, color=(0, 0, 255), thickness=2)

    rel_angle = angle_difference(baseline_angle, current_angle)
    draw_text_with_outline(overlay, f"Relative Angle: {rel_angle:.2f}°", (20, 40))
    draw_text_with_outline(overlay, f"Baseline Angle: {baseline_angle:.2f}°", (20, 80))
    draw_text_with_outline(overlay, f"Orientation: {orientation}", (20, 120))

    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.imwrite(save_path, frame)
    return frame