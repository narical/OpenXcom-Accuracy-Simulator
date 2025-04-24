import sys
import math
import random
import inspect # Needed to get function source code
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGridLayout,
    QSizePolicy, QLineEdit, QProgressDialog, QTabWidget, QComboBox,
    QTextEdit, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QFont

# --- Matplotlib Integration Imports ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# --- SciPy Import for Interpolation ---
from scipy.interpolate import CubicSpline

# --- Constants and Definitions ---
TILE_WIDTH = 16  # Voxels per tile width (X)
TILE_DEPTH = 16  # Voxels per tile depth (Y)
TILE_HEIGHT = 24 # Voxels per tile height (Z)
TARGET_RADIUS = 3.5 # Target cylinder radius (diameter 7)
TARGET_HEIGHT_MIN = 0.0 # Target cylinder bottom Z (float)
TARGET_HEIGHT_MAX = 19.0 # Target cylinder top Z (height 20) (float)
EPSILON = 1e-6 # Small number for float comparisons
EXTRAPOLATION_MARGIN = 1.0 # How much further to extrapolate beyond far edge
SWEEP_MAX_ACCURACY_PERCENT = 120.0
MIN_POINTS_FOR_SPLINE = 4 # Minimum points needed for a visually decent cubic spline
AVERAGE_HIT_RATE_MAX_ACCURACY = 100.0 # Max accuracy for average calculation

# --- Position Class ---
class Position:
    """Represents a 3D position with integer voxel coordinates."""
    def __init__(self, x, y, z):
        self.x = int(round(x))
        self.y = int(round(y))
        self.z = int(round(z))

    def __repr__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"

# --- Accuracy Functions ---
def applyAccuracyVanilla(origin: Position, target_initial: Position, accuracy: float) -> Position:
    """
    VANILLA accuracy algorithm (Previously applyAccuracyNew).
    Accuracy is expected as a fraction (e.g., 0.45 for 45%).
    """
    target = Position(target_initial.x, target_initial.y, target_initial.z)

    xDist = abs(origin.x - target.x)
    yDist = abs(origin.y - target.y)
    zDist = abs(origin.z - target.z)

    xyShift = 0
    zShift = 0
    if xDist / 2 <= yDist:
        xyShift = xDist / 4 + yDist
    else:
        xyShift = (xDist + yDist) / 2

    if xyShift <= zDist:
        zShift = xyShift / 2 + zDist
    else:
        zShift = xyShift + zDist / 2

    deviation_raw = random.randint(0, 100) - (float(accuracy) * 100.0)

    if deviation_raw >= 0:
        deviation_adjusted = deviation_raw + 50
    else:
        deviation_adjusted = deviation_raw + 10

    if abs(zShift) < EPSILON:
        zShift = EPSILON
    deviation = max(1, int(round(float(zShift) * deviation_adjusted / 200.0)))

    target.x += random.randint(0, deviation) - deviation // 2
    target.y += random.randint(0, deviation) - deviation // 2

    z_deviation_range = max(0, deviation // 2)
    z_deviation_offset = z_deviation_range // 2
    if z_deviation_range > 0:
        target.z += random.randint(0, z_deviation_range) - z_deviation_offset

    target.x = int(round(target.x))
    target.y = int(round(target.y))
    target.z = int(round(target.z))
    return target

def applyAccuracyNew(origin: Position, target_initial: Position, accuracy: float) -> Position:
	"""
	NEW/EXPERIMENTAL accuracy function.
	Applies an accuracy modifier to the target coordinates.
	Accuracy is expected as a fraction (e.g., 0.45 for 45%).
	"""
	target = Position(target_initial.x, target_initial.y, target_initial.z)

	xDist = abs(origin.x - target.x)
	yDist = abs(origin.y - target.y)
	zDist = abs(origin.z - target.z)

	xyShift = 0
	zShift = 0

	if xDist <= yDist:
		xyShift = xDist / 4 + yDist
	else:
		xyShift = yDist / 4 + xDist

	# 0.64583
	# 0.86111
	# xyShift = (xDist + yDist) * 1.2
	xyShift *= 0.839

	if xyShift <= zDist:
		zShift = xyShift / 2 + zDist
	else:
		zShift = xyShift + zDist / 2

	deviation_raw = random.randint(0, 100) - (float(accuracy) * 100.0)

	target_dist = math.sqrt(xDist * xDist + yDist * yDist)

	if deviation_raw >= 0:
		deviation_adjusted = deviation_raw + 50
	else:
		deviation_adjusted = deviation_raw + 10

	if abs(zShift) < EPSILON:
		zShift = EPSILON

	deviation = max(1, int(round(float(zShift) * deviation_adjusted / 200.0)))

	OVERALL_SPREAD_COEFF = 1.0
	INNER_SPREAD_COEFF = 0.85

	result_shifted = False

	for i in range(1, 10):

		dx = random.randint(0, deviation) - deviation // 2
		dy = random.randint(0, deviation) - deviation // 2

		deviate_dist = math.sqrt((target.x + dx - origin.x)**2 + (target.y + dy - origin.y)**2)

		if (result_shifted is True
			and deviate_dist > target_dist - 2
			and deviate_dist < target_dist + 2):
				break

		if result_shifted is False:
			
			radius_sq = dx*dx + dy*dy
			if (radius_sq <= (OVERALL_SPREAD_COEFF * deviation // 2) ** 2):
				break

			result_shifted = True
			deviation = int(deviation * INNER_SPREAD_COEFF)

	target.x += dx
	target.y += dy

	z_deviation_range = max(0, deviation // 2)
	z_deviation_offset = z_deviation_range // 2
	if z_deviation_range > 0:
		target.z += random.randint(0, z_deviation_range) - z_deviation_offset

	target.x = int(round(target.x))
	target.y = int(round(target.y))
	target.z = int(round(target.z))
	return target
	# --- End TEMPORARY ---

APPLY_ACCURACY_NEW_SOURCE = """
def applyAccuracyNew(origin: Position, target_initial: Position, accuracy: float) -> Position:

	target = Position(target_initial.x, target_initial.y, target_initial.z)

	xDist = abs(origin.x - target.x)
	yDist = abs(origin.y - target.y)
	zDist = abs(origin.z - target.z)

	xyShift = 0
	zShift = 0

	if xDist <= yDist:
		xyShift = xDist / 4 + yDist
	else:
		xyShift = yDist / 4 + xDist

	xyShift *= 0.839

	if xyShift <= zDist:
		zShift = xyShift / 2 + zDist
	else:
		zShift = xyShift + zDist / 2

	deviation_raw = random.randint(0, 100) - (float(accuracy) * 100.0)

	target_dist = math.sqrt(xDist * xDist + yDist * yDist)

	if deviation_raw >= 0:
		deviation_adjusted = deviation_raw + 50
	else:
		deviation_adjusted = deviation_raw + 10

	if abs(zShift) < EPSILON:
		zShift = EPSILON

	deviation = max(1, int(round(float(zShift) * deviation_adjusted / 200.0)))

	OVERALL_SPREAD_COEFF = 1.0
	INNER_SPREAD_COEFF = 0.85

	result_shifted = False

	for i in range(1, 10):

		dx = random.randint(0, deviation) - deviation // 2
		dy = random.randint(0, deviation) - deviation // 2

		deviate_dist = math.sqrt((target.x + dx - origin.x)**2 + (target.y + dy - origin.y)**2)

		if (result_shifted is True
			and deviate_dist > target_dist - 2
			and deviate_dist < target_dist + 2):
				break

		if result_shifted is False:
			
			radius_sq = dx*dx + dy*dy
			if (radius_sq <= (OVERALL_SPREAD_COEFF * deviation // 2) ** 2):
				break

			result_shifted = True
			deviation = int(deviation * INNER_SPREAD_COEFF)

	target.x += dx
	target.y += dy

	z_deviation_range = max(0, deviation // 2)
	z_deviation_offset = z_deviation_range // 2
	if z_deviation_range > 0:
		target.z += random.randint(0, z_deviation_range) - z_deviation_offset

	target.x = int(round(target.x))
	target.y = int(round(target.y))
	target.z = int(round(target.z))
	return target
"""
# --- Simulation Support Functions ---
def tile_to_voxel(tile_x, tile_y, voxel_x, voxel_y, voxel_z):
    """Converts tile coordinates and voxel within tile to global voxel coordinates."""
    global_x = tile_x * TILE_WIDTH + voxel_x
    global_y = tile_y * TILE_DEPTH + voxel_y
    global_z = voxel_z
    return Position(global_x, global_y, global_z)

# --- Hit Detection Functions ---
def check_angular_hit(origin_pos: Position, final_pos: Position, target_center_xy: tuple, radius: float) -> bool:
    """Checks if the shot trajectory angle falls within the target cylinder's angular span."""
    origin_xy = np.array([float(origin_pos.x), float(origin_pos.y)])
    final_xy = np.array([float(final_pos.x), float(final_pos.y)])
    target_c_xy = np.array(target_center_xy)

    vec_to_target = target_c_xy - origin_xy
    dist_to_target_sq = np.dot(vec_to_target, vec_to_target)

    # Handle edge cases: origin near or inside target
    if dist_to_target_sq < EPSILON: return True
    dist_to_target = math.sqrt(dist_to_target_sq)
    if dist_to_target < radius: return True
    if radius >= dist_to_target - EPSILON: return True
    # Check if dist_to_target is effectively zero before division
    if dist_to_target < EPSILON: return True

    # Calculate angular span
    # Ensure asin argument is valid
    asin_arg = min(1.0, max(-1.0, radius / dist_to_target))
    half_angle = math.asin(asin_arg)
    angle_to_target = math.atan2(vec_to_target[1], vec_to_target[0])

    # Calculate shot angle
    shot_vector = final_xy - origin_xy
    shot_vector_norm = np.linalg.norm(shot_vector)
    if shot_vector_norm < EPSILON:
        shot_angle = angle_to_target
    else:
        shot_angle = math.atan2(shot_vector[1], shot_vector[0])

    # Compare angles
    angle_difference = shot_angle - angle_to_target
    # Normalize angle difference to [-pi, pi]
    while angle_difference <= -math.pi - EPSILON: angle_difference += 2 * math.pi
    while angle_difference > math.pi + EPSILON: angle_difference -= 2 * math.pi

    return abs(angle_difference) <= half_angle + EPSILON

def check_z_overlap_during_intersection(origin_pos: Position, segment_end_pos: Position, target_center_xy: tuple, radius: float, z_min: float, z_max: float) -> bool:
    """Checks Z overlap during horizontal intersection."""
    origin = np.array([float(origin_pos.x), float(origin_pos.y), float(origin_pos.z)])
    end_point = np.array([float(segment_end_pos.x), float(segment_end_pos.y), float(segment_end_pos.z)])
    direction = end_point - origin

    # Handle zero direction vector
    if np.linalg.norm(direction) < EPSILON:
        return z_min - EPSILON <= origin[2] <= z_max + EPSILON

    # Setup for quadratic equation
    cyl_center = np.array([target_center_xy[0], target_center_xy[1]])
    radius_sq = radius * radius
    origin_xy = origin[:2]
    delta_xy = origin_xy - cyl_center
    direction_xy = direction[:2]

    a = np.dot(direction_xy, direction_xy)
    b = 2 * np.dot(delta_xy, direction_xy)
    c_prime = np.dot(delta_xy, delta_xy) - radius_sq

    # Handle vertical line case
    if abs(a) < EPSILON:
        if c_prime > EPSILON: return False
        seg_z1, seg_z2 = sorted([origin[2], end_point[2]])
        return max(z_min, seg_z1) <= min(z_max, seg_z2) + EPSILON

    # Solve quadratic equation for t
    discriminant = b*b - 4*a*c_prime
    if discriminant < -EPSILON: return False
    discriminant = max(0.0, discriminant)
    sqrt_discriminant = math.sqrt(discriminant)

    # Avoid division by zero if a is very small (already checked abs(a) < EPSILON)
    if abs(a) < EPSILON: return False # Should not happen here, but defensive
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    if t1 > t2: t1, t2 = t2, t1 # Ensure t1 <= t2

    # Find intersection interval of the segment [0, 1] with the cylinder interval [t1, t2]
    t_intersect_start = max(0.0, t1)
    t_intersect_end = min(1.0, t2)

    # Check if the intersection interval is valid
    if t_intersect_start > t_intersect_end + EPSILON:
        return False

    # Calculate Z values at the start and end of the horizontal intersection
    z_start = origin[2] + t_intersect_start * direction[2]
    z_end = origin[2] + t_intersect_end * direction[2]
    seg_z_min, seg_z_max = sorted([z_start, z_end])

    # Check if the segment's Z range during intersection overlaps with the cylinder's height
    overlap_min_z = max(z_min, seg_z_min)
    overlap_max_z = min(z_max, seg_z_max)

    return overlap_min_z <= overlap_max_z + EPSILON

# --- Simulation Runner ---
def run_simulation(shooter_tile_pos, target_tile_pos, accuracy_fraction: float, num_shots: int, accuracy_func):
    """
    Runs the shooting simulation for a single accuracy value using the specified accuracy function.
    Returns positions and hit statuses.
    """
    shooter_pos = tile_to_voxel(shooter_tile_pos[0], shooter_tile_pos[1], 8, 8, 20)
    origin_np = np.array([float(shooter_pos.x), float(shooter_pos.y), float(shooter_pos.z)])

    target_center_pos = tile_to_voxel(target_tile_pos[0], target_tile_pos[1], 8, 8, 10)
    target_center_xy = (float(target_center_pos.x), float(target_center_pos.y))
    target_c_xy_np = np.array(target_center_xy)

    # Pre-calculate distances for extrapolation check
    vec_to_target_xy = target_c_xy_np - origin_np[:2]
    dist_to_target_center_xy = np.linalg.norm(vec_to_target_xy)
    if dist_to_target_center_xy < EPSILON:
        dist_to_target_center_xy = EPSILON
    dist_to_target_far_xy = dist_to_target_center_xy + TARGET_RADIUS

    shot_results = []
    shot_statuses = []

    for _ in range(num_shots):
        # 1. Apply selected accuracy function
        final_pos = accuracy_func(shooter_pos, target_center_pos, accuracy_fraction)
        shot_results.append(final_pos)

        # 2. Perform hit check
        status = 'angular_miss'
        pos_for_z_check = final_pos # Assume we check original pos unless extrapolated

        if check_angular_hit(shooter_pos, final_pos, target_center_xy, TARGET_RADIUS):
            status = 'z_miss' # Passed angular, assume Z miss initially

            # 3. Check for extrapolation
            final_pos_np = np.array([float(final_pos.x), float(final_pos.y), float(final_pos.z)])
            shot_vector = final_pos_np - origin_np
            shot_vector_norm = np.linalg.norm(shot_vector)
            shot_vector_xy = shot_vector[:2]
            shot_dist_xy = np.linalg.norm(shot_vector_xy)

            # Extrapolate if shot falls short horizontally
            if shot_vector_norm >= EPSILON and shot_dist_xy < dist_to_target_center_xy - EPSILON:
                norm_direction = shot_vector / shot_vector_norm
                extrapolated_dist = dist_to_target_far_xy + EXTRAPOLATION_MARGIN
                extrapolated_end_point_np = origin_np + norm_direction * extrapolated_dist
                # Use extrapolated point for Z check
                pos_for_z_check = Position(extrapolated_end_point_np[0],
                                           extrapolated_end_point_np[1],
                                           extrapolated_end_point_np[2])

            # 4. Perform Z check (using original or extrapolated point)
            if check_z_overlap_during_intersection(shooter_pos, pos_for_z_check, target_center_xy,
                                                   TARGET_RADIUS, TARGET_HEIGHT_MIN, TARGET_HEIGHT_MAX):
                status = 'hit' # Hit confirmed

        shot_statuses.append(status)

    # Return necessary info
    return shooter_pos, target_center_pos, shot_results, shot_statuses

# --- Matplotlib Plotting Functions ---
def plot_scatter(ax, canvas, shooter_pos, target_center_pos, shot_results, shot_statuses,
                 angles, point_min_angle, point_max_angle, min_angle_deg, max_angle_deg,
                 trajectory_percentage, show_legend): # Added show_legend flag
    """Plots the scatter simulation results on a given Matplotlib Axes object."""
    ax.clear()
    num_shots = len(shot_results)

    # Separate points by status
    angular_miss_x, angular_miss_y = [], []
    z_miss_x, z_miss_y = [], []
    hit_x, hit_y = [], []
    for pos, status in zip(shot_results, shot_statuses):
        if status == 'angular_miss':
            angular_miss_x.append(pos.x)
            angular_miss_y.append(pos.y)
        elif status == 'z_miss':
            z_miss_x.append(pos.x)
            z_miss_y.append(pos.y)
        elif status == 'hit':
            hit_x.append(pos.x)
            hit_y.append(pos.y)

    origin_x, origin_y = shooter_pos.x, shooter_pos.y

    # Draw Trajectory Lines
    num_lines = int(num_shots * trajectory_percentage / 100.0)
    line_alpha = 0.15
    line_width = 0.5
    for i in range(min(num_lines, num_shots)):
        pos = shot_results[i]
        status = shot_statuses[i]
        color = 'gray'
        if status == 'z_miss':
            color = 'orange'
        elif status == 'hit':
            color = 'green'
        ax.plot([origin_x, pos.x], [origin_y, pos.y], color=color,
                alpha=line_alpha, linewidth=line_width, zorder=2)

    # Plot Points
    if angular_miss_x:
        ax.scatter(angular_miss_x, angular_miss_y, color='gray', s=5, alpha=0.4,
                   label='Miss (Angle)', zorder=3)
    if z_miss_x:
        ax.scatter(z_miss_x, z_miss_y, color='orange', s=7, alpha=0.6,
                   label='Miss (Z/Extrapolated)', zorder=3)
    if hit_x:
        ax.scatter(hit_x, hit_y, color='green', s=10, alpha=0.7,
                   label='Hit (Extrapolated)', zorder=3)

    # Plot Shooter and Target
    ax.scatter(shooter_pos.x, shooter_pos.y, marker='^', color='red', s=100,
               label='Shooter', zorder=5)
    target_circle = plt.Circle((target_center_pos.x, target_center_pos.y), TARGET_RADIUS,
                               color='blue', fill=False, linewidth=2,
                               label='Target (cylinder D=7)', zorder=4)
    ax.add_patch(target_circle)
    ax.scatter(target_center_pos.x, target_center_pos.y, marker='+', color='blue',
               s=50, zorder=4)

    # Draw max deviation lines
    if angles and point_min_angle and point_max_angle:
        ax.plot([origin_x, point_min_angle[0]], [origin_y, point_min_angle[1]], 'r--',
                alpha=0.7, label=f'Max. deviation left ({min_angle_deg:.1f}°)', zorder=3)
        ax.plot([origin_x, point_max_angle[0]], [origin_y, point_max_angle[1]], 'g--',
                alpha=0.7, label=f'Max. deviation right ({max_angle_deg:.1f}°)', zorder=3)

    # Axis and Grid Setup
    all_shot_x = [pos.x for pos in shot_results]
    all_shot_y = [pos.y for pos in shot_results]
    all_x = [shooter_pos.x, target_center_pos.x] + all_shot_x
    all_y = [shooter_pos.y, target_center_pos.y] + all_shot_y
    if not all_x: all_x = [0]
    if not all_y: all_y = [0]
    if len(set(all_x)) <= 1: all_x.extend([all_x[0] + 1, all_x[0] - 1])
    if len(set(all_y)) <= 1: all_y.extend([all_y[0] + 1, all_y[0] - 1])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    margin = 2 * TILE_WIDTH
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    x_ticks = np.arange(math.floor((min_x - margin) / TILE_WIDTH) * TILE_WIDTH,
                        math.ceil((max_x + margin) / TILE_WIDTH) * TILE_WIDTH, TILE_WIDTH)
    y_ticks = np.arange(math.floor((min_y - margin) / TILE_DEPTH) * TILE_DEPTH,
                        math.ceil((max_y + margin) / TILE_DEPTH) * TILE_DEPTH, TILE_DEPTH)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f'{int(round(tick / TILE_WIDTH))}' for tick in x_ticks])
    ax.set_yticklabels([f'{int(round(tick / TILE_DEPTH))}' for tick in y_ticks])

    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Tile X")
    ax.set_ylabel("Tile Y")
    ax.set_aspect('equal', adjustable='box')

    # Legend and Layout
    handles, labels = ax.get_legend_handles_labels()
    # *** Conditionally display legend ***
    if show_legend and handles:
        ax.legend(loc='best')
    try:
        ax.figure.tight_layout()
    except ValueError:
        print("Warning: tight_layout() failed.")
    canvas.draw()

def plot_sweep_results(ax, canvas, accuracies_percent,
                       hit_rates_vanilla, dispersion_angles_vanilla,
                       hit_rates_new, dispersion_angles_new):
    """Plots the comparison sweep results with interpolation."""
    # Ensure axes are cleared properly, handle potential lingering twin axes
    fig = ax.figure
    # Remove previous twin axes explicitly before clearing main axes
    for other_ax in fig.get_axes():
        # Check if it's a twin axis sharing the same figure and potentially x-axis
        # This check might need refinement depending on how twin axes are stored/identified
        if other_ax != ax and hasattr(other_ax, 'get_shared_x_axes') and ax in other_ax.get_shared_x_axes().get_siblings(other_ax):
             other_ax.remove()
    ax.clear()

    if not accuracies_percent or len(accuracies_percent) < 2:
        ax.text(0.5, 0.5, "Not enough sweep data to plot.", ha='center', va='center')
        canvas.draw(); return

    # Create smooth X values for interpolation
    min_acc = min(accuracies_percent); max_acc = max(accuracies_percent)
    if max_acc <= min_acc:
        x_smooth = np.array([min_acc])
    else:
        x_smooth = np.linspace(min_acc, max_acc, 300)

    # Interpolation
    can_interpolate = len(accuracies_percent) >= MIN_POINTS_FOR_SPLINE
    hr_v = np.array(hit_rates_vanilla); da_v = np.array(dispersion_angles_vanilla)
    hr_n = np.array(hit_rates_new); da_n = np.array(dispersion_angles_new)
    hr_v_smooth, da_v_smooth = hr_v, da_v; hr_n_smooth, da_n_smooth = hr_n, da_n

    if can_interpolate:
        try:
            spline_hr_v = CubicSpline(accuracies_percent, hr_v); spline_da_v = CubicSpline(accuracies_percent, da_v)
            spline_hr_n = CubicSpline(accuracies_percent, hr_n); spline_da_n = CubicSpline(accuracies_percent, da_n)
            hr_v_smooth = np.maximum(spline_hr_v(x_smooth), 0); da_v_smooth = np.maximum(spline_da_v(x_smooth), 0)
            hr_n_smooth = np.maximum(spline_hr_n(x_smooth), 0); da_n_smooth = np.maximum(spline_da_n(x_smooth), 0)
        except ValueError as e:
            print(f"Warning: Spline interpolation failed ({e}). Plotting raw data.")
            can_interpolate = False; x_smooth = np.array(accuracies_percent)

    # Plotting
    ax2 = ax.twinx() # Create twin axis AFTER clearing ax

    color_hr_v = 'tab:green'; color_da_v = 'tab:blue'
    color_hr_n = 'red'; color_da_n = 'orange'

    ax.set_xlabel('Input Accuracy (%)')
    ax.set_ylabel('Simulated Hit Rate (%)', color=color_hr_v)
    ax2.set_ylabel('Dispersion Angle (°)', color=color_da_v)

    # Plot smooth lines
    line1, = ax.plot(x_smooth, hr_v_smooth, color=color_hr_v, linestyle='-', label='Hit Rate (Vanilla)')
    line2, = ax2.plot(x_smooth, da_v_smooth, color=color_da_v, linestyle='-', label='Dispersion (Vanilla)')
    line3, = ax.plot(x_smooth, hr_n_smooth, color=color_hr_n, linestyle='--', label='Hit Rate (New)')
    line4, = ax2.plot(x_smooth, da_n_smooth, color=color_da_n, linestyle='--', label='Dispersion (New)')

    # Plot original points
    ax.scatter(accuracies_percent, hr_v, color=color_hr_v, marker='.', s=30, zorder=5)
    ax2.scatter(accuracies_percent, da_v, color=color_da_v, marker='.', s=30, zorder=5)
    ax.scatter(accuracies_percent, hr_n, color=color_hr_n, marker='x', s=30, zorder=5)
    ax2.scatter(accuracies_percent, da_n, color=color_da_n, marker='x', s=30, zorder=5)

    # Axis Limits and Ticks
    ax.tick_params(axis='y', labelcolor=color_hr_v)
    ax2.tick_params(axis='y', labelcolor=color_da_v)
    ax.set_ylim(0, max(105, SWEEP_MAX_ACCURACY_PERCENT))
    ax2.set_ylim(0, 60) # Set fixed limit as requested previously
    ax.set_xlim(0, SWEEP_MAX_ACCURACY_PERCENT)

    # Grid
    ax.grid(True, axis='x', linestyle=':')
    ax.grid(True, axis='y', linestyle=':', color=color_hr_v, alpha=0.6)
    ax2.grid(True, axis='y', linestyle=':', color=color_da_v, alpha=0.6)

    # Legend
    lines = [line1, line3, line2, line4]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax.set_title('Comparison: Simulation Results vs. Input Accuracy (Interpolated)')
    try:
        # Use rect to adjust layout and prevent legend overlap
        ax.figure.tight_layout(rect=[0, 0.05, 1, 1])
    except ValueError:
        print("Warning: tight_layout() failed.")
    canvas.draw()


# --- PyQt Main Window Class ---
class AccuracySimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XCOM Accuracy Simulator (PyQt)")
        self.setGeometry(100, 100, 1100, 800)

        # Store the dynamically loaded function
        self.current_applyAccuracyNew = applyAccuracyNew

        self._create_widgets()
        self._setup_layout()
        self._connect_signals()

        # Update scatter plot initially
        self.update_scatter_plot()
        # Initialize sweep plot axes
        self.sweepAx.clear()
        self.sweepAx.set_xlabel('Input Accuracy (%)'); self.sweepAx.set_ylabel('Value')
        self.sweepAx.set_title('Run Simulation Sweep to generate plot'); self.sweepAx.grid(True)
        self.sweepCanvas.draw()

    def _create_widgets(self):
        # --- Input Widgets ---
        self.shooterXSpin = QSpinBox(); self.shooterXSpin.setRange(-50, 50); self.shooterXSpin.setValue(5)
        self.shooterYSpin = QSpinBox(); self.shooterYSpin.setRange(-50, 50); self.shooterYSpin.setValue(0)
        self.targetXSpin = QSpinBox(); self.targetXSpin.setRange(-50, 50); self.targetXSpin.setValue(5)
        self.targetYSpin = QSpinBox(); self.targetYSpin.setRange(-50, 50); self.targetYSpin.setValue(10)
        self.accuracySpin = QDoubleSpinBox(); self.accuracySpin.setRange(0.0, 300.0); self.accuracySpin.setSingleStep(5.0); self.accuracySpin.setValue(45.0); self.accuracySpin.setDecimals(1); self.accuracySpin.setSuffix("%")
        self.numShotsSpin = QSpinBox(); self.numShotsSpin.setRange(10, 10000); self.numShotsSpin.setValue(1000); self.numShotsSpin.setSingleStep(100)
        self.trajectoryPercentSpin = QSpinBox(); self.trajectoryPercentSpin.setRange(0, 100); self.trajectoryPercentSpin.setValue(10); self.trajectoryPercentSpin.setSuffix("%")
        self.measurementStepSpin = QDoubleSpinBox(); self.measurementStepSpin.setRange(0.1, 20.0); self.measurementStepSpin.setValue(2.0); self.measurementStepSpin.setDecimals(1); self.measurementStepSpin.setSuffix("%")
        self.funcSelectorCombo = QComboBox(); self.funcSelectorCombo.addItems(["Vanilla", "New"])
        self.showLegendCheck = QCheckBox("Show Legend"); self.showLegendCheck.setChecked(True)

        # --- Button ---
        self.runSweepButton = QPushButton("Run Simulation Sweep")

        # --- Results Display Widgets (Single Run) ---
        self.distResultEdit = QLineEdit(); self.inputAccResultEdit = QLineEdit()
        self.simAccResultEdit = QLineEdit(); self.dispAngleResultEdit = QLineEdit()
        self.samplesResultEdit = QLineEdit(); self.hitsResultEdit = QLineEdit()
        for edit in [self.distResultEdit, self.inputAccResultEdit, self.simAccResultEdit,
                     self.dispAngleResultEdit, self.samplesResultEdit, self.hitsResultEdit]:
            edit.setReadOnly(True); edit.setText("-")

        # --- Results Display Widgets (Sweep Averages) --- NEW
        self.avgHitRateVanillaEdit = QLineEdit()
        self.avgHitRateNewEdit = QLineEdit()
        self.avgHitRateDiffEdit = QLineEdit()
        for edit in [self.avgHitRateVanillaEdit, self.avgHitRateNewEdit, self.avgHitRateDiffEdit]:
            edit.setReadOnly(True); edit.setText("-")

        # --- Tab Widget and Canvases ---
        self.plotTabs = QTabWidget()
        # Scatter Plot Tab
        self.scatterTab = QWidget()
        self.scatterFig = Figure(figsize=(5, 5), dpi=100)
        self.scatterCanvas = FigureCanvas(self.scatterFig)
        self.scatterCanvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scatterCanvas.updateGeometry()
        self.scatterAx = self.scatterFig.add_subplot(111)
        scatterLayout = QVBoxLayout(self.scatterTab); scatterLayout.setContentsMargins(0,0,0,0)
        scatterLayout.addWidget(self.scatterCanvas)
        self.plotTabs.addTab(self.scatterTab, "Scatter Plot")

        # Sweep Results Tab
        self.sweepTab = QWidget()
        self.sweepFig = Figure(figsize=(5, 5), dpi=100)
        self.sweepCanvas = FigureCanvas(self.sweepFig)
        self.sweepCanvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sweepCanvas.updateGeometry()
        self.sweepAx = self.sweepFig.add_subplot(111)
        sweepLayout = QVBoxLayout(self.sweepTab); sweepLayout.setContentsMargins(0,0,0,0)
        sweepLayout.addWidget(self.sweepCanvas)
        self.plotTabs.addTab(self.sweepTab, "Sweep Results")

        # --- Function Editor Tab ---
        self.editorTab = QWidget()
        editorLayout = QVBoxLayout(self.editorTab)
        warningLabel = QLabel(
            "<b>Warning:</b> Executing code from this editor can be insecure.<br>"
            "Use with caution. Ensure the function signature is correct:<br>"
            "<code>def applyAccuracyNew(origin: Position, target_initial: Position, accuracy: float) -> Position:</code>"
        )
        warningLabel.setStyleSheet("color: red; background-color: #ffeeee; border: 1px solid red; padding: 5px;")
        warningLabel.setWordWrap(True)
        editorLayout.addWidget(warningLabel)
        self.funcEditor = QTextEdit()
        self.funcEditor.setFont(QFont("Courier New", 10))
        self.funcEditor.setAcceptRichText(False)
        try:
            initial_source = APPLY_ACCURACY_NEW_SOURCE
            self.funcEditor.setText(initial_source)
        except (IOError, TypeError, IndexError):
            self.funcEditor.setPlainText(
                "# Could not load initial source code.\n"
                "def applyAccuracyNew(origin: Position, target_initial: Position, accuracy: float) -> Position:\n"
                "    # TODO: Implement logic\n"
                "    # Remember to return a Position object\n"
                "    target = Position(target_initial.x, target_initial.y, target_initial.z)\n"
                "    # Add your code here...\n"
                "    return target"
            )
        editorLayout.addWidget(self.funcEditor)
        self.applyFuncButton = QPushButton("Apply Function Code")
        editorLayout.addWidget(self.applyFuncButton)
        self.plotTabs.addTab(self.editorTab, "Edit Function")

    def _setup_layout(self):
        # --- Input and Results Layout (GridLayout) ---
        inputGroup = QWidget()
        inputLayout = QGridLayout(inputGroup)
        current_row = 0
        # Input fields
        inputLayout.addWidget(QLabel("Shooter Tile X:"), current_row, 0); inputLayout.addWidget(self.shooterXSpin, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Shooter Tile Y:"), current_row, 0); inputLayout.addWidget(self.shooterYSpin, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Target Tile X:"), current_row, 0); inputLayout.addWidget(self.targetXSpin, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Target Tile Y:"), current_row, 0); inputLayout.addWidget(self.targetYSpin, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Accuracy (%):"), current_row, 0); inputLayout.addWidget(self.accuracySpin, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Number of Shots:"), current_row, 0); inputLayout.addWidget(self.numShotsSpin, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Show Trajectories (%):"), current_row, 0); inputLayout.addWidget(self.trajectoryPercentSpin, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Function (Scatter):"), current_row, 0); inputLayout.addWidget(self.funcSelectorCombo, current_row, 1); current_row += 1
        inputLayout.addWidget(self.showLegendCheck, current_row, 0, 1, 2); current_row += 1
        inputLayout.addWidget(QLabel("Measurement Step (%):"), current_row, 0); inputLayout.addWidget(self.measurementStepSpin, current_row, 1); current_row += 1
        inputLayout.addWidget(self.runSweepButton, current_row, 0, 1, 2); current_row += 1

        # Results fields (Single Run - Reflect scatter plot results)
        inputLayout.addWidget(QLabel("--- Single Run Results ---"), current_row, 0, 1, 2); current_row += 1
        inputLayout.addWidget(QLabel("Distance (tiles):"), current_row, 0); inputLayout.addWidget(self.distResultEdit, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Input Accuracy:"), current_row, 0); inputLayout.addWidget(self.inputAccResultEdit, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Simulated Accuracy:"), current_row, 0); inputLayout.addWidget(self.simAccResultEdit, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Dispersion Angle:"), current_row, 0); inputLayout.addWidget(self.dispAngleResultEdit, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Total Samples:"), current_row, 0); inputLayout.addWidget(self.samplesResultEdit, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Hits:"), current_row, 0); inputLayout.addWidget(self.hitsResultEdit, current_row, 1); current_row += 1

        # Results fields (Sweep Averages 0-100%) --- NEW ---
        inputLayout.addWidget(QLabel("--- Sweep Averages (0-100% Acc) ---"), current_row, 0, 1, 2); current_row += 1
        inputLayout.addWidget(QLabel("Avg Hit Rate Vanilla:"), current_row, 0); inputLayout.addWidget(self.avgHitRateVanillaEdit, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Avg Hit Rate New:"), current_row, 0); inputLayout.addWidget(self.avgHitRateNewEdit, current_row, 1); current_row += 1
        inputLayout.addWidget(QLabel("Difference (New vs Vanilla):"), current_row, 0); inputLayout.addWidget(self.avgHitRateDiffEdit, current_row, 1); current_row += 1

        inputLayout.setRowStretch(current_row, 1) # Push controls up
        inputGroup.setMaximumWidth(300)

        # --- Main Layout (Horizontal: Inputs | Plot Tabs) ---
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(inputGroup)
        mainLayout.addWidget(self.plotTabs, 1) # Add TabWidget with stretch factor

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    def _connect_signals(self):
        # Connect button click ONLY to the sweep function
        self.runSweepButton.clicked.connect(self.run_sweep_and_plot)
        # Connect valueChanged signals for automatic SCATTER plot update
        self.shooterXSpin.valueChanged.connect(self.update_scatter_plot)
        self.shooterYSpin.valueChanged.connect(self.update_scatter_plot)
        self.targetXSpin.valueChanged.connect(self.update_scatter_plot)
        self.targetYSpin.valueChanged.connect(self.update_scatter_plot)
        self.accuracySpin.valueChanged.connect(self.update_scatter_plot)
        self.numShotsSpin.valueChanged.connect(self.update_scatter_plot)
        self.trajectoryPercentSpin.valueChanged.connect(self.update_scatter_plot)
        self.funcSelectorCombo.currentIndexChanged.connect(self.update_scatter_plot)
        self.showLegendCheck.stateChanged.connect(self.update_scatter_plot)
        # Connect Apply button on editor tab
        self.applyFuncButton.clicked.connect(self.apply_edited_function)

    # --- Helper to calculate angles/spread ---
    def calculate_angles(self, shooter_pos, target_center_pos, shot_results):
        # Calculates angles/spread for a single simulation result set
        origin_x, origin_y = shooter_pos.x, shooter_pos.y
        if target_center_pos is None or shooter_pos is None or not shot_results:
            return [], None, None, 0.0, 0.0, 0.0
        target_dx = target_center_pos.x - origin_x
        target_dy = target_center_pos.y - origin_y
        target_angle = math.atan2(target_dy, target_dx)
        angles = []
        points_with_angles = []
        all_shot_x = [pos.x for pos in shot_results]
        all_shot_y = [pos.y for pos in shot_results]
        for x, y in zip(all_shot_x, all_shot_y):
            dx = x - origin_x
            dy = y - origin_y
            if abs(dx) < EPSILON and abs(dy) < EPSILON: continue
            angle = math.atan2(dy, dx)
            angle_diff = angle - target_angle
            while angle_diff <= -math.pi + EPSILON: angle_diff += 2 * math.pi
            while angle_diff > math.pi + EPSILON: angle_diff -= 2 * math.pi
            angles.append(angle_diff)
            points_with_angles.append(((x, y), angle_diff))

        point_min_angle = (origin_x, origin_y)
        point_max_angle = (origin_x, origin_y)
        min_angle_deg = 0.0
        max_angle_deg = 0.0
        total_spread_deg = 0.0
        if angles:
            min_angle = min(angles)
            max_angle = max(angles)
            points_min = [p[0] for p in points_with_angles if abs(p[1] - min_angle) < EPSILON]
            points_max = [p[0] for p in points_with_angles if abs(p[1] - max_angle) < EPSILON]
            if points_min: point_min_angle = points_min[0]
            if points_max: point_max_angle = points_max[0]
            min_angle_deg = math.degrees(min_angle)
            max_angle_deg = math.degrees(max_angle)
            total_spread_deg = max_angle_deg - min_angle_deg
        # Return total_spread_deg needed for sweep plot, others for scatter
        return angles, point_min_angle, point_max_angle, min_angle_deg, max_angle_deg, total_spread_deg

    # --- Slot for Applying Edited Function Code ---
    def apply_edited_function(self):
        """Attempts to execute the code in the editor to redefine applyAccuracyNew."""
        code_string = self.funcEditor.toPlainText()
        print("Attempting to apply edited function code...")

        # Define a limited global scope for exec
        safe_builtins = {
            'int': int, 'float': float, 'abs': abs, 'max': max, 'min': min,
            'round': round, 'print': print, 'range': range, 'len': len,
            'True': True, 'False': False, 'None': None,
            'list': list, 'dict': dict, 'tuple': tuple, 'str': str,
            'zip': zip, 'enumerate': enumerate, 'sorted': sorted,
            # Add other safe built-ins if needed by the user's function
        }
        limited_globals = {
            'math': math,
            'random': random,
            'Position': Position,
            'np': np,
            'EPSILON': EPSILON,
            '__builtins__': safe_builtins # Use the safe dictionary
        }
        execution_locals = {}

        try:
            # Execute the function definition in the controlled scope
            exec(code_string, limited_globals, execution_locals)

            # Check if the function was defined correctly
            new_func = execution_locals.get('applyAccuracyNew')
            if callable(new_func):
                self.current_applyAccuracyNew = new_func # Store the new function object
                print("Successfully applied new function code.")
                QMessageBox.information(self, "Success", "New function code applied successfully.")
                # Trigger scatter plot update if 'New' is selected
                if self.funcSelectorCombo.currentText() == "New":
                    self.update_scatter_plot()
            else:
                raise ValueError("Could not find a callable function named 'applyAccuracyNew'.")

        except Exception as e:
            print(f"Error applying function code: {e}")
            QMessageBox.critical(self, "Error", f"Error applying function code:\n{e}")

    # --- Slot for Updating SCATTER Plot ---
    def update_scatter_plot(self):
        # Get inputs relevant to scatter plot
        shooter_tile = (self.shooterXSpin.value(), self.shooterYSpin.value())
        target_tile = (self.targetXSpin.value(), self.targetYSpin.value())
        accuracy_percent = self.accuracySpin.value()
        accuracy_fraction = accuracy_percent / 100.0
        num_shots = self.numShotsSpin.value()
        trajectory_percentage = self.trajectoryPercentSpin.value()
        selected_func_name = self.funcSelectorCombo.currentText()
        show_legend = self.showLegendCheck.isChecked()

        # Determine which accuracy function to use
        accuracy_func = applyAccuracyVanilla if selected_func_name == "Vanilla" else self.current_applyAccuracyNew

        print(f"Updating scatter plot: Func={selected_func_name}, Acc={accuracy_fraction:.3f}, Shots={num_shots}, Traj%={trajectory_percentage}, Legend={show_legend}")

        # Run simulation
        try:
             if num_shots <= 0: raise ValueError("Number of shots must be positive.")
             shooter_v, target_v, results, statuses = run_simulation(
                 shooter_tile, target_tile, accuracy_fraction, num_shots=num_shots, accuracy_func=accuracy_func
             )
        except Exception as e:
             print(f"Error during simulation using {selected_func_name} function: {e}")
             self.scatterAx.clear(); self.scatterAx.text(0.5, 0.5, f"Sim Error ({selected_func_name}):\n{e}", ha='center', va='center', color='red'); self.scatterCanvas.draw()
             for edit in [self.distResultEdit, self.inputAccResultEdit, self.simAccResultEdit,
                          self.dispAngleResultEdit, self.samplesResultEdit, self.hitsResultEdit]: edit.setText("-") # Clear results
             return

        # Calculate results
        hits = statuses.count('hit')
        hit_percentage = (hits / num_shots) * 100 if num_shots > 0 else 0
        dist_tiles = math.sqrt((shooter_tile[0] - target_tile[0])**2 + (shooter_tile[1] - target_tile[1])**2)
        angles, point_min_angle, point_max_angle, min_angle_deg, max_angle_deg, total_spread_deg = self.calculate_angles(shooter_v, target_v, results)

        # Update result display fields
        self.distResultEdit.setText(f"{dist_tiles:.1f}")
        self.inputAccResultEdit.setText(f"{accuracy_percent:.1f}%")
        self.simAccResultEdit.setText(f"{hit_percentage:.1f}%")
        self.dispAngleResultEdit.setText(f"{total_spread_deg:.1f}°")
        self.samplesResultEdit.setText(f"{num_shots}")
        self.hitsResultEdit.setText(f"{hits}")

        # Plot results on the SCATTER canvas
        try:
            plot_scatter(
                self.scatterAx, self.scatterCanvas, shooter_v, target_v,
                results, statuses,
                angles, point_min_angle, point_max_angle, min_angle_deg, max_angle_deg,
                trajectory_percentage,
                show_legend
            )
        except Exception as e:
             print(f"Error during plotting scatter: {e}")
             self.scatterAx.clear(); self.scatterAx.text(0.5, 0.5, f"Plot Error:\n{e}", ha='center', va='center', color='red'); self.scatterCanvas.draw()
             return

    # --- Slot for Running Simulation SWEEP and Plotting ---
    def run_sweep_and_plot(self):
        # Get fixed parameters
        shooter_tile = (self.shooterXSpin.value(), self.shooterYSpin.value())
        target_tile = (self.targetXSpin.value(), self.targetYSpin.value())
        num_shots_per_step = self.numShotsSpin.value()
        accuracy_step_percent = self.measurementStepSpin.value()

        # Clear previous average results --- NEW ---
        for edit in [self.avgHitRateVanillaEdit, self.avgHitRateNewEdit, self.avgHitRateDiffEdit]:
            edit.setText("-")

        if accuracy_step_percent <= 0:
            print("Error: Measurement step must be positive.")
            QMessageBox.warning(self, "Input Error", "Measurement step must be greater than 0.")
            self.sweepAx.clear(); self.sweepAx.text(0.5, 0.5, "Measurement step must be > 0", ha='center', va='center', color='red'); self.sweepCanvas.draw()
            return

        print(f"Starting simulation sweep: Shooter@{shooter_tile}, Target@{target_tile}, Shots/Step={num_shots_per_step}, Step={accuracy_step_percent}%")

        # --- Prepare data lists ---
        sweep_accuracies_percent = []
        sweep_hit_rates_vanilla = []
        sweep_dispersion_angles_vanilla = []
        sweep_hit_rates_new = []
        sweep_dispersion_angles_new = []

        # --- Setup Progress Dialog ---
        max_accuracy = SWEEP_MAX_ACCURACY_PERCENT
        if accuracy_step_percent < EPSILON: accuracy_step_percent = EPSILON
        num_float_steps = (max_accuracy - 0.0) / accuracy_step_percent
        total_steps = int(round(num_float_steps)) + 1

        progress = QProgressDialog("Running accuracy sweep...", "Cancel", 0, total_steps, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal); progress.setWindowTitle("Simulation Progress")
        progress.setValue(0); progress.show(); QApplication.processEvents()

        # --- Accuracy Sweep Loop ---
        cancelled = False; current_step = 0
        dynamic_new_func = self.current_applyAccuracyNew # Use the potentially dynamic function

        for acc_percent in np.arange(0.0, max_accuracy + accuracy_step_percent / 2.0, accuracy_step_percent):
            if acc_percent > max_accuracy + EPSILON: break
            if progress.wasCanceled(): cancelled = True; break

            current_step += 1
            progress.setValue(min(current_step, total_steps))
            progress.setLabelText(f"Simulating Accuracy: {acc_percent:.1f}% ({current_step}/{total_steps})")
            QApplication.processEvents()

            accuracy_fraction = acc_percent / 100.0

            # --- Run simulation for BOTH functions ---
            try:
                if num_shots_per_step <= 0: raise ValueError("Number of shots must be positive.")
                # Vanilla Run
                shooter_v_v, target_v_v, results_v, statuses_v = run_simulation(
                    shooter_tile, target_tile, accuracy_fraction, num_shots=num_shots_per_step, accuracy_func=applyAccuracyVanilla
                )
                # New Run
                shooter_v_n, target_v_n, results_n, statuses_n = run_simulation(
                    shooter_tile, target_tile, accuracy_fraction, num_shots=num_shots_per_step, accuracy_func=dynamic_new_func
                )
            except Exception as e:
                print(f"Error during simulation step (Acc={acc_percent}%): {e}")
                QMessageBox.critical(self, "Simulation Error", f"Error during simulation step (Acc={acc_percent}%):\n{e}")
                cancelled = True; break # Treat simulation error as cancellation

            # Calculate results for Vanilla
            hits_v = statuses_v.count('hit')
            hit_rate_v = (hits_v / num_shots_per_step) * 100.0 if num_shots_per_step > 0 else 0.0
            _, _, _, _, _, dispersion_angle_v = self.calculate_angles(shooter_v_v, target_v_v, results_v)

            # Calculate results for New
            hits_n = statuses_n.count('hit')
            hit_rate_n = (hits_n / num_shots_per_step) * 100.0 if num_shots_per_step > 0 else 0.0
            _, _, _, _, _, dispersion_angle_n = self.calculate_angles(shooter_v_n, target_v_n, results_n)

            # Store results
            sweep_accuracies_percent.append(acc_percent)
            sweep_hit_rates_vanilla.append(hit_rate_v)
            sweep_dispersion_angles_vanilla.append(dispersion_angle_v)
            sweep_hit_rates_new.append(hit_rate_n)
            sweep_dispersion_angles_new.append(dispersion_angle_n)

        progress.setValue(total_steps); progress.close()

        # --- Plot results and calculate averages if not cancelled ---
        if not cancelled and sweep_accuracies_percent:
            print("Sweep finished. Plotting results and calculating averages...")
            try:
                plot_sweep_results(
                    self.sweepAx, self.sweepCanvas, # Plot on the SWEEP canvas/axes
                    sweep_accuracies_percent,
                    sweep_hit_rates_vanilla, sweep_dispersion_angles_vanilla,
                    sweep_hit_rates_new, sweep_dispersion_angles_new
                )
                self.plotTabs.setCurrentWidget(self.sweepTab) # Switch to the sweep results tab

                # --- Calculate and display averages (0-100%) --- NEW ---
                vanilla_rates_0_100 = []
                new_rates_0_100 = []
                for acc, hr_v, hr_n in zip(sweep_accuracies_percent, sweep_hit_rates_vanilla, sweep_hit_rates_new):
                    if 0.0 - EPSILON <= acc <= AVERAGE_HIT_RATE_MAX_ACCURACY + EPSILON:
                        vanilla_rates_0_100.append(hr_v)
                        new_rates_0_100.append(hr_n)

                avg_vanilla = np.mean(vanilla_rates_0_100) if vanilla_rates_0_100 else 0.0
                avg_new = np.mean(new_rates_0_100) if new_rates_0_100 else 0.0

                diff_percent_str = "N/A"
                if avg_vanilla > EPSILON: # Avoid division by zero
                    diff_percent = (avg_new - avg_vanilla) / (avg_vanilla / 100.0)
                    diff_percent_str = f"{diff_percent:+.1f}%" # Show sign
                elif abs(avg_new) > EPSILON:
                    diff_percent_str = "+Inf%" # Or some indicator of large positive change
                elif abs(avg_vanilla) < EPSILON and abs(avg_new) < EPSILON:
                     diff_percent_str = "0.0%" # Both are zero

                self.avgHitRateVanillaEdit.setText(f"{avg_vanilla:.1f}%")
                self.avgHitRateNewEdit.setText(f"{avg_new:.1f}%")
                self.avgHitRateDiffEdit.setText(diff_percent_str)
                print(f"Avg Hit Rate (0-100%): Vanilla={avg_vanilla:.1f}%, New={avg_new:.1f}%, Diff={diff_percent_str}")
                # --- End NEW ---

            except Exception as e:
                 print(f"Error during plotting/calculating sweep results: {e}")
                 QMessageBox.critical(self, "Plotting/Calculation Error", f"Error during plotting or calculating sweep results:\n{e}")
                 self.sweepAx.clear(); self.sweepAx.text(0.5, 0.5, f"Plot/Calc Error:\n{e}", ha='center', va='center', color='red'); self.sweepCanvas.draw()
        elif cancelled:
            print("Sweep cancelled by user or error.")
            # Ensure average fields are cleared if cancelled during simulation
            for edit in [self.avgHitRateVanillaEdit, self.avgHitRateNewEdit, self.avgHitRateDiffEdit]:
                edit.setText("-")
        else:
             print("No data generated for plotting.")
             self.sweepAx.clear(); self.sweepAx.text(0.5, 0.5, "No data to plot.", ha='center', va='center'); self.sweepCanvas.draw()
             # Ensure average fields are cleared if no data
             for edit in [self.avgHitRateVanillaEdit, self.avgHitRateNewEdit, self.avgHitRateDiffEdit]:
                 edit.setText("-")


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a modern style if available (optional)
    app.setStyle('Fusion')
    # Or use a dark theme (optional)
    # dark_palette = QPalette()
    # dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    # # ... set other colors ...
    # app.setPalette(dark_palette)

    window = AccuracySimulatorApp()
    window.show()
    sys.exit(app.exec())

