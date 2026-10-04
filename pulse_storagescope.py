#!/usr/bin/env python3
"""
Pulse extraction based on local minima detection and min-to-min pulse separation.
Each pulse is rescaled so its minimum is 0 and maximum is 254.
Trigger point is where the rising edge crosses 128 in the rescaled data.
Displays fixed 6-sample pre-trigger, 24-sample post-trigger window (left-shifted by 4).
Sample rate: 24 sps. BPM range: 50-180 → pulse length: 8-29 samples.
Viterbi maximum-likelihood waveform overlay with intensity-driven path finding.
"""

VERSION = "2.13"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy import signal
from scipy import ndimage
import sys


def load_pulse_data(csv_file):
    """Load pulse waveform data from CSV."""
    df = pd.read_csv(csv_file)
    return df['waveform_value'].values


def smooth_5point_mean(data):
    """Apply 5-point moving average: mean of self and 2 neighbors on each side."""
    smoothed = np.convolve(data, np.ones(5)/5, mode='same')
    return smoothed


def find_local_minima(data, order=1):
    """Find indices of local minima using scipy.signal."""
    minima_indices = signal.argrelextrema(data, np.less, order=order)[0]
    return minima_indices


def extract_pulses_minmax(waveform, window_before=10, window_after=20,
                          min_samples=8, max_samples=29):
    """
    Extract pulses between consecutive minima.
    Rescale each pulse so min=0, max=254.
    Find trigger point where rising edge crosses 128.

    Args:
        waveform: raw waveform data
        window_before: samples to capture before trigger (10)
        window_after: samples to capture after trigger (20)
        min_samples: minimum pulse length in samples (8 for 180 bpm at 24 sps)
        max_samples: maximum pulse length in samples (29 for 50 bpm at 24 sps)

    Returns:
        pulses: array of extracted pulses (each rescaled to 0-254 range)
        trigger_indices: trigger point within each extracted pulse
        pulse_info: list of dicts with metadata (original min/max, trigger position)
    """
    # Smooth the data
    smoothed = smooth_5point_mean(waveform)

    # Find local minima
    minima_indices = find_local_minima(smoothed, order=1)

    if len(minima_indices) < 2:
        print(f"Warning: found only {len(minima_indices)} local minima")
        return np.array([]), np.array([]), []

    pulses = []
    trigger_indices = []
    pulse_info = []
    discarded_count = 0

    # Extract pulses between consecutive minima
    for i in range(len(minima_indices) - 1):
        start_idx = minima_indices[i]
        end_idx = minima_indices[i + 1]
        pulse_length = end_idx - start_idx

        # Check if pulse length is in valid range
        if pulse_length < min_samples or pulse_length > max_samples:
            discarded_count += 1
            continue

        # Extract pulse from raw (unsmoothed) waveform
        pulse = waveform[start_idx:end_idx + 1]

        # Find min and max in this pulse
        pulse_min = np.min(pulse)
        pulse_max = np.max(pulse)

        # Discard flat pulses
        if pulse_min == pulse_max:
            discarded_count += 1
            continue

        # Rescale pulse: min → 0, max → 254
        pulse_rescaled = (pulse - pulse_min) / (pulse_max - pulse_min) * 254.0

        # Find trigger point: where rising edge crosses 128
        # The rising edge is the early part of the pulse (after the minimum)
        trigger_idx = None

        # Search for the first crossing of 128 on the rising edge
        for j in range(len(pulse_rescaled) - 1):
            if pulse_rescaled[j] <= 128 < pulse_rescaled[j + 1]:
                # Rising edge crossing detected
                trigger_idx = j
                break

        # If no rising edge crossing found, discard
        if trigger_idx is None:
            discarded_count += 1
            continue

        # Extract fixed window around trigger: window_before samples before, window_after after
        # The trigger_idx is within the pulse; we need to map it back to the original waveform
        trigger_in_waveform = start_idx + trigger_idx

        window_start = trigger_in_waveform - window_before
        window_end = trigger_in_waveform + window_after + 1

        # Check if window is valid
        if window_start < 0 or window_end > len(waveform):
            discarded_count += 1
            continue

        # Extract window from raw waveform
        window = waveform[window_start:window_end]

        # Rescale this window the same way (using the pulse's min/max)
        window_rescaled = (window - pulse_min) / (pulse_max - pulse_min) * 254.0

        # Trigger point is at window_before in the extracted window
        trigger_in_window = window_before

        pulses.append(window_rescaled)
        trigger_indices.append(trigger_in_window)

        pulse_info.append({
            'original_min': float(pulse_min),
            'original_max': float(pulse_max),
            'pulse_length': pulse_length,
            'trigger_idx': trigger_idx
        })

    print(f"Extracted {len(pulses)} valid pulses")
    print(f"Discarded {discarded_count} pulses (length out of range or no valid trigger)")

    return np.array(pulses), np.array(trigger_indices), pulse_info


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm.
    Returns a list of (x, y) integer coordinates that make up the line.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points


def draw_rasterized_oscilloscope(pulses, trigger_indices, value_min=0, value_max=254,
                                 x_scale=4, y_scale=2):
    """
    Draw pulses on a rasterized bitmap using Bresenham's line algorithm.

    Args:
        pulses: array of pulse waveforms (already rescaled to 0-254)
        trigger_indices: position of trigger point in each pulse
        value_min, value_max: value range for display
        x_scale, y_scale: scaling factors for bitmap resolution

    Returns:
        intensity_map: 2D numpy array with accumulated pixel intensities
    """
    n_samples = pulses.shape[1]

    # Create high-resolution bitmap with scaling
    h_pixels = 350
    bitmap_height = h_pixels * y_scale
    # Bitmap width: last line segment ends at pixel (n_samples-1)*x_scale
    bitmap_width = (n_samples - 1) * x_scale + 1
    bitmap = np.zeros((bitmap_height, bitmap_width), dtype=np.float32)

    # For each pulse, draw line segments connecting consecutive samples
    for pulse_idx, pulse in enumerate(pulses):
        trigger_idx = trigger_indices[pulse_idx]

        # Draw line segments between consecutive samples
        for sample_idx in range(len(pulse) - 1):
            val0 = pulse[sample_idx]
            val1 = pulse[sample_idx + 1]

            # Map values to pixel coordinates with vertical scaling
            y0 = int(bitmap_height - 1 - ((val0 - value_min) / (value_max - value_min)) * (bitmap_height - 1))
            y1 = int(bitmap_height - 1 - ((val1 - value_min) / (value_max - value_min)) * (bitmap_height - 1))

            # Clamp to valid range
            y0 = np.clip(y0, 0, bitmap_height - 1)
            y1 = np.clip(y1, 0, bitmap_height - 1)

            # Sample indices are x coordinates with horizontal scaling
            x0 = sample_idx * x_scale
            x1 = (sample_idx + 1) * x_scale

            # Draw line using Bresenham's algorithm
            line_points = bresenham_line(x0, y0, x1, y1)

            # Accumulate intensity at each pixel the line passes through
            for x, y in line_points:
                if 0 <= x < bitmap_width and 0 <= y < bitmap_height:
                    bitmap[y, x] += 1.0

    return bitmap


def find_trigger_anchor_fine(bitmap, x_scale, trigger_value=128, value_min=0, value_max=254):
    """
    Find the brightest pixel in the bitmap as the anchor point.
    This represents the natural peak of the waveform, providing a stable anchor.
    Works at full bitmap resolution (4x finer than sample indices for x_scale=4).

    Args:
        bitmap: rasterized intensity map
        x_scale: horizontal scaling factor
        trigger_value: (unused) kept for backward compatibility
        value_min, value_max: value range for display

    Returns:
        anchor_column: bitmap column index of brightest pixel
        anchor_value: the value at that brightest pixel
    """
    bitmap_height = bitmap.shape[0]

    def bitmap_row_to_value(row):
        """Convert bitmap row back to waveform value."""
        return value_min + (bitmap_height - 1 - row) / (bitmap_height - 1) * (value_max - value_min)

    # Find the BRIGHTEST pixel in the entire bitmap (the natural peak)
    max_intensity = np.max(bitmap)
    max_row, max_col = np.unravel_index(np.argmax(bitmap), bitmap.shape)

    anchor_column = max_col
    anchor_value = bitmap_row_to_value(max_row)

    return anchor_column, anchor_value


def find_viterbi_ml_waveform(bitmap, n_samples, x_scale, anchor_sample, anchor_value, max_jump,
                             value_min=0, value_max=254, smoothness_weight=0.02):
    """
    Find maximum-likelihood waveform using Viterbi algorithm with smoothness constraints.

    Args:
        bitmap: rasterized intensity map (bitmap_height × bitmap_width)
        n_samples: number of samples in the pulse
        x_scale: horizontal scaling factor
        anchor_sample: sample index at anchor point (typically 6 for trigger)
        anchor_value: value at anchor point (typically 128)
        max_jump: maximum allowed value change per sample (typically 150 to allow steep edges)
        value_min, value_max: value range for display
        smoothness_weight: penalty coefficient for jumps (0.02); lower = intensity-driven, higher = smoother

    Returns:
        path: dict {sample_idx: value} for the optimal waveform path
    """
    bitmap_height = bitmap.shape[0]

    # Helper: convert value to bitmap row
    def value_to_bitmap_row(value):
        clipped_value = np.clip(value, value_min, value_max)
        return int(bitmap_height - 1 - ((clipped_value - value_min) / (value_max - value_min)) * (bitmap_height - 1))

    # Helper: get normalized intensity at (sample, value) position
    # Normalized = relative brightness at this sample (0-1 range)
    # This makes the algorithm follow local peaks rather than absolute peaks
    def get_normalized_intensity(sample_idx, value):
        bitmap_col = sample_idx * x_scale
        row = value_to_bitmap_row(value)

        if bitmap_col >= bitmap.shape[1] or row < 0 or row >= bitmap_height:
            return 0.01

        # Find the maximum intensity at this sample position (across all y values)
        max_intensity_at_sample = np.max(bitmap[:, bitmap_col])

        if max_intensity_at_sample <= 0:
            return 0.01

        # Return this value's intensity relative to the max at this sample
        actual_intensity = bitmap[row, bitmap_col]
        return max(actual_intensity / max_intensity_at_sample, 0.01)

    # Initialize path at anchor point
    path = {}
    path[anchor_sample] = anchor_value

    # Forward pass: from anchor_sample + 1 to n_samples - 1
    for sample in range(anchor_sample + 1, n_samples):
        prev_value = path[sample - 1]

        best_cost = float('inf')
        best_value = prev_value

        # Search range: constrained by max_jump
        min_val = max(value_min, prev_value - max_jump)
        max_val = min(value_max, prev_value + max_jump)

        for value in range(min_val, max_val + 1):
            norm_intensity = get_normalized_intensity(sample, value)

            # Primary goal: follow high relative intensity at this sample
            # Normalized intensity is 0-1, so -log ranges from -log(0.01)=4.6 to -log(1)=0
            intensity_cost = -np.log(norm_intensity)

            # Secondary goal: prefer smaller jumps (linear penalty, much lighter)
            jump_size = abs(value - prev_value)
            smoothness_cost = smoothness_weight * jump_size

            total_cost = intensity_cost + smoothness_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_value = value

        path[sample] = best_value

    # Backward pass: from anchor_sample - 1 down to 0
    for sample in range(anchor_sample - 1, -1, -1):
        next_value = path[sample + 1]

        best_cost = float('inf')
        best_value = next_value

        min_val = max(value_min, next_value - max_jump)
        max_val = min(value_max, next_value + max_jump)

        for value in range(min_val, max_val + 1):
            norm_intensity = get_normalized_intensity(sample, value)

            # Primary goal: follow high relative intensity at this sample
            intensity_cost = -np.log(norm_intensity)

            # Secondary goal: prefer smaller jumps
            jump_size = abs(value - next_value)
            smoothness_cost = smoothness_weight * jump_size

            total_cost = intensity_cost + smoothness_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_value = value

        path[sample] = best_value

    return path


def extract_intensity_ridge(bitmap, value_min=0, value_max=254, use_centroid=True):
    """
    Extract the intensity ridge from bitmap: for each column, compute either:
    - Peak: the value with maximum intensity (brightest pixel)
    - Centroid: the intensity-weighted average value (center of mass)

    Args:
        bitmap: rasterized intensity map
        value_min, value_max: value range for display
        use_centroid: if True, use intensity-weighted centroid; if False, use peak

    Returns:
        ridge: dict {column_idx: value} of ridge value in each column
    """
    bitmap_height = bitmap.shape[0]
    bitmap_width = bitmap.shape[1]

    def bitmap_row_to_value(row):
        return value_min + (bitmap_height - 1 - row) / (bitmap_height - 1) * (value_max - value_min)

    ridge = {}
    for col in range(bitmap_width):
        col_data = bitmap[:, col]

        if use_centroid:
            # Compute intensity-weighted centroid
            value_at_row = np.array([bitmap_row_to_value(row) for row in range(bitmap_height)])
            total_intensity = np.sum(col_data)

            if total_intensity > 0:
                centroid_value = np.average(value_at_row, weights=col_data)
                ridge[col] = centroid_value
            else:
                ridge[col] = (value_min + value_max) / 2
        else:
            # Use peak (brightest pixel)
            max_row = np.argmax(col_data)
            ridge[col] = bitmap_row_to_value(max_row)

    return ridge


def find_viterbi_ml_waveform_fine(bitmap, x_scale, anchor_column, anchor_value, max_jump,
                                   value_min=0, value_max=254, smoothness_weight=0.15, acceleration_weight=0.05,
                                   ridge_weight=1.0, bitmap_orig=None):
    """
    Find maximum-likelihood waveform using Viterbi algorithm working at bitmap column resolution.
    Uses smoothness, acceleration penalties, and strong attraction to the intensity ridge.

    Args:
        bitmap: rasterized intensity map for cost computation (usually smoothed) (bitmap_height × bitmap_width)
        x_scale: horizontal scaling factor (for scaling max_jump per column)
        anchor_column: column index at anchor point
        anchor_value: value at anchor point
        max_jump: maximum allowed value change per sample; scaled to per-column
        value_min, value_max: value range for display
        smoothness_weight: penalty coefficient for first derivative (jumps)
        acceleration_weight: penalty coefficient for second derivative (acceleration)
        ridge_weight: penalty coefficient for deviation from intensity ridge (default 1.0)
        bitmap_orig: original unsmoothed bitmap for ridge extraction (if None, uses bitmap)

    Returns:
        path: dict {column_idx: value} for the optimal waveform path
    """
    bitmap_height = bitmap.shape[0]
    bitmap_width = bitmap.shape[1]

    # Use original bitmap for ridge extraction if provided, otherwise use the (possibly smoothed) bitmap for cost
    bitmap_for_ridge = bitmap_orig if bitmap_orig is not None else bitmap

    # Extract intensity ridge (brightest pixel at each column)
    # This follows the intensity peaks at each x position
    ridge_raw = extract_intensity_ridge(bitmap_for_ridge, value_min, value_max, use_centroid=False)

    # Smooth the ridge to remove spurious noise spikes that can trap the algorithm
    # Convert to array, smooth with Gaussian filter, then back to dict
    ridge_columns = sorted(ridge_raw.keys())
    ridge_values = np.array([ridge_raw[col] for col in ridge_columns])

    # Apply Gaussian smoothing (sigma=2.5) to suppress outlier spikes and smooth transitions
    ridge_values_smooth = ndimage.gaussian_filter(ridge_values.astype(np.float32), sigma=2.5)

    # In the downslope region (after the peak), suppress spurious high-value spikes by clipping
    # to an expected smooth decay curve. This prevents the algorithm from being attracted to
    # noise artifacts like 254-valued pixels on the descent.

    # Find the peak (maximum value) and its location
    peak_idx = np.argmax(ridge_values_smooth)
    peak_col = ridge_columns[peak_idx]
    peak_value = ridge_values_smooth[peak_idx]

    # For columns after the peak, clip to an aggressive exponential decay envelope
    # This strongly suppresses spurious high-value noise artifacts on the downslope
    end_col = ridge_columns[-1]

    for i, col in enumerate(ridge_columns):
        if col > peak_col:
            # Exponential decay: (1 - progress)^2.0 creates steep initial drop, then gentle tail
            # This matches typical pulse waveform shape: rapid fall after peak, then slow descent
            progress = (col - peak_col) / (end_col - peak_col)
            # Aggressive power law: value decays as peak * (1 - progress)^2.0
            expected_max = peak_value * max(0.0, (1 - progress) ** 2.0)

            # Clip to this expected maximum (suppress spikes that shoot above the decay curve)
            ridge_values_smooth[i] = min(ridge_values_smooth[i], expected_max)

    # Enforce strict monotonic decrease in downslope region: no local maxima allowed
    # This ensures the path cannot have any upward bulges after the peak
    for i in range(peak_idx + 1, len(ridge_columns)):
        if ridge_values_smooth[i] > ridge_values_smooth[i-1]:
            ridge_values_smooth[i] = ridge_values_smooth[i-1]

    ridge = {ridge_columns[i]: ridge_values_smooth[i] for i in range(len(ridge_columns))}

    # Scale max_jump to per-column basis (if max_jump=150 per sample, divide by x_scale)
    max_jump_per_column = max(1, int(max_jump / x_scale))

    def value_to_bitmap_row(value):
        clipped_value = np.clip(value, value_min, value_max)
        return int(bitmap_height - 1 - ((clipped_value - value_min) / (value_max - value_min)) * (bitmap_height - 1))

    def get_normalized_intensity(column, value):
        row = value_to_bitmap_row(value)

        if column >= bitmap_width or row < 0 or row >= bitmap_height:
            return 0.01

        max_intensity_at_column = np.max(bitmap[:, column])

        if max_intensity_at_column <= 0:
            return 0.01

        actual_intensity = bitmap[row, column]
        return max(actual_intensity / max_intensity_at_column, 0.01)

    # Initialize path at anchor point (convert anchor_value to int)
    path = {}
    anchor_value = int(round(anchor_value))
    path[anchor_column] = anchor_value

    # Forward pass: from anchor_column + 1 to bitmap_width - 1
    for column in range(anchor_column + 1, bitmap_width):
        prev_value = path[column - 1]
        prev_prev_value = path.get(column - 2, prev_value)  # Use prev_value if at start

        best_cost = float('inf')
        best_value = prev_value

        # Search range: constrained by max_jump_per_column
        min_val = max(value_min, prev_value - max_jump_per_column)
        max_val = min(value_max, prev_value + max_jump_per_column)

        ridge_value = ridge.get(column, prev_value)

        for value in range(min_val, max_val + 1):
            norm_intensity = get_normalized_intensity(column, value)

            intensity_cost = -np.log(norm_intensity)

            # First derivative penalty: prefer smaller jumps
            jump_size = abs(value - prev_value)
            smoothness_cost = smoothness_weight * jump_size

            # Second derivative penalty: prefer consistent slopes (penalize acceleration)
            prev_jump = abs(prev_value - prev_prev_value)
            acceleration = abs(jump_size - prev_jump)
            acceleration_cost = acceleration_weight * acceleration

            # Ridge penalty: strong attraction to intensity ridge (brightest pixel at this column)
            ridge_cost = ridge_weight * abs(value - ridge_value) / (value_max - value_min)

            total_cost = intensity_cost + smoothness_cost + acceleration_cost + ridge_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_value = value

        path[column] = best_value

    # Backward pass: from anchor_column - 1 down to 0
    for column in range(anchor_column - 1, -1, -1):
        next_value = path[column + 1]
        next_next_value = path.get(column + 2, next_value)  # Use next_value if at end

        best_cost = float('inf')
        best_value = next_value

        min_val = max(value_min, next_value - max_jump_per_column)
        max_val = min(value_max, next_value + max_jump_per_column)

        ridge_value = ridge.get(column, next_value)

        for value in range(min_val, max_val + 1):
            norm_intensity = get_normalized_intensity(column, value)

            intensity_cost = -np.log(norm_intensity)

            # First derivative penalty
            jump_size = abs(value - next_value)
            smoothness_cost = smoothness_weight * jump_size

            # Second derivative penalty
            next_jump = abs(next_value - next_next_value)
            acceleration = abs(jump_size - next_jump)
            acceleration_cost = acceleration_weight * acceleration

            # Ridge penalty
            ridge_cost = ridge_weight * abs(value - ridge_value) / (value_max - value_min)

            total_cost = intensity_cost + smoothness_cost + acceleration_cost + ridge_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_value = value

        path[column] = best_value

    return path


def write_ml_path_csv(ml_path, x_scale, output_file):
    """
    Write the ML path to a CSV file with sample time resolution.

    Args:
        ml_path: dict {column_idx: value} from Viterbi algorithm
        x_scale: horizontal scaling factor (to convert columns back to samples)
        output_file: path to output CSV file
    """
    # Convert bitmap columns to sample positions and sort
    columns = sorted(ml_path.keys())
    sample_positions = [c / x_scale for c in columns]
    values = [ml_path[c] for c in columns]

    # Write CSV with header
    with open(output_file, 'w') as f:
        f.write('time,value\n')
        for sample, value in zip(sample_positions, values):
            f.write(f'{sample:.2f},{value:.2f}\n')

    print(f"ML path written to {output_file}")


def plot_rasterized(pulses, trigger_indices, value_min=0, value_max=254,
                    x_scale=4, y_scale=2):
    """Display rasterized waveforms as intensity map."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw rasterized oscilloscope display
    bitmap = draw_rasterized_oscilloscope(pulses, trigger_indices, value_min, value_max,
                                          x_scale, y_scale)

    # Display bitmap with intensity mapping using power law scaling
    cmap = plt.cm.hot.copy()
    cmap.set_under('black')

    n_samples = pulses.shape[1]
    im = ax.imshow(bitmap[::-1, :], aspect='auto', cmap=cmap,
                   extent=[0, n_samples-1, value_min, value_max],
                   origin='lower', interpolation='nearest',
                   norm=PowerNorm(gamma=0.25, vmin=0.5, vmax=bitmap.max()))

    # Find anchor on the original bitmap (brightest pixel before smoothing)
    # This gives us the true peak of the data
    anchor_column, anchor_value = find_trigger_anchor_fine(bitmap, x_scale, trigger_value=128,
                                                           value_min=value_min, value_max=value_max)

    # Add trigger marker at the detected rising edge
    anchor_sample = anchor_column / x_scale

    # Debug: print anchor point info
    print(f"Anchor detected at column {anchor_column} (sample {anchor_sample:.2f}), value {anchor_value:.1f}")

    ax.axvline(x=anchor_sample, color='lime', linestyle='--', linewidth=1.5,
               label='Rising edge trigger', alpha=0.8)

    # Apply very gentle smoothing (3x3 equivalent) to de-emphasize local peaks
    # while preserving overall structure. This helps the algorithm follow smooth ridges.
    bitmap_smooth = ndimage.gaussian_filter(bitmap.astype(np.float32), sigma=0.3)

    # Compute Viterbi maximum-likelihood waveform at bitmap column resolution
    # Using the smoothed bitmap for intensity costs (de-emphasizes local noise peaks)
    # But extract ridge from original bitmap (true peaks), smoothed to remove noise
    # Tuning parameters:
    #   smoothness_weight: penalty for slope changes (0.15)
    #   acceleration_weight: penalty for curvature changes (0.05)
    #   ridge_weight: attraction to intensity ridge (50.0 = very strong, enforces smooth ridge-following)
    ml_path = find_viterbi_ml_waveform_fine(bitmap_smooth, x_scale,
                                            anchor_column=anchor_column, anchor_value=anchor_value,
                                            max_jump=150, value_min=value_min, value_max=value_max,
                                            smoothness_weight=0.15, acceleration_weight=0.05,
                                            ridge_weight=50.0, bitmap_orig=bitmap)

    # Extract ordered waveform values and convert columns back to sample space for plotting
    columns = sorted(ml_path.keys())
    sample_positions = np.array([c / x_scale for c in columns])
    ml_values = np.array([ml_path[c] for c in columns])

    ax.plot(sample_positions, ml_values, color='white', linewidth=2.0, alpha=0.9,
            label='Maximum likelihood waveform')

    ax.set_xlabel('Sample number', fontsize=11)
    ax.set_ylabel('Waveform value', fontsize=11)
    ax.set_title(f'Pulse Waveforms Rasterized Display ({len(pulses)} pulses)', fontsize=12)
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Accumulated intensity')

    plt.tight_layout()
    return fig, ml_path


def main():
    print(f"Pulse Oscilloscope Rasterized Display v{VERSION}")

    if len(sys.argv) < 2:
        print("Usage: python pulse_oscilloscope_minmax.py <csv_file> [output_file] [x_scale] [y_scale]")
        print("  Extracts pulses based on local minima detection")
        print("  Rescales each pulse: min→0, max→254")
        print("  Triggers on rising edge crossing 128 (window: 6 pre-trigger, 24 post-trigger)")
        print("  Displays rasterized waveforms with Viterbi maximum-likelihood overlay")
        print("  x_scale: horizontal scaling factor (default 4)")
        print("  y_scale: vertical scaling factor (default 2)")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    x_scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    y_scale = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    # Load raw waveform data
    waveform = load_pulse_data(csv_file)
    print(f"Loaded {len(waveform)} samples from {csv_file}")

    # Extract pulses based on local minima
    # Shifted left by 4 counts: window_before=6, window_after=24
    pulses, trigger_indices, pulse_info = extract_pulses_minmax(
        waveform, window_before=6, window_after=24,
        min_samples=8, max_samples=29
    )

    if len(pulses) == 0:
        print("No valid pulses found!")
        sys.exit(1)

    # Show statistics
    print(f"Pulse length range: {min(p['pulse_length'] for p in pulse_info)}-{max(p['pulse_length'] for p in pulse_info)} samples")
    print(f"Original value ranges (min-max) across all pulses:")
    original_mins = [p['original_min'] for p in pulse_info]
    original_maxes = [p['original_max'] for p in pulse_info]
    print(f"  Min values: {min(original_mins):.1f} to {max(original_mins):.1f}")
    print(f"  Max values: {min(original_maxes):.1f} to {max(original_maxes):.1f}")

    print(f"Horizontal resolution: {pulses.shape[1] * x_scale} pixels ({x_scale}x scaling) for {pulses.shape[1]} samples")
    print(f"Vertical resolution: {350 * y_scale} pixels ({y_scale}x scaling) for value range [0, 254]")

    # Create and display rasterized display
    fig, ml_path = plot_rasterized(pulses, trigger_indices, value_min=0, value_max=254,
                                   x_scale=x_scale, y_scale=y_scale)

    # Always generate CSV output from the input CSV filename
    csv_output = csv_file.rsplit('.', 1)[0] + '_fit.csv'
    write_ml_path_csv(ml_path, x_scale, csv_output)

    if output_file:
        fig.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
