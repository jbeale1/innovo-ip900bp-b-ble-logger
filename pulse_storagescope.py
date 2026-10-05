#!/usr/bin/env python3
"""
Pulse extraction based on local minima detection and min-to-min pulse separation.
Each pulse is rescaled so its minimum is 0 and maximum is 254.
Reference point: midpoint between pulse minimum and maximum (sub-sample precision).
Displays fixed 5-sample pre-minimum, 30-sample post-minimum window for analysis (36 total = 1.5 seconds).
Sample rate: 24 sps. BPM range: 50-180 → pulse length: 8-29 samples.
Median trace extraction: selects the actual pulse closest to per-sample median across all pulses.
Supports chunking by time duration for long recordings.
Time axis: midpoint between min/max = t=0 sec (pulse reference); extended to 1.5 seconds.
Display uses linear interpolation with 4x oversampling for smoother curves.
J.Beale 4-Oct-2026
"""

VERSION = "2.35"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy import signal
from scipy import ndimage
from scipy.interpolate import CubicSpline
import sys
import os
import re
from datetime import datetime, timedelta


def find_value_position(pulse, target_value):
    """
    Find the fractional sample position where a pulse reaches a target value.
    Uses linear interpolation between samples.

    Args:
        pulse: array of pulse values
        target_value: the value to find

    Returns:
        fractional_index: position in the pulse array (may be fractional)
        or None if target_value is not found
    """
    # Find the first sample >= target_value
    indices = np.where(pulse >= target_value)[0]

    if len(indices) == 0:
        return None

    idx = indices[0]

    if idx == 0:
        return 0.0

    # Interpolate between pulse[idx-1] and pulse[idx] to find exact position
    val_before = pulse[idx - 1]
    val_after = pulse[idx]

    if val_before == val_after:
        return float(idx)

    # Linear interpolation: fractional position between idx-1 and idx
    fraction = (target_value - val_before) / (val_after - val_before)
    return (idx - 1) + fraction


def resample_pulse_linear(pulse, upsample_factor=4):
    """
    Resample a pulse to higher resolution using linear interpolation.

    Args:
        pulse: array of pulse values (original sample points)
        upsample_factor: factor to increase resolution (e.g., 4 = 4x resolution)

    Returns:
        resampled_pulse: array of values at higher resolution
        resampled_indices: sample indices in original space for each resampled point
    """
    original_indices = np.arange(len(pulse))

    # Generate resampled indices (every 1/upsample_factor in original space)
    new_length = (len(pulse) - 1) * upsample_factor + 1
    resampled_indices = np.linspace(0, len(pulse) - 1, new_length)

    # Linear interpolation
    resampled_pulse = np.interp(resampled_indices, original_indices, pulse)

    # Clamp to valid range [0, 254] (should not be needed for linear, but safe)
    resampled_pulse = np.clip(resampled_pulse, 0, 254)

    return resampled_pulse, resampled_indices


def parse_datetime_from_filename(csv_filename):
    """
    Parse start datetime from filename like '20261004_002237_pulse_waveform.csv'.

    Args:
        csv_filename: filename (basename only, or full path)

    Returns:
        datetime object or None if pattern not found
    """
    basename = os.path.basename(csv_filename)
    # Pattern: YYYYMMDD_HHMMSS
    match = re.match(r'(\d{8})_(\d{6})', basename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS

        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        return datetime(year, month, day, hour, minute, second)
    return None


def generate_chunk_output_filenames(csv_file, chunk_start_time, png_file=None):
    """
    Generate output filenames with chunk start time appended.

    Args:
        csv_file: input CSV file path
        chunk_start_time: datetime object for chunk start
        png_file: optional output PNG file path

    Returns:
        (csv_output, png_output): output file paths with chunk time
    """
    base = csv_file.rsplit('.', 1)[0]  # Remove .csv extension
    time_str = chunk_start_time.strftime('%H%M')  # HHMM format

    csv_output = f"{base}_{time_str}_fit.csv"
    png_output = f"{base}_{time_str}.png"

    return csv_output, png_output


def load_pulse_data(csv_file):
    """Load pulse waveform data from CSV."""
    df = pd.read_csv(csv_file)
    # Try multiple possible column names for compatibility
    for col_name in ['waveform_value', 'value']:
        if col_name in df.columns:
            return df[col_name].values
    raise ValueError(f"No valid waveform column found. Available columns: {df.columns.tolist()}")


def smooth_5point_mean(data):
    """Apply 5-point moving average: mean of self and 2 neighbors on each side."""
    smoothed = np.convolve(data, np.ones(5)/5, mode='same')
    return smoothed


def find_local_minima(data, order=1):
    """Find indices of local minima using scipy.signal."""
    minima_indices = signal.argrelextrema(data, np.less, order=order)[0]
    return minima_indices


def extract_pulses_minmax(waveform, window_before=5, window_after=30,
                          min_samples=8, max_samples=34):
    """
    Extract pulses with two complete min/max cycles within a contiguous window.
    Window: starts 5 samples before the first minimum, contains 36 samples.
    Validation: only the FIRST min/max must have raw values != 0 and != 254.
    Rescaling: uses only the FIRST min/max for rescaling.

    Args:
        waveform: raw waveform data
        window_before: samples to capture before pulse start (5)
        window_after: samples to capture after pulse start (30, gives 36 total)
        min_samples: minimum pulse length in samples (8 for 180 bpm at 24 sps)
        max_samples: maximum pulse length in samples (34 for 50 bpm at 24 sps)

    Returns:
        pulses: array of extracted pulses (each rescaled using first min/max)
        pulse_start_indices: pulse start point (first minimum) within each extracted pulse
        pulse_info: list of dicts with metadata (first min/max, second min/max, pulse length)
    """
    # Smooth the data
    smoothed = smooth_5point_mean(waveform)

    # Find local minima
    minima_indices = find_local_minima(smoothed, order=1)

    if len(minima_indices) < 2:
        print(f"Warning: found only {len(minima_indices)} local minima")
        return np.array([]), np.array([]), []

    pulses = []
    pulse_start_indices = []
    pulse_info = []
    discarded_count = 0

    # Extract pulses using the first minimum as reference
    for i in range(len(minima_indices) - 1):
        first_min_idx = minima_indices[i]
        next_min_idx = minima_indices[i + 1]
        pulse_length = next_min_idx - first_min_idx

        # Check if pulse length is in valid range
        if pulse_length < min_samples or pulse_length > max_samples:
            discarded_count += 1
            continue

        # Extract fixed window: window_before before first minimum, window_after after
        window_start = first_min_idx - window_before
        window_end = first_min_idx + window_after + 1

        # Check if window is valid
        if window_start < 0 or window_end > len(waveform):
            discarded_count += 1
            continue

        # Extract window from raw waveform
        window = waveform[window_start:window_end]

        # Find FIRST minimum and maximum within the window
        # First minimum should be at index window_before
        first_min_in_window = window_before
        first_min_value = window[first_min_in_window]

        # Find first maximum: search after first minimum
        first_max_search = window[first_min_in_window:first_min_in_window + pulse_length]
        if len(first_max_search) == 0:
            discarded_count += 1
            continue
        first_max_in_search = np.argmax(first_max_search)
        first_max_in_window = first_min_in_window + first_max_in_search
        first_max_value = window[first_max_in_window]

        # VALIDATE: First min should not be 0, first max should not be 254 (raw values)
        # Also reject if first_min == first_max (would cause divide by zero in rescaling)
        if first_min_value == 0 or first_max_value == 254 or first_min_value == first_max_value:
            discarded_count += 1
            continue

        # Find SECOND minimum and maximum within the window
        # Second minimum should be near first_min_in_window + pulse_length
        second_min_search_start = first_max_in_window + 1
        if second_min_search_start >= len(window):
            discarded_count += 1
            continue

        second_min_search = window[second_min_search_start:]
        if len(second_min_search) == 0:
            discarded_count += 1
            continue
        second_min_in_search = np.argmin(second_min_search)
        second_min_in_window = second_min_search_start + second_min_in_search
        second_min_value = window[second_min_in_window]

        # Find second maximum: after second min (in the second pulse's rising edge)
        second_max_search = window[second_min_in_window + 1:]
        if len(second_max_search) > 0:
            second_max_in_search = np.argmax(second_max_search)
            second_max_in_window = second_min_in_window + 1 + second_max_in_search
            second_max_value = window[second_max_in_window]
        else:
            # If no data after second min, estimate using pulse_length spacing
            # The second max should occur roughly halfway to the next minimum
            estimated_next_min = second_min_in_window + pulse_length
            second_max_in_window = (second_min_in_window + min(estimated_next_min, len(window) - 1)) // 2
            second_max_value = window[second_max_in_window]

        # Rescale window using ONLY the FIRST min/max
        window_rescaled = (window - first_min_value) / (first_max_value - first_min_value) * 254.0

        # Pulse start point is at window_before in the extracted window (the first minimum)
        pulse_start_in_window = window_before

        pulses.append(window_rescaled)
        pulse_start_indices.append(pulse_start_in_window)

        # Calculate midpoint indices for both min/max pairs
        # Use average of min/max indices rather than searching for value
        # This is more robust and avoids issues with multiple occurrences of the same value
        first_midpoint_idx = (first_min_in_window + first_max_in_window) / 2.0
        second_midpoint_idx = (second_min_in_window + second_max_in_window) / 2.0

        pulse_info.append({
            'first_min': float(first_min_value),
            'first_max': float(first_max_value),
            'first_min_idx': first_min_in_window,
            'first_max_idx': first_max_in_window,
            'first_midpoint_idx': first_midpoint_idx,
            'second_min': float(second_min_value),
            'second_max': float(second_max_value),
            'second_min_idx': second_min_in_window,
            'second_max_idx': second_max_in_window,
            'second_midpoint_idx': second_midpoint_idx,
            'pulse_length': pulse_length
        })


    print(f"Extracted {len(pulses)} valid pulses")
    print(f"Discarded {discarded_count} pulses (length out of range or invalid window)")

    return np.array(pulses), np.array(pulse_start_indices), pulse_info


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


def write_ml_path_csv(ml_path, x_scale, output_file, midpoint_fractional_idx=4.0):
    """
    Write the median trace to a CSV file with time resolution.
    Time is calculated as: t = (sample_idx - midpoint_fractional_idx) / 24 seconds,
    where midpoint_fractional_idx = t=0 (pulse reference point, typically the midpoint between min/max).

    Args:
        ml_path: dict {column_idx: value} from median trace extraction
        x_scale: horizontal scaling factor (to convert columns back to samples)
        output_file: path to output CSV file
        midpoint_fractional_idx: reference point in original sample space (default 4.0 for backward compatibility)
    """
    # Convert bitmap columns to sample positions and sort
    columns = sorted(ml_path.keys())
    sample_positions = [c / x_scale for c in columns]
    # Convert samples to time: t = (sample - midpoint_fractional_idx) / 24 seconds
    time_positions = [(s - midpoint_fractional_idx) / 24 for s in sample_positions]
    values = [ml_path[c] for c in columns]

    # Write CSV with header
    with open(output_file, 'w') as f:
        f.write('time,value\n')
        for time, value in zip(time_positions, values):
            f.write(f'{time:.3f},{value:.2f}\n')

    print(f"Median trace written to {output_file}")


def find_median_trace(pulses, exclude_last=False):
    """
    Find the median trace by selecting the actual pulse closest to the per-sample median.

    Args:
        pulses: array of pulse waveforms (n_pulses × n_samples)
        exclude_last: if True, exclude the last pulse from being selected as median trace
                     (useful when the last pulse is at chunk boundary and may lack following pulse data)

    Returns:
        (median_pulse, selected_pulse_idx): the pulse from the dataset closest to the
                                            per-sample median values, and its index
    """
    # Compute per-sample median across all pulses
    median_trace = np.median(pulses, axis=0)

    # For each pulse, compute the total Euclidean distance from the median trace
    distances = []
    max_idx = len(pulses) - 1 if exclude_last else len(pulses)

    for i, pulse in enumerate(pulses):
        if i >= max_idx:
            distances.append(float('inf'))  # Exclude this pulse
            continue
        # Sum of squared distances at each sample point (Euclidean metric)
        euclidean_distance = np.sqrt(np.sum((pulse - median_trace) ** 2))
        distances.append(euclidean_distance)

    # Select the pulse closest to the median trace
    closest_pulse_idx = np.argmin(distances)
    median_pulse = pulses[closest_pulse_idx]

    if exclude_last and closest_pulse_idx == len(pulses) - 1:
        print(f"WARNING: Last pulse excluded from median selection (at chunk boundary)")

    print(f"Selected pulse {closest_pulse_idx} as median trace (distance: {distances[closest_pulse_idx]:.2f})")
    print(f"Median trace values: min={np.min(median_trace):.1f}, max={np.max(median_trace):.1f}")

    return median_pulse, closest_pulse_idx


def find_second_extrema(pulse, first_min_idx=None, min_threshold=50, min_distance=12):
    """
    Find the 2nd minimum and 2nd maximum in a pulse waveform.

    The 2nd minimum is the start of the following pulse, found by:
    1. Searching at least min_distance samples ahead from the first minimum
    2. Finding a point that is no more than min_threshold counts above the first minimum
    3. Confirming it's a local minimum (value increases after it)

    This ignores dicrotic notches and post-peak ripples.

    Args:
        pulse: array of pulse values (rescaled 0-254)
        first_min_idx: index of the first minimum (if None, finds it)
        min_threshold: maximum height above first minimum for a point to be considered 2nd min (default 50)
        min_distance: minimum samples to search ahead before looking for 2nd min (default 12, ~0.5 sec at 24 sps)

    Returns:
        (second_min_idx, second_max_idx, second_min_value, second_max_value) or (None, None, None, None) if not found
    """
    if len(pulse) < min_distance + 2:
        return None, None, None, None

    # Find first minimum if not provided
    if first_min_idx is None:
        first_min_idx = np.argmin(pulse)

    first_min_value = pulse[first_min_idx]

    # Search for 2nd minimum, but start at least min_distance samples ahead
    # This avoids finding immediate post-minimum ripples or dicrotic notches
    second_min_idx = None
    search_start = first_min_idx + min_distance

    # First, look for a point that meets the threshold
    for i in range(search_start, len(pulse)):
        if pulse[i] <= (first_min_value + min_threshold):
            # Check if this is a local minimum or valley (values increase after it)
            if i < len(pulse) - 1 and pulse[i + 1] > pulse[i]:
                second_min_idx = i
                break

    # If not found within threshold, find the actual local minimum in the search region
    # This handles cases where the pulse descent hasn't completed by the window end
    if second_min_idx is None and search_start < len(pulse):
        remaining_pulse = pulse[search_start:]
        local_min_offset = np.argmin(remaining_pulse)
        candidate_idx = search_start + local_min_offset

        # Accept the local minimum if it's reasonably a valley point
        # Check: is it lower than at least one neighbor?
        is_valley = False

        if candidate_idx < len(pulse) - 1:
            # Not at end: skip over any flat region (adjacent equal-valued points at minimum)
            last_min_idx = candidate_idx
            while last_min_idx < len(pulse) - 1 and pulse[last_min_idx + 1] == pulse[candidate_idx]:
                last_min_idx += 1

            # Check if value increases after the flat region
            if last_min_idx < len(pulse) - 1 and pulse[last_min_idx + 1] > pulse[candidate_idx]:
                is_valley = True
        elif candidate_idx > search_start:
            # At or near end: check if value decreases before it
            if pulse[candidate_idx - 1] > pulse[candidate_idx]:
                is_valley = True

        if is_valley:
            second_min_idx = candidate_idx

    if second_min_idx is None:
        return None, None, None, None

    second_min_value = pulse[second_min_idx]

    # Find maximum after the 2nd minimum
    if second_min_idx < len(pulse) - 1:
        remaining_pulse = pulse[second_min_idx + 1:]
        second_max_offset = np.argmax(remaining_pulse)
        second_max_idx = second_min_idx + 1 + second_max_offset
        second_max_value = pulse[second_max_idx]
    else:
        return None, None, None, None

    return second_min_idx, second_max_idx, second_min_value, second_max_value


def plot_rasterized(pulses, pulse_start_indices, csv_filename=None, pulse_info=None, value_min=0, value_max=254,
                    x_scale=4, y_scale=2, plot_title=None, show_colorbar=True):
    """
    Display rasterized waveforms as intensity map with time axis.
    Resamples pulses to 4x resolution for display, using midpoint between min/max as t=0 reference.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Resample all pulses to 4x resolution for smoother display
    upsample_factor = 4
    pulses_resampled = []
    midpoint_positions_resampled = []

    for i, pulse in enumerate(pulses):
        resampled_pulse, resampled_indices = resample_pulse_linear(pulse, upsample_factor)
        pulses_resampled.append(resampled_pulse)

        # Find first midpoint position in resampled pulse (use first midpoint as reference)
        if pulse_info and i < len(pulse_info):
            # In new structure, use first_midpoint_idx which is the reference point
            original_midpoint = pulse_info[i]['first_midpoint_idx']
            resampled_midpoint = original_midpoint * upsample_factor
            midpoint_positions_resampled.append(resampled_midpoint)
        else:
            # Fallback: estimate from resampled pulse
            midpoint_positions_resampled.append(len(resampled_pulse) / 2)

    pulses_resampled = np.array(pulses_resampled)

    # Compute average midpoint position for reference
    avg_midpoint_position = np.mean(midpoint_positions_resampled)

    # Scale pulse_start_indices to resampled space (not needed for display, but kept for compatibility)
    pulse_start_indices_resampled = pulse_start_indices * upsample_factor

    # Draw rasterized oscilloscope display using resampled pulses
    bitmap = draw_rasterized_oscilloscope(pulses_resampled, pulse_start_indices_resampled, value_min, value_max,
                                          x_scale, y_scale)

    # Display bitmap with greyscale intensity mapping (15% to 75% grey)
    cmap = plt.cm.gray.copy()
    cmap.set_under('black')

    n_samples_resampled = pulses_resampled.shape[1]
    # Convert resampled sample indices to time
    # Resampled sample rate = 24 * 4 = 96 sps
    # Time for each resampled sample: t = (resampled_sample_idx - avg_midpoint_position) / 96
    time_min = (0 - avg_midpoint_position) / 96
    time_max = (n_samples_resampled - 1 - avg_midpoint_position) / 96

    # Map intensity range to greyscale: 0.15 (15% grey) to 0.75 (75% grey)
    im = ax.imshow(bitmap[::-1, :], aspect='auto', cmap=cmap,
                   extent=[time_min, time_max, value_min, value_max],
                   origin='lower', interpolation='nearest',
                   norm=PowerNorm(gamma=0.25, vmin=0.5, vmax=bitmap.max()))

    # Find anchor on the original bitmap (brightest pixel before smoothing)
    # This gives us the true peak of the data
    anchor_column, anchor_value = find_trigger_anchor_fine(bitmap, x_scale, trigger_value=128,
                                                           value_min=value_min, value_max=value_max)

    # Convert anchor to sample position (for informational purposes)
    anchor_sample = anchor_column / x_scale

    # Debug: print anchor point info
    print(f"Peak detected at sample {anchor_sample:.2f}, value {anchor_value:.1f}")

    # Find the median trace: select the actual pulse closest to the per-sample median
    # This is done on the original (non-resampled) pulses for accurate distance calculation
    # Exclude the last pulse to ensure we can find the following pulse's minimum if needed
    median_pulse, selected_pulse_idx = find_median_trace(pulses, exclude_last=True)

    # Resample the median pulse to 4x resolution for display
    median_pulse_resampled, _ = resample_pulse_linear(median_pulse, upsample_factor=4)

    # Get the first midpoint position of the selected pulse for proper time alignment
    # The first midpoint is the reference point (t=0 in the time axis)
    if pulse_info and selected_pulse_idx < len(pulse_info):
        median_midpoint = pulse_info[selected_pulse_idx]['first_midpoint_idx'] * 4
    else:
        median_midpoint = avg_midpoint_position

    # Convert resampled median pulse to time positions for plotting
    # Resampled sample index i maps to time t = (i - median_midpoint) / 96 seconds
    sample_indices_resampled = np.arange(len(median_pulse_resampled))
    time_positions = (sample_indices_resampled - median_midpoint) / 96
    ml_values = median_pulse_resampled

    # Convert to column dict format for CSV export compatibility
    # Each sample index maps to a column in the bitmap
    ml_path = {int(sample_idx * x_scale): value for sample_idx, value in enumerate(median_pulse)}

    ax.plot(time_positions, ml_values, color='lime', linewidth=2.0, alpha=0.95,
            label='Median trace (selected pulse)')

    # Draw two vertical dotted lines at the midpoints between min/max pairs
    # Both midpoint indices are in the extracted window coordinate system

    first_midpoint_idx = pulse_info[selected_pulse_idx]['first_midpoint_idx']
    second_midpoint_idx = pulse_info[selected_pulse_idx]['second_midpoint_idx']

    # Convert midpoint indices to time coordinates
    # Time = (sample_index - reference_point) / 24 seconds
    # Reference point is the first midpoint (t=0), so we shift both relative to it
    first_midpoint_time = 0.0  # By definition, first midpoint is at t=0
    second_midpoint_time = (second_midpoint_idx - first_midpoint_idx) / 24.0

    # Draw first vertical line at first midpoint (already exists at t=0, but draw with label)
    ax.axvline(x=first_midpoint_time, color='lime', linestyle='--', linewidth=1.5,
               alpha=0.8, label='1st midpoint (t=0)')

    # Draw second vertical line at second midpoint
    ax.axvline(x=second_midpoint_time, color='lime', linestyle=':', linewidth=2.0,
               alpha=0.9, label='2nd midpoint')

    # Calculate BPM from the time between the two midpoints
    time_between_midpoints = second_midpoint_time  # Since first is at 0
    if time_between_midpoints > 0:
        bpm = 60.0 / time_between_midpoints
        # Display BPM on the plot at top-center to avoid obscuring data or legend
        ax.text(0.5, 0.97, f'Heart Rate: {bpm:.1f} BPM',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"Time between midpoints: {time_between_midpoints:.3f} seconds, Heart Rate: {bpm:.1f} BPM")

    ax.set_xlabel('Time (seconds, t=0 at pulse start)', fontsize=11)
    ax.set_ylabel('Waveform value', fontsize=11)

    # Ensure axis limits exactly match the imshow extent (no empty space)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(value_min, value_max)

    # Set x-axis ticks: major at 0.2 sec, minor at 0.1 sec
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # Set title from provided plot_title or construct from filename
    if plot_title:
        title_text = f'{plot_title} ({len(pulses)} pulses)'
    elif csv_filename:
        title_base = os.path.splitext(os.path.basename(csv_filename))[0]
        title_text = f'{title_base} ({len(pulses)} pulses)'
    else:
        title_text = f'Pulse Analysis ({len(pulses)} pulses)'

    ax.set_title(title_text, fontsize=12)
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)

    # Add colorbar only if requested
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, label='Accumulated intensity')

    plt.tight_layout()

    # Add version number in bottom-right corner (AFTER tight_layout to avoid clipping)
    ax.text(0.99, 0.01, f'v{VERSION}',
           transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
           horizontalalignment='right', color='gray', alpha=0.6)
    return fig, ml_path, selected_pulse_idx


def process_chunk(waveform_chunk, csv_file, chunk_start_time, output_file, x_scale, y_scale):
    """
    Process a single chunk of waveform data.

    Args:
        waveform_chunk: waveform data for this chunk
        csv_file: original input CSV file (for generating output names)
        chunk_start_time: datetime object for chunk start time
        output_file: base output PNG file path (or None)
        x_scale, y_scale: scaling factors
    """
    # Extract pulses from this chunk
    pulses, pulse_start_indices, pulse_info = extract_pulses_minmax(
        waveform_chunk, window_before=5, window_after=30,
        min_samples=8, max_samples=34
    )

    if len(pulses) == 0:
        print(f"  No pulses found in chunk starting {chunk_start_time.strftime('%H:%M:%S')}")
        return

    print(f"  Chunk {chunk_start_time.strftime('%H:%M:%S')}: {len(pulses)} pulses")

    # Construct plot title from CSV filename date and chunk start time
    # CSV filename format: YYYYMMDD_HHMMSS_... → extract YYYYMMDD
    plot_title = None
    csv_basename = os.path.basename(csv_file)
    if '_' in csv_basename:
        date_str = csv_basename.split('_')[0]  # e.g., "20261004"
        if len(date_str) == 8 and date_str.isdigit():
            try:
                # Parse date as YYYYMMDD
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                # Format as YYYY-MM-DD and add chunk time as HH:MM
                plot_title = f"{date_obj.strftime('%Y-%m-%d')}  {chunk_start_time.strftime('%H:%M')}"
            except ValueError:
                pass  # Fall back to default title if parsing fails

    # Create and display rasterized display (without colorbar for cleaner chunked output)
    fig, ml_path, selected_pulse_idx = plot_rasterized(pulses, pulse_start_indices, csv_filename=csv_file,
                                                       pulse_info=pulse_info,
                                                       value_min=0, value_max=254,
                                                       x_scale=x_scale, y_scale=y_scale,
                                                       plot_title=plot_title, show_colorbar=False)

    # Generate output filenames with chunk time
    csv_output, png_output = generate_chunk_output_filenames(csv_file, chunk_start_time, output_file)

    # Get the first midpoint reference from the selected pulse for CSV time axis
    midpoint_idx = pulse_info[selected_pulse_idx]['first_midpoint_idx'] if pulse_info else 4.0

    # Write ML path to CSV
    write_ml_path_csv(ml_path, x_scale, csv_output, midpoint_fractional_idx=midpoint_idx)

    # Save PNG
    fig.savefig(png_output, dpi=100, bbox_inches='tight')
    print(f"  Plot saved to {png_output}")

    plt.close(fig)


def main():
    print(f"Pulse Oscilloscope Rasterized Display v{VERSION}")

    if len(sys.argv) < 2:
        print("Usage: python pulse_oscilloscope_minmax.py <csv_file> [output_file] [x_scale] [y_scale] [-chunk MINUTES]")
        print("  Extracts pulses based on local minima detection")
        print("  Rescales each pulse: min→0, max→254")
        print("  Triggers on rising edge crossing 128 (window: 6 pre-trigger, 24 post-trigger)")
        print("  Displays rasterized waveforms with Viterbi maximum-likelihood overlay")
        print("  x_scale: horizontal scaling factor (default 4)")
        print("  y_scale: vertical scaling factor (default 2)")
        print("  -chunk MINUTES: divide data into N-minute chunks (filename must contain YYYYMMDD_HHMMSS)")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    x_scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    y_scale = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    # Check for -chunk argument
    chunk_minutes = None
    if '-chunk' in sys.argv:
        idx = sys.argv.index('-chunk')
        if idx + 1 < len(sys.argv):
            chunk_minutes = int(sys.argv[idx + 1])

    # Load raw waveform data
    waveform = load_pulse_data(csv_file)
    print(f"Loaded {len(waveform)} samples from {csv_file}")

    # If chunking is enabled, process by time chunks
    if chunk_minutes:
        print(f"Processing in {chunk_minutes}-minute chunks")

        # Parse start datetime from filename
        start_time = parse_datetime_from_filename(csv_file)
        if not start_time:
            print("Error: cannot parse datetime from filename. Expected format: YYYYMMDD_HHMMSS_...")
            sys.exit(1)

        print(f"Start time from filename: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Calculate chunk size in samples (24 sps)
        samples_per_chunk = chunk_minutes * 60 * 24
        num_chunks = int(np.ceil(len(waveform) / samples_per_chunk))

        print(f"Total {len(waveform)} samples = {num_chunks} chunks of ~{samples_per_chunk} samples")

        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * samples_per_chunk
            end_idx = min((chunk_idx + 1) * samples_per_chunk, len(waveform))

            chunk_start_time = start_time + timedelta(minutes=chunk_idx * chunk_minutes)
            waveform_chunk = waveform[start_idx:end_idx]

            process_chunk(waveform_chunk, csv_file, chunk_start_time, output_file, x_scale, y_scale)

    else:
        # Original behavior: process entire file
        # Parse start datetime from filename for title formatting
        start_time = parse_datetime_from_filename(csv_file)

        # Extract pulses based on local minima
        pulses, pulse_start_indices, pulse_info = extract_pulses_minmax(
            waveform, window_before=5, window_after=30,
            min_samples=8, max_samples=34
        )

        if len(pulses) == 0:
            print("No valid pulses found!")
            sys.exit(1)

        # Show statistics
        print(f"Pulse length range: {min(p['pulse_length'] for p in pulse_info)}-{max(p['pulse_length'] for p in pulse_info)} samples")
        print(f"Original value ranges (min-max) across all pulses:")
        original_mins = [p['first_min'] for p in pulse_info]
        original_maxes = [p['first_max'] for p in pulse_info]
        print(f"  Min values: {min(original_mins):.1f} to {max(original_mins):.1f}")
        print(f"  Max values: {min(original_maxes):.1f} to {max(original_maxes):.1f}")

        print(f"Horizontal resolution: {pulses.shape[1] * x_scale} pixels ({x_scale}x scaling) for {pulses.shape[1]} samples")
        print(f"Vertical resolution: {350 * y_scale} pixels ({y_scale}x scaling) for value range [0, 254]")
        print(f"Time axis: midpoint between min/max = t=0 sec (pulse reference), range -0.17 to +1.08 seconds")

        # Construct plot title from CSV filename date and start time
        plot_title = None
        csv_basename = os.path.basename(csv_file)
        if '_' in csv_basename and start_time:
            date_str = csv_basename.split('_')[0]  # e.g., "20261004"
            if len(date_str) == 8 and date_str.isdigit():
                try:
                    # Parse date as YYYYMMDD
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                    # Format as YYYY-MM-DD and add start time as HH:MM
                    plot_title = f"{date_obj.strftime('%Y-%m-%d')}  {start_time.strftime('%H:%M')}"
                except ValueError:
                    pass  # Fall back to default title if parsing fails

        # Create and display rasterized display
        fig, ml_path, selected_pulse_idx = plot_rasterized(pulses, pulse_start_indices, csv_filename=csv_file,
                                                           pulse_info=pulse_info,
                                                           value_min=0, value_max=254,
                                                           x_scale=x_scale, y_scale=y_scale,
                                                           plot_title=plot_title)

        # Always generate CSV output from the input CSV filename
        csv_output = csv_file.rsplit('.', 1)[0] + '_fit.csv'

        # Get the midpoint reference from the selected pulse for CSV time axis
        midpoint_idx = pulse_info[selected_pulse_idx]['first_midpoint_idx'] if pulse_info else 4.0

        write_ml_path_csv(ml_path, x_scale, csv_output, midpoint_fractional_idx=midpoint_idx)

        if output_file:
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()


if __name__ == '__main__':
    main()
