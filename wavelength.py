"""
Wavelength extraction from horizontal wave video.

Processes 9cm14hz.mp4 to measure the wavelength of horizontal waves
by detecting intensity minima (valleys) in column-averaged profiles.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --------------- Configuration ---------------
VIDEO_PATH = "9cm14hz.mp4"
SCALE_MM_PER_PIXEL = 0.1  # mm per pixel

# --------------- Parameters ---------------
GAUSSIAN_SIGMA = 3
PROMINENCE_THRESHOLD = 5
OUTPUT_PLOT = "wavelength_result.png"
DEFAULT_ROI = (866, 480, 1074, 685)  # x, y, w, h


def get_first_frame(video_path):
    """Read the first frame of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read the first frame.")

    return frame


def select_roi(video_path):
    """Open the first frame and let the user draw a ROI."""
    frame = get_first_frame(video_path)

    # Resize for display so the window fits on screen
    max_display_width = 960
    orig_h, orig_w = frame.shape[:2]
    scale = min(max_display_width / orig_w, 1.0)
    display_frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))

    # Draw default ROI (Green box)
    dx_def = int(DEFAULT_ROI[0] * scale)
    dy_def = int(DEFAULT_ROI[1] * scale)
    dw_def = int(DEFAULT_ROI[2] * scale)
    dh_def = int(DEFAULT_ROI[3] * scale)
    cv2.rectangle(display_frame, (dx_def, dy_def), (dx_def + dw_def, dy_def + dh_def), (0, 255, 0), 2)
    cv2.putText(display_frame, "Green=Default. Draw new or ENTER for default.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    roi = cv2.selectROI("Select ROI", display_frame, showCrosshair=True)
    cv2.destroyAllWindows()

    dx, dy, dw, dh = roi

    # If user pressed Enter without drawing (area=0), use default
    if dw == 0 or dh == 0:
        print("Using default ROI.")
        x, y, w, h = DEFAULT_ROI
    else:
        # Scale ROI back to original resolution
        x = int(dx / scale)
        y = int(dy / scale)
        w = int(dw / scale)
        h = int(dh / scale)

    if w == 0 or h == 0:
        raise ValueError("Empty ROI selected. Exiting.")
    return x, y, w, h, frame


def smooth_profile(col_avg, gaussian_sigma):
    """Apply Gaussian smoothing to the 1D profile (sigma=0 disables)."""
    if gaussian_sigma > 0:
        return gaussian_filter1d(col_avg, sigma=gaussian_sigma)
    return col_avg


def process_frame(frame, roi, params):
    """Process a single frame: crop, grayscale, column-average, smooth, find valleys."""
    x, y, w, h = roi
    cropped = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Column-average: mean intensity along each column (average across rows)
    col_avg = gray.mean(axis=0)

    # Smooth
    smoothed = smooth_profile(col_avg, params["gaussian_sigma"])

    # Valley detection: find peaks on negated signal
    valleys, properties = find_peaks(-smoothed, prominence=params["prominence"])

    # Compute pixel wavelength from consecutive valley distances
    if len(valleys) >= 2:
        diffs = np.diff(valleys)
        avg_pixel_wavelength = diffs.mean()
        real_wavelength_mm = avg_pixel_wavelength * SCALE_MM_PER_PIXEL
        valley_intensity_std = float(np.std(smoothed[valleys]))
        avg_prominence = float(np.mean(properties.get("prominences", [])))
        return real_wavelength_mm, smoothed, valleys, valley_intensity_std, avg_prominence
    return None, smoothed, valleys, None, None


def evaluate_params(video_path, roi, params, collect_sample=False, collect_series=False):
    frame_wavelengths = []
    valley_counts = []
    valley_intensity_stds = []
    valley_prominences = []
    frame_idx = 0
    sample_smoothed = None
    sample_valleys = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        wl, smoothed, valleys, valley_intensity_std, avg_prominence = process_frame(frame, roi, params)

        if wl is not None:
            frame_wavelengths.append(wl)
            valley_counts.append(len(valleys))
            if valley_intensity_std is not None:
                valley_intensity_stds.append(valley_intensity_std)
            if avg_prominence is not None:
                valley_prominences.append(avg_prominence)

        if collect_sample and sample_smoothed is None and wl is not None:
            sample_smoothed = smoothed
            sample_valleys = valleys

        frame_idx += 1

    cap.release()

    if frame_wavelengths:
        wavelengths = np.array(frame_wavelengths)
        mean_wl = float(wavelengths.mean())
        std_wl = float(wavelengths.std())
        avg_valleys = float(np.mean(valley_counts))
        avg_valley_intensity_std = float(np.mean(valley_intensity_stds)) if valley_intensity_stds else float("nan")
        avg_valley_prominence = float(np.mean(valley_prominences)) if valley_prominences else float("nan")
    else:
        wavelengths = np.array([])
        mean_wl = float("nan")
        std_wl = float("nan")
        avg_valleys = float("nan")
        avg_valley_intensity_std = float("nan")
        avg_valley_prominence = float("nan")

    valid_ratio = len(frame_wavelengths) / frame_idx if frame_idx else 0.0

    return {
        "mean": mean_wl,
        "std": std_wl,
        "valid_ratio": valid_ratio,
        "valid_count": len(frame_wavelengths),
        "total_frames": frame_idx,
        "avg_valleys": avg_valleys,
        "avg_valley_intensity_std": avg_valley_intensity_std,
        "avg_valley_prominence": avg_valley_prominence,
        "wavelengths": wavelengths if collect_series else None,
        "sample_smoothed": sample_smoothed,
        "sample_valleys": sample_valleys,
    }


def main():
    x, y, w, h, _ = select_roi(VIDEO_PATH)
    roi = (x, y, w, h)
    print(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
    params = {
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "prominence": PROMINENCE_THRESHOLD,
    }

    # --- Process all frames using chosen params ---
    stats = evaluate_params(
        VIDEO_PATH,
        roi,
        params,
        collect_sample=True,
        collect_series=True,
    )

    if stats["valid_count"] == 0:
        print("No valid wavelength measurements obtained.")
        return

    wavelengths = stats["wavelengths"]
    mean_wl = stats["mean"]
    std_wl = stats["std"]
    frame_idx = stats["total_frames"]
    sample_smoothed = stats["sample_smoothed"]
    sample_valleys = stats["sample_valleys"]

    print(f"\n{'='*50}")
    print(f"Frames processed:  {frame_idx}")
    print(f"Valid measurements: {stats['valid_count']}")
    print(f"Mean wavelength:   {mean_wl:.3f} mm")
    print(f"Std deviation:     {std_wl:.3f} mm")
    print(f"{'='*50}")

    # --- Diagnostic plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: sample intensity profile with detected valleys
    ax1 = axes[0]
    pixel_positions = np.arange(len(sample_smoothed))
    ax1.plot(pixel_positions, sample_smoothed, label="Smoothed intensity")
    ax1.plot(sample_valleys, sample_smoothed[sample_valleys], "rv", markersize=8, label="Detected minima")
    ax1.set_xlabel("Column (pixels)")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Sample Frame: Column-Averaged Intensity Profile")
    ax1.legend()

    # Bottom: wavelength across frames
    ax2 = axes[1]
    ax2.plot(wavelengths, linewidth=0.8)
    ax2.axhline(mean_wl, color="r", linestyle="--", 
                label=f"Mean = {mean_wl:.3f} mm, Std = {std_wl:.3f} mm")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Wavelength (mm)")
    ax2.set_title("Wavelength Over Time")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nDiagnostic plot saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
