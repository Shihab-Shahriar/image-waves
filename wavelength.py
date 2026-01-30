"""
Wavelength extraction from horizontal wave videos.

Batch-processes videos listed in tree.txt to measure wavelength by detecting
intensity minima (valleys) in column-averaged profiles.
"""

import argparse
from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MM_PER_PIXEL = {
    1: 0.1162655505173817,
    2: 0.11737089201877934,
    3: 0.11717834544176235,
    4: 0.12001920307249161,
    5: 0.12022120702091847,
    6: 0.11682242990654207,
    7: 0.12175818823815902,
    8: 0.12285012285012284,
    9: 0.12454851164528583,
    10: 0.1249375312343828,
    11: 0.126806999746386,
    12: 0.12883277505797474,
}

CM_TO_FREQ = {
    (1, 8): 2.19,
    (1, 10): 2.19,
    (1, 12): 2.2,
    (1, 14): 2.22,
    (1, 16): 2.22,
    (1, 18): 2.22,
    (1, 20): 2.22,
    (1, 22): 2.22,
    (1, 24): 2.28,
    (1, 26): 2.28,
    (2, 8): 2.28,
    (2, 10): 2.28,
    (2, 12): 2.28,
    (2, 14): 2.27,
    (2, 16): 2.27,
    (2, 18): 2.27,
    (2, 20): 2.27,
    (2, 22): 2.27,
    (2, 24): 2.27,
    (2, 26): 2.27,
    (3, 8): 2.27,
    (3, 10): 2.27,
    (3, 12): 2.27,
    (3, 14): 2.27,
    (3, 16): 2.27,
    (3, 18): 2.27,
    (3, 20): 2.27,
    (3, 22): 2.27,
    (3, 24): 2.28,
    (3, 26): 2.28,
    (4, 8): 2.28,
    (4, 10): 2.29,
    (4, 12): 2.29,
    (4, 14): 2.29,
    (4, 16): 2.29,
    (4, 18): 2.29,
    (4, 20): 2.29,
    (4, 22): 2.29,
    (4, 24): 2.29,
    (4, 26): 2.29,
    (5, 8): 2.29,
    (5, 10): 2.29,
    (5, 12): 2.29,
    (5, 14): 2.29,
    (5, 16): 2.29,
    (5, 18): 2.29,
    (5, 20): 2.29,
    (5, 22): 2.29,
    (5, 24): 2.29,
    (5, 26): 2.29,
    (6, 8): 2.3,
    (6, 10): 2.3,
    (6, 12): 2.31,
    (6, 14): 2.31,
    (6, 16): 2.31,
    (6, 18): 2.31,
    (6, 20): 2.31,
    (6, 22): 2.31,
    (6, 24): 2.31,
    (6, 26): 2.31,
    (7, 8): 2.31,
    (7, 10): 2.31,
    (7, 12): 2.31,
    (7, 14): 2.31,
    (7, 16): 2.31,
    (7, 18): 2.31,
    (7, 20): 2.31,
    (7, 22): 2.31,
    (7, 24): 2.31,
    (7, 26): 2.31,
    (8, 8): 2.31,
    (8, 10): 2.31,
    (8, 12): 2.31,
    (8, 14): 2.31,
    (8, 16): 2.31,
    (8, 18): 2.31,
    (8, 20): 2.31,
    (8, 22): 2.31,
    (8, 24): 2.31,
    (8, 26): 2.31,
    (9, 8): 2.31,
    (9, 10): 2.31,
    (9, 12): 2.31,
    (9, 14): 2.31,
    (9, 16): 2.31,
    (9, 18): 2.31,
    (9, 20): 2.31,
    (9, 22): 2.31,
    (9, 24): 2.31,
    (9, 26): 2.31,
    (10, 8): 2.31,
    (10, 10): 2.31,
    (10, 12): 2.31,
    (10, 14): 2.31,
    (10, 16): 2.31,
    (10, 18): 2.32,
    (10, 20): 2.32,
    (10, 22): 2.32,
    (10, 24): 2.32,
    (10, 26): 2.32,
    (11, 8): 2.31,
    (11, 10): 2.31,
    (11, 12): 2.31,
    (11, 14): 2.31,
    (11, 16): 2.31,
    (11, 18): 2.31,
    (11, 20): 2.28,
    (11, 22): 2.28,
    (11, 24): 2.28,
    (11, 26): 2.28,
    (12, 8): 2.28,
    (12, 10): 2.28,
    (12, 12): 2.28,
    (12, 14): 2.28,
    (12, 16): 2.28,
    (12, 18): 2.28,
    (12, 20): 2.28,
    (12, 22): 2.28,
    (12, 24): 2.28,
    (12, 26): 2.28,
}

# --------------- Parameters ---------------
GAUSSIAN_SIGMA = 3
PROMINENCE_THRESHOLD = 5
DEFAULT_ROI = (866, 480, 1074, 685)  # x, y, w, h

VIDEO_NAME_RE = re.compile(r"(?P<cm>\d+)cm(?P<hz>\d+)hz\.mp4$", re.IGNORECASE)
TREE_FOLDER_RE = re.compile(r"^\s*\+---\s*(?P<name>.+?)\s*$")
CM_DIR_RE = re.compile(r"^(?P<cm>\d+)cm$", re.IGNORECASE)


def get_first_frame(video_path):
    """Read the first frame of the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read the first frame.")

    return frame


def clamp_roi_to_frame(roi, frame_shape):
    x, y, w, h = roi
    frame_h, frame_w = frame_shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    if w <= 0 or h <= 0:
        raise ValueError("Default ROI is outside the frame.")
    return x, y, w, h


def get_default_roi(video_path):
    """Return the default ROI for a video without prompting."""
    frame = get_first_frame(video_path)
    return clamp_roi_to_frame(DEFAULT_ROI, frame.shape)


def smooth_profile(col_avg, gaussian_sigma):
    """Apply Gaussian smoothing to the 1D profile (sigma=0 disables)."""
    if gaussian_sigma > 0:
        return gaussian_filter1d(col_avg, sigma=gaussian_sigma)
    return col_avg


def process_frame(frame, roi, params, scale_mm_per_pixel, signal_freq):
    """Process a single frame: crop, grayscale, column-average, smooth, find valleys."""
    x, y, w, h = roi
    cropped = frame[y : y + h, x : x + w]
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
        real_wavelength_mm = avg_pixel_wavelength * scale_mm_per_pixel
        valley_intensity_std = float(np.std(smoothed[valleys]))
        avg_prominence = float(np.mean(properties.get("prominences", [])))
        velocity = real_wavelength_mm * signal_freq  # mm/s
        return real_wavelength_mm, smoothed, valleys, valley_intensity_std, avg_prominence, velocity
    return None, smoothed, valleys, None, None, None


def evaluate_params(
    video_path,
    roi,
    params,
    scale_mm_per_pixel,
    signal_freq,
    collect_sample=False,
    collect_series=False,
):
    frame_wavelengths = []
    valley_counts = []
    valley_intensity_stds = []
    valley_prominences = []
    frame_idx = 0
    sample_smoothed = None
    sample_valleys = None
    velocities = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        wl, smoothed, valleys, valley_intensity_std, avg_prominence, velocity = process_frame(
            frame,
            roi,
            params,
            scale_mm_per_pixel,
            signal_freq,
        )

        if wl is not None:
            frame_wavelengths.append(wl)
            valley_counts.append(len(valleys))
            if valley_intensity_std is not None:
                valley_intensity_stds.append(valley_intensity_std)
            if avg_prominence is not None:
                valley_prominences.append(avg_prominence)
            if velocity is not None:
                velocities.append(velocity)

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
        avg_valley_intensity_std = (
            float(np.mean(valley_intensity_stds)) if valley_intensity_stds else float("nan")
        )
        avg_valley_prominence = (
            float(np.mean(valley_prominences)) if valley_prominences else float("nan")
        )
        avg_velocity = float(np.mean(velocities)) if velocities else float("nan")
    else:
        wavelengths = np.array([])
        mean_wl = float("nan")
        std_wl = float("nan")
        avg_valleys = float("nan")
        avg_valley_intensity_std = float("nan")
        avg_valley_prominence = float("nan")
        avg_velocity = float("nan")
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
        "avg_velocity": avg_velocity,
    }


def parse_tree_file(tree_path):
    current_folder = Path(".")
    video_paths = []
    try:
        lines = tree_path.read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"tree.txt not found: {tree_path}")

    for line in lines:
        folder_match = TREE_FOLDER_RE.match(line)
        if folder_match:
            current_folder = Path(folder_match.group("name").strip())
            continue

        line_stripped = line.strip()
        file_match = VIDEO_NAME_RE.search(line_stripped)
        if file_match:
            filename = file_match.group(0)
            video_paths.append(current_folder / filename)

    return video_paths


def discover_videos(root):
    video_paths = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Root not found: {root}")

    for entry in entries:
        if not entry.is_dir():
            continue
        if not CM_DIR_RE.match(entry.name):
            continue
        for video in sorted(entry.iterdir(), key=lambda p: p.name):
            if not video.is_file():
                continue
            if VIDEO_NAME_RE.search(video.name):
                video_paths.append(video.relative_to(root))

    return video_paths


def parse_cm_hz(filename):
    match = VIDEO_NAME_RE.search(filename)
    if not match:
        return None, None
    return int(match.group("cm")), int(match.group("hz"))


def plot_diagnostics(wavelengths, mean_wl, std_wl, sample_smoothed, sample_valleys, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax1 = axes[0]
    pixel_positions = np.arange(len(sample_smoothed))
    ax1.plot(pixel_positions, sample_smoothed, label="Smoothed intensity")
    ax1.plot(sample_valleys, sample_smoothed[sample_valleys], "rv", markersize=8, label="Detected minima")
    ax1.set_xlabel("Column (pixels)")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Sample Frame: Column-Averaged Intensity Profile")
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(wavelengths, linewidth=0.8)
    ax2.axhline(
        mean_wl,
        color="r",
        linestyle="--",
        label=f"Mean = {mean_wl:.3f} mm, Std = {std_wl:.3f} mm",
    )
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Wavelength (mm)")
    ax2.set_title("Wavelength Over Time")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_video(video_path, output_path):
    cm_value, hz_value = parse_cm_hz(video_path.name)
    if cm_value is None or hz_value is None:
        print(f"Skipping (missing cm/hz): {video_path}")
        return

    if cm_value not in MM_PER_PIXEL:
        print(f"Skipping (unknown cm scale): {video_path}")
        return

    freq_key = (cm_value, hz_value)
    if freq_key not in CM_TO_FREQ:
        print(f"Skipping (missing CM_TO_FREQ mapping): {video_path}")
        return

    try:
        roi = get_default_roi(video_path)
    except (RuntimeError, ValueError) as exc:
        print(f"Skipping (ROI error): {video_path} ({exc})")
        return

    params = {
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "prominence": PROMINENCE_THRESHOLD,
    }

    try:
        stats = evaluate_params(
            str(video_path),
            roi,
            params,
            MM_PER_PIXEL[cm_value],
            CM_TO_FREQ[freq_key],
            collect_sample=True,
            collect_series=True,
        )
    except RuntimeError as exc:
        print(f"Skipping (video error): {video_path} ({exc})")
        return

    if stats["valid_count"] == 0:
        print(f"No valid wavelength measurements: {video_path}")
        return

    plot_diagnostics(
        stats["wavelengths"],
        stats["mean"],
        stats["std"],
        stats["sample_smoothed"],
        stats["sample_valleys"],
        output_path,
    )

    print(
        f"{video_path} | velocity={stats['avg_velocity']:.3f} mm/s | "
        f"mean_wl={stats['mean']:.3f} mm | std={stats['std']:.3f} mm | "
        f"valid={stats['valid_count']}/{stats['total_frames']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Batch wavelength extraction")
    parser.add_argument(
        "--root",
        default=".",
        help="Root of the dataset folder structure",
    )
    parser.add_argument(
        "--tree",
        default=None,
        help="Optional path to tree.txt listing the folder structure",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory (relative to root) for output plots",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tree:
        tree_path = Path(args.tree)
        if not tree_path.is_absolute():
            tree_path = root / tree_path
        video_rel_paths = parse_tree_file(tree_path)
    else:
        video_rel_paths = discover_videos(root)

    print(f"Found {len(video_rel_paths)} videos to process.")
    if not video_rel_paths:
        print("No videos found in tree.txt.")
        return

    for rel_path in video_rel_paths:
        video_path = root / rel_path
        if not video_path.exists():
            print(f"Missing file, skipping: {video_path}")
            continue

        cm_value, hz_value = parse_cm_hz(video_path.name)
        if cm_value is None or hz_value is None:
            print(f"Skipping (missing cm/hz): {video_path}")
            continue

        output_name = f"wavelength_result_{cm_value}cm{hz_value}hz.png"
        output_path = output_dir / output_name
        process_video(video_path, output_path)


if __name__ == "__main__":
    main()
