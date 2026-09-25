"""
Stereo disparity and relative-depth estimation using OpenCV StereoSGBM.
Default inputs are OpenCV's aloeL/aloeR sample images.
Download the two images into data/ first, or pass their paths with --left and --right.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys


def discard_foreign_user_site_packages() -> None:
    """Prevent an IDE from loading packages built for a different Python version."""
    active_tag = f"python{sys.version_info.major}{sys.version_info.minor}"
    user_site_pattern = re.compile(
        r"appdata[\\/]+roaming[\\/]+python[\\/]+(python\d+)[\\/]+site-packages",
        re.IGNORECASE,
    )
    retained_paths = []
    for search_path in sys.path:
        matched_path = user_site_pattern.search(search_path)
        if matched_path and matched_path.group(1).lower() != active_tag:
            continue
        retained_paths.append(search_path)
    sys.path[:] = retained_paths


discard_foreign_user_site_packages()

import cv2
import numpy as np


def find_default_image(file_name: str, data_directory: Path) -> Path:
    """Locate an OpenCV sample image without assuming fixed installation path."""
    candidate = data_directory / file_name
    if candidate.exists():
        return candidate

    try:
        sample_path = cv2.samples.findFile(file_name, required=False)
    except TypeError:
        sample_path = ""

    if sample_path:
        return Path(sample_path)

    raise FileNotFoundError(
        f"Cannot find {file_name}. Run download_opencv_samples.py first, or use "
        f"--left/--right to specify the stereo pair."
    )


def normalize_for_display(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Rescale image intensity for visualization, ignore invalid pixels."""
    preview = np.zeros(image.shape, dtype=np.uint8)
    if np.any(valid_mask):
        low, high = np.percentile(image[valid_mask], (2, 98))
        if high > low:
            rescaled = (image - low) * 255.0 / (high - low)
            preview = np.clip(rescaled, 0, 255).astype(np.uint8)
    return preview


def main() -> None:
    # Define command-line options
    parser = argparse.ArgumentParser(description="OpenCV stereo depth estimation")
    parser.add_argument("--left", type=Path, help="Path to the left image")
    parser.add_argument("--right", type=Path, help="Path to the right image")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=128,
        help="Must be divisible by 16"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Odd matching window size"
    )
    options = parser.parse_args()

    # Check matching parameters
    if options.num_disparities <= 0 or options.num_disparities % 16 != 0:
        parser.error("--num-disparities must be a positive multiple of 16")
    if options.block_size < 3 or options.block_size % 2 == 0:
        parser.error("--block-size must be an odd integer >= 3")

    # Read the left and right views
    left_path = options.left or find_default_image("aloeL.jpg", options.data_dir)
    right_path = options.right or find_default_image("aloeR.jpg", options.data_dir)

    left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

    if left_image is None or right_image is None:
        raise RuntimeError("One or both input images could not be read")
    if left_image.shape != right_image.shape:
        raise ValueError(
            f"Image sizes must match; got {left_image.shape} and {right_image.shape}"
        )

    # Prepare grayscale inputs
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    window_size = options.block_size

    # Configure the stereo matcher
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=options.num_disparities,
        blockSize=window_size,
        P1=8 * (window_size ** 2),
        P2=32 * (window_size ** 2),
        disp12MaxDiff=1,
        uniquenessRatio=10,
        preFilterCap=63,
        speckleRange=2,
        speckleWindowSize=100,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # Convert fixed-point matching results to pixel disparities
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    valid_pixels = disparity > 0

    # Render the disparity preview
    disparity_preview = normalize_for_display(disparity, valid_pixels)
    disparity_color = cv2.applyColorMap(disparity_preview, cv2.COLORMAP_TURBO)
    disparity_color[~valid_pixels] = 0

    # Estimate uncalibrated relative depth from inverse disparity
    relative_depth = np.zeros_like(disparity)
    relative_depth[valid_pixels] = 1.0 / disparity[valid_pixels]
    depth_preview = normalize_for_display(relative_depth, valid_pixels)
    depth_color = cv2.applyColorMap(255 - depth_preview, cv2.COLORMAP_TURBO)
    depth_color[~valid_pixels] = 0

    # Create the destination directory
    options.output.mkdir(parents=True, exist_ok=True)

    # Write images and disparity data
    cv2.imwrite(str(options.output / "left.png"), left_image)
    cv2.imwrite(str(options.output / "right.png"), right_image)
    cv2.imwrite(str(options.output / "disparity_gray.png"), disparity_preview)
    cv2.imwrite(str(options.output / "disparity_color.png"), disparity_color)
    cv2.imwrite(str(options.output / "relative_depth_color.png"), depth_color)
    np.save(options.output / "disparity_pixels.npy", disparity)

    # Assemble a compact comparison panel
    thumbnails = [
        cv2.resize(img, (0, 0), fx=0.33, fy=0.33)
        for img in (left_image, disparity_color, depth_color)
    ]
    panel = cv2.hconcat(thumbnails)
    cv2.imwrite(str(options.output / "stereo_result_panel.png"), panel)

    # Report input details and output location
    print(f"Left image: {left_path}")
    print(f"Right image: {right_path}")
    print(f"Image size: {left_image.shape[1]} x {left_image.shape[0]} px")
    valid_ratio = valid_pixels.mean() * 100
    print(f"Valid disparity ratio: {valid_ratio:.2f}%")
    print(f"Saved results to: {options.output.resolve()}")


if __name__ == "__main__":
    main()
