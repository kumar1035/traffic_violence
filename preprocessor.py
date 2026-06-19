"""
preprocessor.py
Handles image enhancement before detection: resizing, contrast enhancement,
denoising, and special modes (night / rain) for poor visibility conditions.
"""

import cv2
import numpy as np


def resize_image(image, target_size=640):
    """
    Resize image so the longer side equals target_size, keeping aspect ratio.
    YOLOv8 handles letterbox padding internally, so this just speeds up inference.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Improves visibility in low-light / shadowed images without blowing out
    bright regions (like headlights), since it works on local tiles.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def denoise_image(image, strength=10):
    """
    Removes noise from rain / motion blur / low-quality camera feeds.
    fastNlMeansDenoisingColored works well for color images without
    excessive blurring of edges.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, strength, strength, 7, 21
    )
    return denoised


def sharpen_image(image):
    """
    Mild unsharp mask to recover edge detail lost during denoising.
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened


def night_mode(image):
    """
    Aggressive enhancement pipeline for low-light / nighttime footage.
    Boosts brightness, applies stronger CLAHE, then denoises.
    """
    brightened = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    enhanced = apply_clahe(brightened, clip_limit=3.5, tile_grid_size=(8, 8))
    denoised = denoise_image(enhanced, strength=8)
    return denoised


def rain_mode(image):
    """
    Pipeline tuned for rain streaks / wet-lens blur.
    Stronger denoise pass followed by a sharpen to recover edges.
    """
    denoised = denoise_image(image, strength=15)
    sharpened = sharpen_image(denoised)
    return sharpened


def apply_roi_mask(image, polygon_points):
    """
    Masks out everything outside a given zone polygon.
    polygon_points: list of (x, y) tuples defining the zone of interest,
    e.g. the stop-line area or no-parking zone.
    Returns the masked image and the mask itself (for visualization).
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask


def preprocess_pipeline(image, mode="normal", roi_polygon=None, target_size=640):
    """
    Main entry point. Runs the full preprocessing chain based on selected mode.

    mode: "normal" | "night" | "rain"
    roi_polygon: optional list of (x, y) points to mask a specific zone
    target_size: resize target for YOLO input

    Returns the processed image ready for detection.
    """
    processed = image.copy()

    if mode == "night":
        processed = night_mode(processed)
    elif mode == "rain":
        processed = rain_mode(processed)
    else:
        processed = apply_clahe(processed, clip_limit=1.5)

    if roi_polygon is not None:
        processed, _ = apply_roi_mask(processed, roi_polygon)

    processed = resize_image(processed, target_size)

    return processed


if __name__ == "__main__":
    # Quick standalone test — run this file directly to sanity-check the pipeline
    import sys
    import os

    test_dir = "test_images"
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print(f"No images found in {test_dir}/ — add a test image first.")
        sys.exit(1)

    sample_file = os.listdir(test_dir)[0]
    sample_path = os.path.join(test_dir, sample_file)

    img = cv2.imread(sample_path)
    if img is None:
        print(f"Could not read image: {sample_path}")
        sys.exit(1)

    print(f"Testing preprocessing on: {sample_path}")
    print(f"Original size: {img.shape}")

    normal_result = preprocess_pipeline(img, mode="normal")
    night_result = preprocess_pipeline(img, mode="night")
    rain_result = preprocess_pipeline(img, mode="rain")

    cv2.imwrite("test_output_normal.jpg", normal_result)
    cv2.imwrite("test_output_night.jpg", night_result)
    cv2.imwrite("test_output_rain.jpg", rain_result)

    print("Saved test_output_normal.jpg, test_output_night.jpg, test_output_rain.jpg")
    print("Open these in VSCode to visually confirm the enhancement looks correct.")