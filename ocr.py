"""
ocr.py
Extracts license plate text from vehicle crops using EasyOCR, validates
against Indian plate format patterns, and provides face-blurring for
privacy-compliant evidence images.
"""

import re

import cv2
import easyocr
import numpy as np


# Indian plate format: 2 letters (state) + 2 digits (district) + 1-3 letters
# (series) + 4 digits (number). E.g. KA 01 AB 1234, MH12AB1234.
# We allow flexible spacing/no-spacing since OCR output varies.
INDIAN_PLATE_PATTERN = re.compile(
    r'^[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{3,4}$'
)


class PlateReader:
    def __init__(self, languages=None, gpu=True):
        """
        Initializes EasyOCR reader once. Loading this is slow (~5-10s),
        so keep one instance alive and reuse it across calls.
        """
        if languages is None:
            languages = ["en"]
        print("Loading EasyOCR reader...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR reader loaded.")

    def crop_plate_region(self, image, vehicle_bbox, region="lower_center"):
        """
        Crops a likely plate region from a vehicle's bounding box.
        Since we don't have a dedicated plate detector, we use a heuristic:
        the plate is typically in the lower-center portion of the vehicle.

        Returns the cropped image region.
        """
        x1, y1, x2, y2 = vehicle_bbox
        w = x2 - x1
        h = y2 - y1

        if region == "lower_center":
            # bottom 35% of the vehicle bbox, center 60% horizontally
            crop_y1 = y1 + int(h * 0.65)
            crop_y2 = y2
            crop_x1 = x1 + int(w * 0.2)
            crop_x2 = x2 - int(w * 0.2)
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = x1, y1, x2, y2

        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(image.shape[1], crop_x2)
        crop_y2 = min(image.shape[0], crop_y2)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        return image[crop_y1:crop_y2, crop_x1:crop_x2]

    def read_plate(self, plate_crop):
        """
        Runs EasyOCR on a cropped plate region.
        Returns (text, confidence) or (None, 0.0) if nothing readable found.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0

        # upscale small crops — OCR works better on larger text
        h, w = plate_crop.shape[:2]
        if h < 50:
            scale = 50 / h
            plate_crop = cv2.resize(plate_crop, (int(w * scale), 50))

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        results = self.reader.readtext(gray)

        if not results:
            return None, 0.0

        # pick the highest confidence text result
        best = max(results, key=lambda r: r[2])
        raw_text = best[1]
        confidence = best[2]

        cleaned = self._clean_plate_text(raw_text)
        return cleaned, confidence

    def _clean_plate_text(self, raw_text):
        """
        Removes spaces/special characters and uppercases, common OCR
        noise cleanup before validation.
        """
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
        return cleaned

    def validate_plate_format(self, plate_text):
        """
        Checks if cleaned plate text plausibly matches Indian plate format.
        Returns True/False. This filters out OCR noise that doesn't look
        like a real plate at all.
        """
        if plate_text is None:
            return False
        return bool(INDIAN_PLATE_PATTERN.match(plate_text))

    def extract_plate_from_vehicle(self, image, vehicle_bbox):
        """
        Full pipeline: crop -> OCR -> clean -> validate.
        Returns dict: {text, confidence, valid_format, crop, raw_text}

        Even if the cleaned text doesn't pass strict format validation,
        raw_text is still returned — useful for transparency in the UI
        ("here's our best guess, even if we're not fully confident it's
        a complete/correct plate read").
        """
        crop = self.crop_plate_region(image, vehicle_bbox)
        text, confidence = self.read_plate(crop)
        valid = self.validate_plate_format(text)

        return {
            "text": text if valid else None,
            "raw_text": text,
            "confidence": round(confidence, 3) if confidence else 0.0,
            "valid_format": valid,
            "crop": crop,
        }


def blur_faces(image, face_bboxes, blur_strength=51):
    """
    Applies Gaussian blur to face regions for privacy-compliant evidence
    images. face_bboxes: list of (x1, y1, x2, y2) tuples.

    In practice, face regions can come from the pose model's head
    keypoints (nose/eyes/ears) expanded into a small bbox, since we don't
    have a dedicated face detector in this pipeline.
    """
    blurred = image.copy()

    for bbox in face_bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        face_region = blurred[y1:y2, x1:x2]
        blurred_region = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
        blurred[y1:y2, x1:x2] = blurred_region

    return blurred


def face_bbox_from_pose(pose_keypoints, padding_ratio=0.6):
    """
    Derives an approximate face bounding box from pose keypoints
    (nose, eyes, ears) so we can blur faces without a dedicated face model.

    pose_keypoints: list of 17 [x, y, confidence] from detector.py's pose output
    Returns (x1, y1, x2, y2) or None if head keypoints aren't visible enough.
    """
    head_indices = [0, 1, 2, 3, 4]  # nose, left/right eye, left/right ear
    head_points = [pose_keypoints[i] for i in head_indices if pose_keypoints[i][2] > 0.3]

    if len(head_points) < 2:
        return None

    xs = [p[0] for p in head_points]
    ys = [p[1] for p in head_points]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    w = x2 - x1
    h = y2 - y1
    pad_x = max(w * padding_ratio, 15)
    pad_y = max(h * padding_ratio, 15)

    return (
        int(x1 - pad_x), int(y1 - pad_y),
        int(x2 + pad_x), int(y2 + pad_y),
    )


if __name__ == "__main__":
    # Standalone test
    import os
    import sys
    from detector import TrafficDetector

    test_dir = "test_images"
    image_files = [f for f in os.listdir(test_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    if not image_files:
        print(f"No images found in {test_dir}/")
        sys.exit(1)

    detector = TrafficDetector()
    plate_reader = PlateReader(gpu=True)

    for sample_file in image_files:
        sample_path = os.path.join(test_dir, sample_file)
        img = cv2.imread(sample_path)
        if img is None:
            continue

        print(f"\n{'='*60}")
        print(f"Image: {sample_file}")

        objects = detector.detect_objects(img, conf_threshold=0.4)
        vehicles = [o for o in objects if o["class_name"] in
                    ("car", "motorcycle", "bus", "truck")]

        print(f"Vehicles found: {len(vehicles)}")

        for i, vehicle in enumerate(vehicles):
            result = plate_reader.extract_plate_from_vehicle(img, vehicle["bbox"])
            print(f"  Vehicle {i+1} ({vehicle['class_name']}): "
                  f"text='{result['text']}' raw='{result['raw_text']}' "
                  f"conf={result['confidence']} valid_format={result['valid_format']}")

            if result["crop"] is not None:
                crop_name = f"plate_crop_{sample_file}_{i}.jpg"
                cv2.imwrite(crop_name, result["crop"])

        # face blur test on poses
        poses = detector.detect_pose(img, conf_threshold=0.4)
        face_boxes = []
        for pose in poses:
            fb = face_bbox_from_pose(pose["keypoints"])
            if fb:
                face_boxes.append(fb)

        if face_boxes:
            blurred_img = blur_faces(img, face_boxes)
            blur_name = f"face_blurred_{sample_file}"
            cv2.imwrite(blur_name, blurred_img)
            print(f"  Faces blurred: {len(face_boxes)}, saved {blur_name}")

    print("\nDone. Check the plate_crop_*.jpg and face_blurred_*.jpg files")
    print("saved in your project root to visually verify both work correctly.")