

import math


SEVERITY_SCORES = {
    "helmet_violation": 8,
    "triple_riding": 7,
    "seatbelt_violation": 6,
    "wrong_side_driving": 9,
    "stop_line_violation": 7,
    "red_light_violation": 10,
    "illegal_parking": 4,
}


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _bbox_overlaps(bbox_a, bbox_b):
    """Returns True if two bounding boxes overlap at all."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _person_on_vehicle(person_bbox, vehicle_bbox, overlap_threshold=0.3):
   
    ax1, ay1, ax2, ay2 = person_bbox
    bx1, by1, bx2, by2 = vehicle_bbox

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix1 >= ix2 or iy1 >= iy2:
        return False

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    person_area = (ax2 - ax1) * (ay2 - ay1)

    if person_area == 0:
        return False

    overlap_ratio = intersection_area / person_area
    return overlap_ratio >= overlap_threshold


def _helmet_near_motorcycle(helmet_bbox, moto_bbox, max_distance_ratio=0.6):
    
    hx1, hy1, hx2, hy2 = helmet_bbox
    mx1, my1, mx2, my2 = moto_bbox

    head_center_x = (hx1 + hx2) / 2
    head_bottom_y = hy2

    moto_height = my2 - my1
    moto_width = mx2 - mx1
    max_gap = moto_height * max_distance_ratio

    
    horizontal_ok = (mx1 - moto_width * 0.3) <= head_center_x <= (mx2 + moto_width * 0.3)

    
    vertical_ok = (my1 - max_gap) <= head_bottom_y <= (my1 + moto_height * 0.5)

    return horizontal_ok and vertical_ok


def check_helmet(helmet_detections, motorcycles, conf_threshold=0.35):
    
    violations = []

    for det in helmet_detections:
        if det["class_name"] != "Without Helmet":
            continue
        if det["confidence"] < conf_threshold:
            continue

        on_motorcycle = any(
            _helmet_near_motorcycle(det["bbox"], m["bbox"]) for m in motorcycles
        )
        if not on_motorcycle:
            continue

        violations.append({
            "type": "helmet_violation",
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "reason": (
                f"Fine-tuned helmet model classified this head region as "
                f"'Without Helmet' with {det['confidence']:.2f} confidence, "
                f"positioned above a detected motorcycle."
            ),
            "severity": SEVERITY_SCORES["helmet_violation"],
        })

    return violations


def check_triple_riding(poses, motorcycles, max_legal_riders=2):
    
    violations = []

    for moto in motorcycles:
        riders = [p for p in poses if _person_on_vehicle(p["bbox"], moto["bbox"])]

        if len(riders) > max_legal_riders:
            avg_conf = sum(r["confidence"] for r in riders) / len(riders)
            violations.append({
                "type": "triple_riding",
                "confidence": round(avg_conf, 3),
                "bbox": tuple(int(v) for v in moto["bbox"]),
                "reason": (
                    f"{len(riders)} people detected overlapping a single "
                    f"motorcycle bounding box (legal limit: {max_legal_riders})."
                ),
                "severity": SEVERITY_SCORES["triple_riding"],
            })

    return violations


def check_stop_line(objects, stop_line_y, signal_is_red=True):
    
    violations = []

    if not signal_is_red:
        return violations

    vehicle_classes = {"car", "motorcycle", "bus", "truck"}

    for obj in objects:
        if obj["class_name"] not in vehicle_classes:
            continue

        x1, y1, x2, y2 = obj["bbox"]

        if y2 > stop_line_y:
            violations.append({
                "type": "stop_line_violation",
                "confidence": obj["confidence"],
                "bbox": obj["bbox"],
                "reason": (
                    f"{obj['class_name']} bbox bottom edge (y={y2}) crosses "
                    f"stop line (y={stop_line_y}) while signal is red."
                ),
                "severity": SEVERITY_SCORES["stop_line_violation"],
            })

    return violations


def check_illegal_parking(tracked_vehicle, zone_polygon, dwell_seconds, dwell_threshold=120):
    
    if dwell_seconds < dwell_threshold:
        return None

    return {
        "type": "illegal_parking",
        "confidence": tracked_vehicle["confidence"],
        "bbox": tracked_vehicle["bbox"],
        "reason": (
            f"Vehicle (track ID {tracked_vehicle['track_id']}) stationary "
            f"inside no-parking zone for {dwell_seconds}s "
            f"(threshold: {dwell_threshold}s)."
        ),
        "severity": SEVERITY_SCORES["illegal_parking"],
    }


def check_wrong_side(tracked_vehicle, trajectory_points, expected_direction):
   
    if len(trajectory_points) < 2:
        return None

    start = trajectory_points[0]
    end = trajectory_points[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    direction_map = {
        "left_to_right": dx > 0,
        "right_to_left": dx < 0,
        "top_to_bottom": dy > 0,
        "bottom_to_top": dy < 0,
    }

    moving_correctly = direction_map.get(expected_direction, True)

    if not moving_correctly:
        return {
            "type": "wrong_side_driving",
            "confidence": tracked_vehicle["confidence"],
            "bbox": tracked_vehicle["bbox"],
            "reason": (
                f"Vehicle (track ID {tracked_vehicle['track_id']}) trajectory "
                f"moves opposite to expected '{expected_direction}' flow "
                f"(dx={dx:.0f}, dy={dy:.0f})."
            ),
            "severity": SEVERITY_SCORES["wrong_side_driving"],
        }

    return None


def run_all_checks(objects, poses, helmets, stop_line_y=None, signal_is_red=False, conf_filter=0.6):
   
    motorcycles = [o for o in objects if o["class_name"] == "motorcycle"]

    all_violations = []
    all_violations.extend(check_helmet(helmets, motorcycles))
    all_violations.extend(check_triple_riding(poses, motorcycles))

    if stop_line_y is not None:
        all_violations.extend(check_stop_line(objects, stop_line_y, signal_is_red))

    filtered = [v for v in all_violations if v["confidence"] >= conf_filter]
    return filtered


if __name__ == "__main__":
    # Standalone test using detector.py output
    import os
    import sys
    import cv2
    from detector import TrafficDetector, draw_detections

    test_dir = "test_images"
    output_dir = "test_outputs_violations"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print(f"No images found in {test_dir}/ — add test images first.")
        sys.exit(1)

    image_files = [f for f in os.listdir(test_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    if not image_files:
        print(f"No valid image files found in {test_dir}/")
        sys.exit(1)

    print(f"Found {len(image_files)} images. Loading models once...\n")
    detector = TrafficDetector()

    for sample_file in image_files:
        sample_path = os.path.join(test_dir, sample_file)
        img = cv2.imread(sample_path)
        if img is None:
            print(f"Could not read image: {sample_path}, skipping.")
            continue

        print(f"{'='*60}")
        print(f"Image: {sample_file}")

        results = detector.detect_all(img, conf_threshold=0.25)

        violations = run_all_checks(
            results["objects"],
            results["poses"],
            results["helmets"],
            stop_line_y=None,
            signal_is_red=False,
            conf_filter=0.25,
        )

        print(f"Objects detected: {len(results['objects'])} | "
              f"People with poses: {len(results['poses'])} | "
              f"Helmet detections: {len(results['helmets'])} | "
              f"Violations: {len(violations)}")

        for obj in results["objects"]:
            print(f"  [object] {obj['class_name']} | conf {obj['confidence']} | bbox {obj['bbox']}")

        for h in results["helmets"]:
            print(f"  [helmet model] {h['class_name']} | conf {h['confidence']} | bbox {h['bbox']}")

        motos = [o for o in results["objects"] if o["class_name"] == "motorcycle"]
        if not motos:
            print("  [debug] No motorcycle detected in this frame — helmet violations cannot attach to a vehicle.")

        for v in violations:
            print(f"  -> {v['type']} | conf {v['confidence']} | severity {v['severity']}/10")
            print(f"     reason: {v['reason']}")

        annotated = draw_detections(img, results["objects"])
        for v in violations:
            x1, y1, x2, y2 = v["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, v["type"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out_name = f"violations_{os.path.splitext(sample_file)[0]}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"Done. Check the '{output_dir}/' folder for all annotated outputs.")