"""
test_tracking_violations.py
Integration test: runs detection + tracking + wrong-side/parking violation
checks together on a video file. This proves the full tracking-based
violation pipeline works end-to-end before wiring it into the main app.

Usage:
  python test_tracking_violations.py path/to/video.mp4
"""

import os
import sys

import cv2

from detector import TrafficDetector
from tracker import VehicleTracker, draw_tracks
from violations import check_wrong_side, check_illegal_parking


# CONFIGURE THESE based on your video's actual layout.
# Open one frame of your video in an image viewer and note pixel coordinates.

# Example no-parking zone — a rectangle in the lower-left area of frame.
# Replace these 4 points with coordinates that make sense for YOUR video.
NO_PARKING_ZONE = [(50, 300), (250, 300), (250, 450), (50, 450)]

# Expected traffic flow direction for wrong-side detection.
# Options: "left_to_right" | "right_to_left" | "top_to_bottom" | "bottom_to_top"
EXPECTED_DIRECTION = "left_to_right"

# How many seconds a vehicle must dwell in the no-parking zone to be flagged
PARKING_DWELL_THRESHOLD = 5  # lowered for quick testing; use 120 in production


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_tracking_violations.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path} | {frame_w}x{frame_h} @ {fps:.1f} fps")
    print(f"No-parking zone: {NO_PARKING_ZONE}")
    print(f"Expected direction: {EXPECTED_DIRECTION}")
    print(f"Parking dwell threshold: {PARKING_DWELL_THRESHOLD}s (lowered for testing)\n")

    detector = TrafficDetector()
    vehicle_tracker = VehicleTracker()

    out_path = "test_output_tracking_violations.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    frame_skip = 2
    frame_idx = 0
    flagged_violations = []  # collect unique violations seen across the video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        objects = detector.detect_objects(frame, conf_threshold=0.4)
        tracked = vehicle_tracker.update(objects, frame)

        annotated = draw_tracks(frame, tracked, vehicle_tracker.trajectories)

        # draw the no-parking zone outline for visual reference
        pts = NO_PARKING_ZONE
        for i in range(len(pts)):
            cv2.line(annotated, pts[i], pts[(i + 1) % len(pts)], (0, 0, 255), 2)
        cv2.putText(annotated, "NO PARKING ZONE", (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for v in tracked:
            # --- Wrong-side check ---
            trajectory = vehicle_tracker.get_trajectory(v["track_id"])
            wrong_side = check_wrong_side(v, trajectory, EXPECTED_DIRECTION)
            if wrong_side:
                key = f"wrong_side_{v['track_id']}"
                if key not in [f["key"] for f in flagged_violations]:
                    flagged_violations.append({"key": key, **wrong_side})
                    print(f"[FRAME {frame_idx}] {wrong_side['reason']}")
                x1, y1, x2, y2 = v["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, "WRONG SIDE", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- Illegal parking check ---
            dwell = vehicle_tracker.check_zone_dwell(v, NO_PARKING_ZONE)
            if dwell is not None:
                parking_violation = check_illegal_parking(
                    v, NO_PARKING_ZONE, dwell, dwell_threshold=PARKING_DWELL_THRESHOLD
                )
                if parking_violation:
                    key = f"parking_{v['track_id']}"
                    if key not in [f["key"] for f in flagged_violations]:
                        flagged_violations.append({"key": key, **parking_violation})
                        print(f"[FRAME {frame_idx}] {parking_violation['reason']}")
                    x1, y1, x2, y2 = v["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 3)
                    cv2.putText(annotated, "ILLEGAL PARKING", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        out.write(annotated)

        if frame_idx % 40 == 0:
            print(f"  ...processed frame {frame_idx}, {len(tracked)} active tracks")

    cap.release()
    out.release()

    print(f"\n{'='*60}")
    print(f"Total unique violations flagged: {len(flagged_violations)}")
    for f in flagged_violations:
        print(f"  - {f['type']} (track {f.get('bbox')})")
    print(f"\nSaved annotated video to: {out_path}")
    print("Red boxes = wrong-side driving | Orange boxes = illegal parking")


if __name__ == "__main__":
    main()