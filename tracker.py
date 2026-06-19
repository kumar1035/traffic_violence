"""
tracker.py
Wraps DeepSORT to assign persistent IDs to vehicles across video frames.
This is what enables wrong-side driving (trajectory direction over time)
and illegal parking (dwell time in a zone) detection, since single-frame
detection alone can't tell if the same car is the same car in the next frame.
"""

import time
from collections import defaultdict, deque

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    def __init__(self, max_age=30, n_init=3, trajectory_length=15):
        """
        max_age: how many frames a track survives without a new detection
                 before being dropped (handles brief occlusions)
        n_init: how many consecutive detections needed before a track is
                confirmed (filters out flickering false detections)
        trajectory_length: how many recent center-points to keep per track,
                            used for wrong-side direction calculation
        """
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)
        self.trajectory_length = trajectory_length

        # track_id -> deque of (x, y) center points, most recent last
        self.trajectories = defaultdict(lambda: deque(maxlen=trajectory_length))

        # track_id -> timestamp when first seen inside a no-parking zone
        # (None if not currently inside any zone)
        self.zone_entry_time = {}

    def update(self, objects, frame):
        """
        Feeds one frame's detections into the tracker and returns tracked
        vehicles with persistent IDs.

        objects: list of detection dicts from detector.py's detect_objects()
                 (class_name, confidence, bbox)
        frame: the actual image/frame (needed by DeepSORT for appearance
               embedding matching)

        Returns: list of dicts {track_id, class_name, confidence, bbox, center}
        """
        vehicle_classes = {"car", "motorcycle", "bus", "truck"}
        vehicle_dets = [o for o in objects if o["class_name"] in vehicle_classes]

        # DeepSORT expects format: ([x1, y1, w, h], confidence, class_name)
        formatted = []
        for det in vehicle_dets:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            formatted.append(([x1, y1, w, h], det["confidence"], det["class_name"]))

        tracks = self.tracker.update_tracks(formatted, frame=frame)

        tracked_vehicles = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            self.trajectories[track_id].append(center)

            tracked_vehicles.append({
                "track_id": track_id,
                "class_name": track.get_det_class() or "vehicle",
                "confidence": track.get_det_conf() or 0.5,
                "bbox": (x1, y1, x2, y2),
                "center": center,
            })

        return tracked_vehicles

    def get_trajectory(self, track_id):
        """Returns the recent center-point history for a track ID."""
        return list(self.trajectories.get(track_id, []))

    def check_zone_dwell(self, tracked_vehicle, zone_polygon):
        """
        Tracks how long a vehicle has continuously been inside a zone polygon.
        Call this every frame for every tracked vehicle. Returns dwell time
        in seconds if currently inside the zone, else None.
        """
        track_id = tracked_vehicle["track_id"]
        center = tracked_vehicle["center"]

        inside = _point_in_polygon(center, zone_polygon)

        if inside:
            if track_id not in self.zone_entry_time:
                self.zone_entry_time[track_id] = time.time()
            dwell = time.time() - self.zone_entry_time[track_id]
            return dwell
        else:
            # vehicle left the zone, reset its timer
            if track_id in self.zone_entry_time:
                del self.zone_entry_time[track_id]
            return None


def _point_in_polygon(point, polygon):
    """
    Standard ray-casting point-in-polygon test.
    polygon: list of (x, y) tuples
    point: (x, y) tuple
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def draw_tracks(frame, tracked_vehicles, trajectories=None):
    """
    Draws tracked vehicle boxes with their persistent ID labels.
    Optionally draws trajectory trails if trajectories dict is provided.
    """
    annotated = frame.copy()

    for v in tracked_vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = f'ID {v["track_id"]} {v["class_name"]}'

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        if trajectories is not None:
            points = trajectories.get(v["track_id"], [])
            for i in range(1, len(points)):
                pt1 = tuple(int(c) for c in points[i - 1])
                pt2 = tuple(int(c) for c in points[i])
                cv2.line(annotated, pt1, pt2, (0, 200, 255), 2)

    return annotated


if __name__ == "__main__":
    # Standalone test — run on a video file to verify tracking works.
    import os
    import sys
    from detector import TrafficDetector

    video_path = "test_images/test_video.mp4"  # change to your video filename

    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        print("Place your test video in test_images/ and update the filename")
        print("in this script's __main__ block, or pass a path as an argument:")
        print("  python tracker.py path/to/your/video.mp4")
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
        else:
            sys.exit(1)

    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video file.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, total frames: {frame_count}")

    detector = TrafficDetector()
    vehicle_tracker = VehicleTracker()

    out_path = "test_output_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 20, (frame_w, frame_h))

    # process every 2nd frame to keep things fast during testing
    frame_skip = 2
    frame_idx = 0
    processed = 0

    print("Processing video... (this may take a minute)")

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
        out.write(annotated)

        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed} frames, {len(tracked)} active tracks")

    cap.release()
    out.release()

    print(f"\nDone. Processed {processed} frames.")
    print(f"Saved annotated video to: {out_path}")
    print("Open it to verify each vehicle keeps the SAME ID number across frames")
    print("(this confirms tracking, not just per-frame detection, is working).")