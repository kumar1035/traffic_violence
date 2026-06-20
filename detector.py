

import cv2
import torch
from ultralytics import YOLO


# COCO class IDs we care about (YOLOv8 default pretrained classes)
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TrafficDetector:
    def __init__(self, det_model_path="models/yolov8m.pt", pose_model_path="models/yolov8m-pose.pt",
                 helmet_model_path="models/helmet_best.pt"):
        
        print(f"Loading detection model on {DEVICE}...")
        self.det_model = YOLO(det_model_path)
        self.det_model.to(DEVICE)

        print(f"Loading pose model on {DEVICE}...")
        self.pose_model = YOLO(pose_model_path)
        self.pose_model.to(DEVICE)

        print(f"Loading helmet model on {DEVICE}...")
        self.helmet_model = YOLO(helmet_model_path)
        self.helmet_model.to(DEVICE)

        print("Models loaded successfully.")

    def detect_helmets(self, image, conf_threshold=0.4):
        """
        Runs the fine-tuned helmet detection model.
        Returns a list of dicts: {class_name, confidence, bbox}
        class_name will be "With Helmet" or "Without Helmet"
        """
        results = self.helmet_model(image, conf=conf_threshold, device=DEVICE, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            names = result.names
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": round(confidence, 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        return detections

    def detect_objects(self, image, conf_threshold=0.4):
        
        results = self.det_model(image, conf=conf_threshold, device=DEVICE, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in RELEVANT_CLASSES:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": RELEVANT_CLASSES[class_id],
                    "confidence": round(confidence, 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        return detections

    def detect_pose(self, image, conf_threshold=0.4):
        
        results = self.pose_model(image, conf=conf_threshold, device=DEVICE, verbose=False)

        poses = []
        for result in results:
            if result.keypoints is None:
                continue

            boxes = result.boxes
            keypoints_data = result.keypoints.data  # shape: (N, 17, 3)

            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                kpts = keypoints_data[i].tolist()  # list of [x, y, conf]

                poses.append({
                    "confidence": round(confidence, 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "keypoints": kpts,
                })

        return poses

    def detect_all(self, image, conf_threshold=0.4):
        
        objects = self.detect_objects(image, conf_threshold)
        poses = self.detect_pose(image, conf_threshold)
        helmets = self.detect_helmets(image, conf_threshold)
        return {"objects": objects, "poses": poses, "helmets": helmets}


def draw_detections(image, detections, color=(0, 255, 0)):
    
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


if __name__ == "__main__":
    # Standalone test — run this file directly to verify detection works
    import os
    import sys

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

    print(f"Running detection on: {sample_path}")

    detector = TrafficDetector()
    results = detector.detect_all(img)

    print(f"\nObjects detected: {len(results['objects'])}")
    for obj in results["objects"]:
        print(f"  {obj['class_name']} - confidence {obj['confidence']} - bbox {obj['bbox']}")

    print(f"\nPeople with pose keypoints: {len(results['poses'])}")
    for pose in results["poses"]:
        print(f"  person - confidence {pose['confidence']} - bbox {pose['bbox']}")

    print(f"\nHelmet detections: {len(results['helmets'])}")
    for h in results["helmets"]:
        print(f"  {h['class_name']} - confidence {h['confidence']} - bbox {h['bbox']}")

    annotated = draw_detections(img, results["objects"])
    helmet_colors = {"With Helmet": (0, 255, 0), "Without Helmet": (0, 0, 255)}
    for h in results["helmets"]:
        color = helmet_colors.get(h["class_name"], (255, 255, 0))
        annotated = draw_detections(annotated, [h], color=color)

    cv2.imwrite("test_output_detection.jpg", annotated)
    print("\nSaved test_output_detection.jpg — open it in VSCode to verify bounding boxes look correct.")