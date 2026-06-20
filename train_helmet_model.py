

import torch
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_helmet_model():
    print(f"Training on device: {DEVICE}")
    if DEVICE != "cuda":
        print("WARNING: CUDA not detected. Training on CPU will be very slow.")
        print("Stop now and check your environment if this is unexpected.")

    model = YOLO("models/yolov8s.pt")

    results = model.train(
        data="helmet_dataset/data.yaml",
        epochs=30,
        imgsz=416,         # lowered from 640 to reduce VRAM usage
        batch=4,            # lowered from 16 — 4GB card can't fit larger batches
        device=DEVICE,
        project="runs/detect",
        name="helmet_train",
        patience=10,        # stop early if no improvement for 10 epochs
        verbose=True,
        plots=True,         # saves training curves — useful for your presentation
        workers=0,           # disable multiprocessing workers — fixes Windows shared-memory crash
        cache=False,         # don't cache images in RAM, keep memory usage low
        amp=True,            # mixed precision — roughly halves VRAM usage
    )

    print("\nTraining complete.")
    print("Best weights saved to: runs/detect/helmet_train/weights/best.pt")
    print("\nNext step: copy that file into models/helmet_best.pt")
    print("Run this command in your terminal:")
    print("  copy runs\\detect\\helmet_train\\weights\\best.pt models\\helmet_best.pt")


if __name__ == "__main__":
    train_helmet_model()