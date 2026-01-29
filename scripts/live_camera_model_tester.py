"""
Live camera tester for YOLO11n-Pose models (PT and TFLite).

Features:
- Opens the default webcam and runs pose inference.
- Supports switching between the PT model and TFLite variants (float16, int8).
- Press 's' to cycle models, 'q' to quit.
"""

from pathlib import Path
import time
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATHS = {
    "YOLO11n-Pose (PT)": PROJECT_ROOT / "models" / "tflite_models" / "no-pruning" / "yolo11n-pose_float32.tflite",
    "YOLO11n-Pose Float16 (TFLite)": PROJECT_ROOT / "models" / "tflite_models" / "25pruning" / "yolo11n-pose-pruned-25_float32.tflite",
    "YOLO11n-Pose Int8 (TFLite)": PROJECT_ROOT / "models" / "tflite_models" / "25pruning" / "yolo11n-pose-pruned-25_int8.tflite",
}

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# COCO-17 keypoint index map (matching training)
KPT = {
    "nose": 0,
    "l_sho": 5,
    "r_sho": 6,
    "l_elb": 7,
    "r_elb": 8,
    "l_wri": 9,
    "r_wri": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_kne": 13,
    "r_kne": 14,
    "l_ank": 15,
    "r_ank": 16,
}

EDGES = [
    ("l_sho", "r_sho"),
    ("l_sho", "l_elb"),
    ("l_elb", "l_wri"),
    ("r_sho", "r_elb"),
    ("r_elb", "r_wri"),
    ("l_sho", "l_hip"),
    ("r_sho", "r_hip"),
    ("l_hip", "r_hip"),
    ("l_hip", "l_kne"),
    ("l_kne", "l_ank"),
    ("r_hip", "r_kne"),
    ("r_kne", "r_ank"),
]


def draw_pose(frame, kpts):
    """Draw skeleton and keypoints on frame."""
    for a, b in EDGES:
        p1 = tuple(kpts[KPT[a]].astype(int))
        p2 = tuple(kpts[KPT[b]].astype(int))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for point in kpts:
        cv2.circle(frame, tuple(point.astype(int)), 4, (0, 0, 255), -1)
    return frame


def load_model(model_name):
    """Load YOLO model by name."""
    path = MODEL_PATHS[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Live camera tester for YOLO11n-Pose models.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_PATHS.keys()),
        default=list(MODEL_PATHS.keys())[0],
        help="Which model to start with.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (set to 0 for default webcam).")
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = list(MODEL_PATHS.keys())
    current_idx = model_names.index(args.model)
    model = load_model(model_names[current_idx])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    prev_time = time.time()

    print(f"Starting with model: {model_names[current_idx]}")
    print(f"Device: {DEVICE}")
    print("Press 's' to switch model, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Run inference
        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=DEVICE,
            verbose=False,
        )[0]

        overlay = frame.copy()

        if result.keypoints is not None and len(result.keypoints) > 0:
            kpts = result.keypoints.xy[0].cpu().numpy()
            draw_pose(overlay, kpts)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-8)
        prev_time = now

        # HUD
        cv2.putText(overlay, f"Model: {model_names[current_idx]}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLO11n-Pose Live", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            current_idx = (current_idx + 1) % len(model_names)
            print(f"Switching to: {model_names[current_idx]}")
            model = load_model(model_names[current_idx])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
