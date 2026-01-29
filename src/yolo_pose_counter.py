"""
YOLO Pose - Jumping Jacks Counter
==================================

This script uses Ultralytics YOLO11 pose detection to count jumping jacks from video input.
It processes video frames and generates annotated output videos.

Usage:
    python yolo_pose_counter.py

Configuration:
    - Update VIDEO_PATH to your video file
    - Adjust IMGSZ, CONF_THRES, and DEVICE as needed
    - Output is saved to outputs/ folder
"""

import os
import time

# Fix Qt plugin issues with OpenCV
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
from ultralytics import YOLO

# --- CONFIGURATION ---
# Update these paths for your system
VIDEO_PATH = "data/7.mp4"  # Update this path
MODEL_PATH = "yolo11n-pose.pt"          # Will auto-download if not found
OUTDIR = "outputs"                      # Output directory

# YOLO Parameters
IMGSZ = 640             # Model input size
CONF_THRES = 0.25       # Confidence threshold
DEVICE = None           # None (auto), "cpu", or GPU device ID ("0", "1", etc.)

os.makedirs(OUTDIR, exist_ok=True)

# --- COCO-17 KEYPOINTS MAPPING ---
KPT = {
    "nose": 0,
    "l_sho": 5,  "r_sho": 6,      # Shoulders
    "l_elb": 7,  "r_elb": 8,      # Elbows
    "l_wri": 9,  "r_wri": 10,     # Wrists
    "l_hip": 11, "r_hip": 12,     # Hips
    "l_kne": 13, "r_kne": 14,     # Knees
    "l_ank": 15, "r_ank": 16,     # Ankles
}

# Skeleton edges for visualization
EDGES = [
    ("l_sho", "r_sho"),
    ("l_sho", "l_elb"), ("l_elb", "l_wri"),
    ("r_sho", "r_elb"), ("r_elb", "r_wri"),
    ("l_sho", "l_hip"), ("r_sho", "r_hip"),
    ("l_hip", "r_hip"),
    ("l_hip", "l_kne"), ("l_kne", "l_ank"),
    ("r_hip", "r_kne"), ("r_kne", "r_ank"),
]

# --- UTILITY FUNCTIONS ---
def dist(a, b):
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def clamp01(x):
    """Clamp value between 0.0 and 1.0."""
    return max(0.0, min(1.0, x))


# --- JUMPING JACK COUNTER CLASS ---
class JumpingJackCounter:
    """Detects jumping jack state transitions and counts reps."""
    
    def __init__(self, window=7):
        """
        Args:
            window: Sliding window size for smoothing
        """
        self.state = "UNKNOWN"
        self.count = 0
        self.open_hist = deque(maxlen=window)
        self.closed_hist = deque(maxlen=window)

    def update(self, open_score, closed_score):
        """
        Update counter with new frame data.
        
        Args:
            open_score: Score indicating "open" position (legs spread, arms up)
            closed_score: Score indicating "closed" position (legs together, arms down)
            
        Returns:
            tuple: (open_avg, closed_avg, state, count)
        """
        self.open_hist.append(open_score)
        self.closed_hist.append(closed_score)

        # Smoothed averages
        o = np.mean(self.open_hist)
        c = np.mean(self.closed_hist)

        # State transition logic
        prev_state = self.state
        if o > 0.6 and o > c + 0.1:
            self.state = "OPEN"
        elif c > 0.6 and c > o + 0.1:
            self.state = "CLOSED"

        # Count on transition from CLOSED to OPEN
        if prev_state == "CLOSED" and self.state == "OPEN":
            self.count += 1

        return o, c, self.state, self.count


# --- MAIN PROCESSING ---
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video file: {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info: {W}x{H} @ {fps} FPS")

# Create video writers for outputs
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_overlay = cv2.VideoWriter(f"{OUTDIR}/overlay.mp4", fourcc, fps, (W, H))
out_kpts = cv2.VideoWriter(f"{OUTDIR}/keypoints_only.mp4", fourcc, fps, (W, H))
out_side = cv2.VideoWriter(f"{OUTDIR}/side_by_side.mp4", fourcc, fps, (W * 2, H))

counter = JumpingJackCounter()
logs = []

print("Processing frames...")
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    result = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, device=DEVICE, verbose=False)[0]

    # Initialize output images
    overlay = frame.copy()
    kpt_img = np.zeros_like(frame)

    open_s = closed_s = 0.0

    # Process keypoints if detected
    if result.keypoints is not None and len(result.keypoints) > 0:
        kpts = result.keypoints.xy[0].cpu().numpy()
        if result.keypoints.conf is not None:
            confs = result.keypoints.conf[0].cpu().numpy()
        else:
            confs = np.ones(len(kpts))
        
        # Check if we have all required keypoints (COCO-17 format)
        if len(kpts) >= 17:
            # Calculate metrics
            sho_w = dist(kpts[KPT["l_sho"]], kpts[KPT["r_sho"]])
            ank_w = dist(kpts[KPT["l_ank"]], kpts[KPT["r_ank"]])
            legs_ratio = ank_w / (sho_w + 1e-6)

            # Arm height: how much higher wrists are than shoulders
            wrists_y = (kpts[KPT["l_wri"]][1] + kpts[KPT["r_wri"]][1]) / 2
            shoulders_y = (kpts[KPT["l_sho"]][1] + kpts[KPT["r_sho"]][1]) / 2
            arms_up = clamp01((shoulders_y - wrists_y) / 100)

            # Combined scores
            open_s = 0.5 * clamp01((legs_ratio - 0.8)) + 0.5 * arms_up
            closed_s = 1.0 - open_s

            # Draw skeleton
            for a, b in EDGES:
                p1 = tuple(kpts[KPT[a]].astype(int))
                p2 = tuple(kpts[KPT[b]].astype(int))
                cv2.line(overlay, p1, p2, (0, 255, 0), 2)
                cv2.line(kpt_img, p1, p2, (0, 255, 0), 2)

            # Draw keypoints
            for i in range(len(kpts)):
                cv2.circle(overlay, tuple(kpts[i].astype(int)), 3, (0, 0, 255), -1)
                cv2.circle(kpt_img, tuple(kpts[i].astype(int)), 3, (0, 0, 255), -1)

    # Update counter
    o, c, state, count = counter.update(open_s, closed_s)

    # Draw UI
    cv2.putText(overlay, f"Count: {count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(overlay, f"State: {state}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Create side-by-side view
    side = np.concatenate([kpt_img, overlay], axis=1)

    # Write outputs
    out_overlay.write(overlay)
    out_kpts.write(kpt_img)
    out_side.write(side)

    # Log frame data
    logs.append({
        "frame": frame_id,
        "open_score": o,
        "closed_score": c,
        "state": state,
        "count": count
    })

    frame_id += 1
    if frame_id % 30 == 0:
        print(f"  Processed {frame_id} frames...", end='\r')

cap.release()
out_overlay.release()
out_kpts.release()
out_side.release()

# Save logs to CSV
df = pd.DataFrame(logs)
df.to_csv(f"{OUTDIR}/per_frame_log.csv", index=False)

# Generate signal plot
plt.figure(figsize=(12, 4))
plt.plot(df["open_score"], label="Open Score", linewidth=2)
plt.plot(df["closed_score"], label="Closed Score", linewidth=2)
plt.legend(fontsize=12)
plt.xlabel("Frame")
plt.ylabel("Score")
plt.title("Jumping Jack Detection Signal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/signals_plot.png", dpi=200)
plt.close()

# Print results
print("\n" + "=" * 50)
print("✅ PROCESSING COMPLETE")
print("=" * 50)
print(f"Total frames processed: {frame_id}")
print(f"Final Jumping Jacks Count: {df['count'].iloc[-1] if len(df) > 0 else 0}")
print(f"Output files saved in: {OUTDIR}/")
print(f"  - overlay.mp4 (pose overlay on original)")
print(f"  - keypoints_only.mp4 (skeleton only)")
print(f"  - side_by_side.mp4 (combined view)")
print(f"  - per_frame_log.csv (detailed logs)")
print(f"  - signals_plot.png (signal visualization)")
print("=" * 50)
