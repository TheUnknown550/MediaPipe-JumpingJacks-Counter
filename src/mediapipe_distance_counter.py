"""
MediaPipe Pose Landmarker - Distance-Based Jumping Jacks Counter
================================================================

This script uses MediaPipe's Pose Landmarker to detect jumping jacks from webcam input.
It uses distance-based thresholds to determine arm and leg positions.

Usage:
    python mediapipe_distance_counter.py

Configuration:
    - Update camera index (currently 1) if needed
    - Adjust distance thresholds based on distance from camera
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- IMPORTS ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
# Model setup
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_NAME = 'pose_landmarker_lite.task'
MODEL_PATH = os.path.join(script_dir, '..', 'models', MODEL_FILE_NAME)

if not os.path.exists(MODEL_PATH):
    print(f"\nCRITICAL ERROR: Model file not found at {MODEL_PATH}")
    exit()

# Distance thresholds (normalized coordinates 0.0 to 1.0)
# Adjust these based on your camera distance and body size
FEET_DIST_OPEN = 0.4      # If dist > 0.4, legs are OPEN
FEET_DIST_CLOSE = 0.2     # If dist < 0.2, legs are CLOSED
HAND_HIP_DIST_UP = 0.5    # If wrist is > 0.5 away from hip, arms are UP
HAND_HIP_DIST_DOWN = 0.2  # If wrist is < 0.2 away from hip, arms are DOWN


def calculate_distance(a, b):
    """
    Calculates the Euclidean distance between two landmarks.
    
    Args:
        a, b: MediaPipe landmarks with x, y attributes
        
    Returns:
        float: Distance (roughly 0.0 to 1.0 for normalized coordinates)
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    
    # Euclidean Distance: sqrt((x2-x1)^2 + (y2-y1)^2)
    dist = np.linalg.norm(a - b)
    return dist


# --- SETUP MEDIAPIPE ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)

# --- APP VARIABLES ---
counter = 0
stage = "down" 
feedback = "Stand in frame"
start_time = time.time()

# --- MAIN LOOP ---
cap = cv2.VideoCapture(1)  # Change to 0 if camera index 1 doesn't work

print("Starting Jumping Jacks Counter (Distance-Based)...")
print("Press 'q' to quit")
print("-" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # 1. Prepare Image
    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # 2. Detect pose
    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 3. Process Detection & Count
    if detection_result.pose_landmarks:
        lm = detection_result.pose_landmarks[0]
        
        # --- GET KEYPOINTS ---
        l_wrist = lm[15]
        r_wrist = lm[16]
        l_hip = lm[23]
        r_hip = lm[24]
        l_ankle = lm[27]
        r_ankle = lm[28]

        # --- CALCULATE DISTANCES ---
        
        # 1. Feet Spread (Distance between ankles)
        feet_spread = calculate_distance(l_ankle, r_ankle)
        
        # 2. Arm Lift (Distance from Wrist to Hip)
        # Average left and right side
        l_arm_dist = calculate_distance(l_wrist, l_hip)
        r_arm_dist = calculate_distance(r_wrist, r_hip)
        avg_arm_dist = (l_arm_dist + r_arm_dist) / 2

        # --- COUNTING LOGIC ---
        
        # UP STATE: Feet wide apart AND Hands far from hips
        if feet_spread > FEET_DIST_OPEN and avg_arm_dist > HAND_HIP_DIST_UP:
            if stage == "down":
                stage = "up"
                counter += 1
                feedback = "✓ GOOD REP"
        
        # DOWN STATE: Feet close together AND Hands close to hips
        elif feet_spread < FEET_DIST_CLOSE and avg_arm_dist < HAND_HIP_DIST_DOWN:
            stage = "down"
            feedback = "Ready"

        # --- VISUALIZATION ---
        h, w, _ = image.shape
        
        # Draw line between feet for visual feedback
        cx1, cy1 = int(l_ankle.x * w), int(l_ankle.y * h)
        cx2, cy2 = int(r_ankle.x * w), int(r_ankle.y * h)
        cv2.line(image, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        
        # Draw skeleton points
        for landmark in lm:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (245, 117, 66), -1)

    # --- UI DISPLAY ---
    cv2.rectangle(image, (0,0), (400,160), (245,117,16), -1)
    
    cv2.putText(image, f'COUNT: {counter}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f'STAGE: {stage}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show real-time distance values for debugging
    if detection_result.pose_landmarks:
        cv2.putText(image, f'Feet Dist: {feet_spread:.2f}', (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(image, f'Arm Dist:  {avg_arm_dist:.2f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow('Jumping Jack Counter - Distance Based', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print(f"\nFinal Count: {counter}")
        break

cap.release()
cv2.destroyAllWindows()
