# Jumping Jacks Counter 🤸‍♂️

A computer vision-based jumping jacks counter application using pose estimation. This project implements multiple approaches to count jumping jacks from video input using MediaPipe and YOLO pose detection models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Approaches](#approaches)
- [Models](#models)
- [Requirements](#requirements)

## 🎯 Overview

This project detects and counts jumping jacks from video or webcam input using pose estimation. It provides multiple implementations:
- **MediaPipe-based counters** (angle-based and distance-based detection)
- **YOLO Pose-based counter** (using Ultralytics YOLO11 pose model)

Each approach calculates body joint angles and positions to determine when a jumping jack is completed.

### 🍓 IoT/Edge Deployment

This project is designed for edge computing on **Raspberry Pi** and other IoT devices:
- ✅ Lightweight MediaPipe models for RPi 4+
- ✅ Real-time processing on limited hardware
- ✅ Optional Edge TPU acceleration (Coral)
- ✅ Headless/server-mode operation
- ✅ System service auto-start capability

**See [IOT_DEPLOYMENT_GUIDE.md](IOT_DEPLOYMENT_GUIDE.md) for Raspberry Pi setup.**

## ✨ Features

- Real-time jumping jack counting from webcam or video file
- Multiple pose estimation backends (MediaPipe, YOLO)
- Configurable detection thresholds for arm/leg angles
- Video output with pose visualization
- Frame-by-frame analysis with angle calculations

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenCV (cv2)
- numpy
- pandas
- matplotlib

### Setup

1. Clone or navigate to the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required model files:
   - MediaPipe: `pose_landmarker_lite.task` (placed in root or script directory)
   - YOLO: `yolo11n-pose.pt` (automatically downloaded on first run)

## 📁 Project Structure

```
MediaPipe-JackpingJacks-Counter/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/                              # Main source code
│   ├── mediapipe_angle_counter.py    # MediaPipe angle-based approach
│   ├── mediapipe_distance_counter.py # MediaPipe distance-based approach
│   └── yolo_pose_counter.py          # YOLO pose-based approach
├── scripts/                          # Utility and helper scripts
│   ├── test_gpu.py                   # Test GPU availability
│   └── yolo_training_scripts/        # YOLO model training/quantization
├── models/                           # Pre-trained models storage
│   ├── pose_landmarker_lite.task     # MediaPipe pose model
│   ├── yolo11n-pose.pt               # YOLO pose model
│   ├── yolo11n-pose.onnx             # ONNX format
│   ├── yolo11n-pose.torchscript      # TorchScript format
│   ├── yolo11n-pose.mnn              # MNN format
│   └── yolo11n-pose_*.../            # Converted model formats
├── data/                             # Data and calibration
│   └── calibration_image_sample_data_20x128x128x3_float32.npy
└── outputs/                          # Generated outputs (results, frames, videos)
```

## 🚀 Usage

### Using MediaPipe (Webcam Input)
```bash
python src/mediapipe_angle_counter.py
```
Detects jumping jacks using arm/leg angle thresholds from webcam feed.

### Using MediaPipe (Video Input)
```bash
python src/mediapipe_distance_counter.py
```
Alternative approach using distance-based detection from video.

### Using YOLO Pose
```bash
python src/yolo_pose_counter.py
```
YOLO-based pose detection and jumping jack counting.

**Note:** Adjust file paths and configuration in each script as needed.

## 🔍 Approaches

### 1. MediaPipe Angle-Based (Recommended for Webcam)
- Uses MediaPipe Pose Landmarker
- Calculates angles between body joints
- Configurable thresholds:
  - `ARM_STRAIGHT_THRESH`: How straight arms must be
  - `ARM_UP_ANGLE`: Threshold for "arms up" position
  - `LEG_SPREAD_UP`: Threshold for legs spread

**File:** `src/mediapipe_angle_counter.py`

### 2. MediaPipe Distance-Based
- Uses point-to-point distance calculations
- Alternative approach for different environments

**File:** `src/mediapipe_distance_counter.py`

### 3. YOLO Pose-Based
- Uses Ultralytics YOLO11 pose detection
- Faster inference, good for real-time processing
- Support for multiple model formats (ONNX, TorchScript, MNN)

**File:** `src/yolo_pose_counter.py`

## 🧠 Models

### MediaPipe
- **Model:** `pose_landmarker_lite.task`
- **Type:** MediaPipe Pose Landmarker (Lite)
- **Keypoints:** 33 body landmarks
- **Download:** [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

### YOLO 11
- **Model:** `yolo11n-pose.pt`
- **Type:** Ultralytics YOLO11 Nano Pose
- **Keypoints:** COCO-17 format (17 body joints)
- **Download:** Automatically via `from ultralytics import YOLO`
- **Variants Available:**
  - `yolo11n-pose.pt` (PyTorch)
  - `yolo11n-pose.onnx` (ONNX)
  - `yolo11n-pose.torchscript` (TorchScript)
  - `yolo11n-pose.mnn` (MNN)

## 📊 Configuration

Each script contains configuration parameters at the top:

```python
# Angle thresholds (in degrees)
ARM_STRAIGHT_THRESH = 130    # How straight must arms be
ARM_UP_ANGLE = 130           # Threshold for "arms up"
ARM_DOWN_ANGLE = 40          # Threshold for "arms down"
LEG_SPREAD_UP = 172          # Legs spread threshold
LEG_SPREAD_DOWN = 175        # Legs close threshold

# YOLO parameters
IMGSZ = 640                  # Model input size
CONF_THRES = 0.25            # Confidence threshold
DEVICE = None                # "cpu" or GPU device ID
```

## 🔧 Troubleshooting

### Model Not Found
Ensure model files are in the correct directory or update the path in the script.

### Qt Platform Issues
Some systems may have Qt plugin issues. This is handled with:
```python
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
```

### GPU Issues
Check GPU availability with:
```bash
python scripts/test_gpu.py
```

## 🍓 Raspberry Pi & IoT Deployment

This project is optimized for deployment on **Raspberry Pi** and edge devices:

### Quick Start on RPi
1. **Setup:** See [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) (30-45 minutes)
2. **IoT Guide:** See [IOT_DEPLOYMENT_GUIDE.md](IOT_DEPLOYMENT_GUIDE.md)
3. **Requirements:** `requirements-rpi.txt` (optimized for RPi)

### Hardware Recommendations
- **Minimum:** RPi 4B (4GB RAM)
- **Recommended:** RPi 4B (8GB RAM) or RPi 5
- **Optional:** Google Coral TPU for faster inference

### Performance on RPi 4
- **MediaPipe Angle:** 15-20 FPS, 30-40% CPU
- **YOLO (without TPU):** 2-5 FPS, 95-100% CPU
- **YOLO (with Coral TPU):** 25-30 FPS, 20-30% CPU

### Key Features for IoT
- ✅ Real-time processing on limited hardware
- ✅ Headless/server mode operation
- ✅ Auto-start systemd service
- ✅ Cloud integration support
- ✅ Remote monitoring capabilities
- ✅ Edge TPU acceleration support

### Installation on RPi
```bash
# Use optimized dependencies
pip install -r requirements-rpi.txt

# Then see RASPBERRY_PI_SETUP.md for complete guide
```

## 📝 License

This project is part of an educational exercise for Week 3 - Group 3.

## 👥 Authors

- Group 3, Week 3
- Finland Project

---

**Note:** Update file paths in scripts to match your system configuration before running.
