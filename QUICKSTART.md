# Quick Start Guide 🚀

## Installation (First Time Only)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download/Move model files to models/ folder
#    - pose_landmarker_lite.task (from MediaPipe)
#    - yolo11n-pose.pt (will auto-download or move existing)

# 3. Move data files to data/ folder
#    - calibration_image_sample_data_*.npy
```

## Running the Counters

### Option 1: MediaPipe Angle-Based (Recommended for Webcam)
```bash
python src/mediapipe_angle_counter.py
```
**Best for:** Real-time webcam counting with strict form validation

### Option 2: MediaPipe Distance-Based (Alternative)
```bash
python src/mediapipe_distance_counter.py
```
**Best for:** Alternative distance-based detection method

### Option 3: YOLO Pose (Video Files)
```bash
python src/yolo_pose_counter.py
```
**Best for:** Processing video files with detailed output analysis

## Configuration

Edit these settings in the script files:

**Camera Index** (for webcam scripts):
```python
cap = cv2.VideoCapture(1)  # 0 or 1, depends on your system
```

**Angle Thresholds** (mediapipe_angle_counter.py):
```python
ARM_STRAIGHT_THRESH = 130    # How straight arms must be
ARM_UP_ANGLE = 130           # Threshold for arms up
LEG_SPREAD_UP = 172          # Threshold for legs spread
```

**Distance Thresholds** (mediapipe_distance_counter.py):
```python
FEET_DIST_OPEN = 0.4         # Feet spread distance
HAND_HIP_DIST_UP = 0.5       # Hand-hip distance for arms up
```

**YOLO Settings** (yolo_pose_counter.py):
```python
VIDEO_PATH = "path/to/video.mp4"
IMGSZ = 640                  # Model input size
CONF_THRES = 0.25            # Confidence threshold
DEVICE = None                # "cpu" or GPU device ID
```

## Troubleshooting

**Model not found:**
```
Make sure models are in the models/ folder and path is correct
```

**Camera not found:**
```
Try changing camera index: cap = cv2.VideoCapture(0)
```

**Qt errors:**
```
Already handled in YOLO script, should work on most systems
```

**GPU not detected:**
```
Check with: python scripts/test_gpu.py
```

## File Organization

- `src/` - Main counter scripts
- `models/` - Pre-trained models
- `data/` - Calibration and data files
- `scripts/` - Utility scripts
- `outputs/` - Generated results (from YOLO counter)

## Tips for Best Results

1. **Good lighting** - Makes pose detection more accurate
2. **Full body visible** - Keep entire body in frame
3. **Adjust thresholds** - Change values based on your form
4. **Test first** - Run a quick test to calibrate thresholds
5. **Check distances** - Distance-based approach shows real-time metrics

---

**Need help?** Check `README.md` for detailed documentation.
