# 📊 Repository Structure & Data Flow

## Complete Directory Tree

```
MediaPipe-JackpingJacks-Counter/
│
├── 📄 00_START_HERE.md ⭐
│   └── Read this first! Overview of the reorganization
│
├── 📖 README.md
│   └── Full documentation, features, troubleshooting
│
├── 📖 QUICKSTART.md
│   └── Quick reference for running scripts
│
├── 📖 FILE_MIGRATION_CHECKLIST.md
│   └── Step-by-step guide for finishing setup
│
├── 📖 REORGANIZATION_SUMMARY.md
│   └── Details about what was reorganized
│
├── 📄 requirements.txt
│   └── All Python dependencies (install with: pip install -r requirements.txt)
│
├── 📄 .gitignore
│   └── Git configuration (Python, project-specific)
│
├── 🗂️  src/ (NEW - Main Source Code)
│   ├── 🐍 mediapipe_angle_counter.py
│   │   ├── Input: Webcam stream
│   │   ├── Method: Angle-based detection
│   │   ├── Output: Live GUI with count
│   │   └── Best for: Real-time validation
│   │
│   ├── 🐍 mediapipe_distance_counter.py
│   │   ├── Input: Webcam stream
│   │   ├── Method: Distance-based detection
│   │   ├── Output: Live GUI with metrics
│   │   └── Best for: Alternative approach
│   │
│   └── 🐍 yolo_pose_counter.py
│       ├── Input: Video file
│       ├── Method: YOLO11 pose detection
│       ├── Output: Videos + plots + CSV
│       └── Best for: Post-processing
│
├── 🗂️  scripts/ (Utilities)
│   ├── 🐍 test_gpu.py
│   │   └── Checks GPU availability
│   │
│   └── 🗂️  yolo_training_scripts/ (existing)
│       ├── evaluate-model.py
│       ├── model-convert.py
│       └── yolo11n-pose.pt
│
├── 🗂️  models/ (NEW - Pre-trained Models)
│   ├── TODO: Move pose_landmarker_lite.task here
│   ├── TODO: Move yolo11n-pose.* here
│   ├── TODO: Move yolo11n-pose_ncnn_model/ here
│   └── TODO: Move yolo11n-pose_openvino_model/ here
│
├── 🗂️  data/ (NEW - Data & Calibration)
│   └── TODO: Move calibration_image_sample_data_*.npy here
│
├── 🗂️  outputs/ (AUTO-CREATED)
│   ├── overlay.mp4
│   ├── keypoints_only.mp4
│   ├── side_by_side.mp4
│   ├── signals_plot.png
│   └── per_frame_log.csv
│
├── 🗂️  runs/ (Existing - YOLO validation results)
│   └── pose/
│       └── val/, val2/, val3/, ... (many validation runs)
│
├── 🗂️  yolo11n-pose_ncnn_model/ (Legacy - can move to models/)
│   ├── metadata.yaml
│   ├── model_ncnn.py
│   └── model.ncnn.param
│
├── 🗂️  yolo11n-pose_openvino_model/ (Legacy - can move to models/)
│   ├── metadata.yaml
│   └── yolo11n-pose.xml
│
├── 🗂️  yolopose/ (Legacy - old code)
│   ├── Quantization/
│   └── YOLOPOSE/
│
├── 🗂️  utils/ (Empty, ready for helper functions)
│
├── 📹 JumpingJacks.mp4 (Sample video)
│
└── ⚠️  Old .py files (can delete after testing)
    ├── cam_angle_jumping_jack_counter.py
    ├── distance_jumping_jack_counter.py
    ├── vdo_angle_jumping_jack_counter.py
    └── test.py
```

---

## 🔄 Data Flow Diagram

### MediaPipe Angle Counter
```
┌─────────────┐
│   Webcam    │
│   Stream    │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│ mediapipe_angle_counter.py   │
│ ┌────────────────────────────┤
│ │ 1. Read frame               │
│ │ 2. Detect pose landmarks    │
│ │ 3. Calculate angles         │
│ │ 4. Check thresholds         │
│ │ 5. Update counter           │
│ │ 6. Draw skeleton            │
│ └────────────────────────────┤
└──────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Display    │
│  GUI        │
│  Count      │
└─────────────┘
```

### YOLO Pose Counter (Video Processing)
```
┌──────────────────┐
│   Video File     │
│  (models/video)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│   yolo_pose_counter.py       │
│ ┌────────────────────────────┤
│ │ 1. Load YOLO model          │
│ │ 2. Read video frames        │
│ │ 3. Detect poses (YOLO)      │
│ │ 4. Calculate metrics        │
│ │ 5. Update counter           │
│ │ 6. Log frame data           │
│ └────────────────────────────┤
└──────────────────────────────┘
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
┌──────────────────────┐    ┌─────────────────────┐
│  Output Videos       │    │   Analysis Files    │
│ (outputs/)           │    │ (outputs/)          │
│ ├─ overlay.mp4       │    │ ├─ per_frame_log    │
│ ├─ keypoints.mp4     │    │ ├─ signals_plot.png │
│ └─ side_by_side.mp4  │    │ └─ .csv log file    │
└──────────────────────┘    └─────────────────────┘
```

---

## 📋 Model File Organization

### MediaPipe Models
```
models/
└── pose_landmarker_lite.task
    └── Used by:
        - mediapipe_angle_counter.py
        - mediapipe_distance_counter.py
```

### YOLO Models
```
models/
├── yolo11n-pose.pt              (PyTorch - primary)
├── yolo11n-pose.onnx            (ONNX format)
├── yolo11n-pose.torchscript     (TorchScript format)
├── yolo11n-pose.mnn             (MNN format)
│
├── yolo11n-pose_ncnn_model/     (Converted NCNN)
│   ├── metadata.yaml
│   ├── model_ncnn.py
│   └── model.ncnn.param
│
└── yolo11n-pose_openvino_model/ (Converted OpenVINO)
    ├── metadata.yaml
    └── yolo11n-pose.xml
```

All used by: `yolo_pose_counter.py`

---

## 🎯 Quick Access Map

### To run MediaPipe angle counter:
```
1. Read:   QUICKSTART.md
2. Run:    python src/mediapipe_angle_counter.py
3. Adjust: ARM_STRAIGHT_THRESH, etc.
```

### To run MediaPipe distance counter:
```
1. Read:   QUICKSTART.md
2. Run:    python src/mediapipe_distance_counter.py
3. Adjust: FEET_DIST_OPEN, HAND_HIP_DIST_UP, etc.
```

### To run YOLO video processor:
```
1. Read:   README.md (Approaches section)
2. Edit:   src/yolo_pose_counter.py (set VIDEO_PATH)
3. Run:    python src/yolo_pose_counter.py
4. Check:  outputs/ folder for results
```

---

## 🔧 Configuration Files Location

| Config | File | Location |
|--------|------|----------|
| MediaPipe angle settings | Lines 22-34 | `src/mediapipe_angle_counter.py` |
| MediaPipe distance settings | Lines 21-26 | `src/mediapipe_distance_counter.py` |
| YOLO settings | Lines 27-32 | `src/yolo_pose_counter.py` |
| Dependencies | All | `requirements.txt` |
| Git ignore | All | `.gitignore` |

---

## 📊 Dependencies Visualization

```
Python 3.8+
│
├── opencv-python (4.8+)
│   └── Used by: All counter scripts
│
├── mediapipe (0.10+)
│   └── Used by: mediapipe_*_counter.py scripts
│
├── ultralytics (8.0+)
│   └── Used by: yolo_pose_counter.py
│
├── torch (1.9+)
│   └── Used by: YOLO, GPU support
│
├── numpy (1.21+)
│   └── Used by: All scripts for calculations
│
├── pandas (1.3+)
│   └── Used by: YOLO counter for logging
│
└── matplotlib (3.5+)
    └── Used by: YOLO counter for plotting
```

Install all with: `pip install -r requirements.txt`

---

## 🎬 Script Execution Comparison

| Feature | Angle | Distance | YOLO |
|---------|-------|----------|------|
| Input | Webcam | Webcam | Video file |
| Processing | Real-time | Real-time | Batch |
| Method | Angles | Distances | Deep Learning |
| Output | GUI | GUI | Videos + CSV |
| Model | MediaPipe | MediaPipe | YOLO11 |
| Speed | Fast | Fast | Very Fast |
| Accuracy | High | Medium | High |
| Best For | Validation | Testing | Analysis |

---

*Repository Map • January 29, 2026*
