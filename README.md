# MediaPipe - Jumping Jacks Counter

This project uses computer vision to count jumping jacks in videos. It leverages the YOLOv11-pose model for accurate pose estimation and provides a robust counting mechanism.

## Features

- **Jumping Jack Counting**: Automatically counts jumping jacks from a video file.
- **Pose Visualization**: Generates output videos with the detected pose skeleton overlaid.
- **Detailed Analysis**: Creates a frame-by-frame CSV log of the detection scores and states.
- **Signal Plotting**: Visualizes the "open" and "closed" state scores over time.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Poetry](https) for dependency management (recommended)
- FFmpeg (for video processing)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/MediaPipe-JackpingJacks-Counter.git
    cd MediaPipe-JackpingJacks-Counter
    ```

2.  **Create a virtual environment:**

    We recommend using a virtual environment to keep dependencies isolated.

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    Install the required Python packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Place your video:**

    Put the video file you want to process into the `data/` directory.

2.  **Update the configuration:**

    Open the `src/yolo_pose_counter.py` file and update the `VIDEO_PATH` variable to point to your video file.

    ```python
    # src/yolo_pose_counter.py

    # --- CONFIGURATION ---
    VIDEO_PATH = "data/your_video.mp4"  # <-- Change this
    MODEL_PATH = "models/pt_models/yolo11n-pose.pt"
    ...
    ```

3.  **Run the script:**

    Execute the main script from the root of the project.

    ```bash
    python src/yolo_pose_counter.py
    ```

4.  **Check the results:**

    The output files will be saved in the `outputs/` directory, including:
    - `overlay.mp4`: The original video with the pose skeleton overlaid.
    - `keypoints_only.mp4`: A video showing only the animated skeleton.
    - `side_by_side.mp4`: A combined view of the skeleton and the overlay.
    - `per_frame_log.csv`: A detailed log of the analysis for each frame.
    - `signals_plot.png`: A graph showing the detection scores over time.

## Project Structure

```
.
├── data/                  # Input videos and ground truth data
├── models/                # Trained model files (.pt, .tflite)
│   ├── pt_models/
│   └── tflite_models/
├── optimization/          # Scripts for model conversion and optimization
├── outputs/               # Generated output files (videos, logs, plots) - gitignored
├── src/                   # Main source code
│   └── yolo_pose_counter.py # Core jumping jack counting script
├── .gitignore             # Files and directories to be ignored by Git
├── README.md              # This file
└── requirements.txt       # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
