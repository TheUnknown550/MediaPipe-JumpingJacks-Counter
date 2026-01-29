"""
Model Evaluation Script for Jumping Jack Counter
=================================================
Evaluates YOLO11n-Pose models based on their ability to count jumping jacks correctly.

Metrics calculated:
- Accuracy: Percentage of correct predictions
- Precision: Of detected jumping jacks, how many were correct
- Recall (Sensitivity): Of actual jumping jacks, how many were detected
- True Positive Rate (TPR): TP / (TP + FN)
- False Positive Rate (FPR): FP / (FP + TN)
- Mean Absolute Error (MAE): Average difference between predicted and actual counts
"""

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Paths & configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_FOLDER = PROJECT_ROOT / "data"
GROUND_TRUTH_CSV = DATA_FOLDER / "ground_truth.csv"
EXPORT_FOLDER = DATA_FOLDER / "exported_videos"
MODELS_DIR = BASE_DIR / "models"
IMGSZ = 640
CONF_THRES = 0.25
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Define models to evaluate (PT + TFLite exports)
MODEL_PATHS = {
    "YOLO11n-Pose (PT)": BASE_DIR / "yolo11n-pose.pt",
    "YOLO11n-Pose Float16 (TFLite)": MODELS_DIR / "yolo11n-pose_float16.tflite",
    "YOLO11n-Pose Int8 (TFLite)": MODELS_DIR / "yolo11n-pose_int8.tflite",
}

# COCO-17 KEYPOINTS MAPPING (from yolo_pose_counter.py)
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


def dist(a, b):
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def clamp01(x):
    """Clamp value between 0.0 and 1.0."""
    return max(0.0, min(1.0, x))


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

class JumpingJackEvaluator:
    """Evaluate jumping jack counting models"""
    
    def __init__(self, ground_truth_csv):
        """Initialize evaluator with ground truth data"""
        self.ground_truth_csv = Path(ground_truth_csv)
        self.results = {}
        self.ground_truth = self._load_ground_truth()
        
        # Create export folder if it doesn't exist
        EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    def _load_ground_truth(self):
        """Load ground truth data from CSV"""
        if not Path(self.ground_truth_csv).exists():
            print(f"❌ Ground truth CSV not found at {self.ground_truth_csv}")
            return {}
        
        ground_truth = {}
        try:
            df = pd.read_csv(self.ground_truth_csv)
            for _, row in df.iterrows():
                video_name = row['video_name']
                correct_count = row['correct_count']
                ground_truth[video_name] = correct_count
            print(f"✅ Loaded {len(ground_truth)} ground truth labels")
            return ground_truth
        except Exception as e:
            print(f"❌ Error loading ground truth CSV: {e}")
            return {}
    
    def get_video_files(self):
        """Get all video files from data folder"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        videos = []
        
        if not DATA_FOLDER.exists():
            print(f"❌ Data folder not found at {DATA_FOLDER}")
            return videos
        
        for file in DATA_FOLDER.iterdir():
            if file.is_file() and any(file.name.lower().endswith(ext) for ext in video_extensions):
                videos.append(file.name)
        
        print(f"✅ Found {len(videos)} video files")
        return videos
    
    def evaluate_model(self, model_name, model_path):
        """Evaluate a single model on all test videos"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            return None
        
        try:
            # Load YOLO model
            model = YOLO(model_path)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
        
        videos = self.get_video_files()
        if not videos:
            print("❌ No videos found to evaluate")
            return None
        
        predictions = {}
        
        for video_file in videos:
            if video_file not in self.ground_truth:
                print(f"⚠️  Skipping {video_file} - no ground truth label found")
                continue
            
            video_path = DATA_FOLDER / video_file
            correct_count = self.ground_truth[video_file]
            
            print(f"\nProcessing: {video_file} (Expected: {correct_count} JJs)")
            
            # Count jumping jacks and export video
            predicted_count = self._count_jumping_jacks(video_path, model, model_name, video_file)
            predictions[video_file] = {
                "correct": correct_count,
                "predicted": predicted_count
            }
            
            print(f"  → Predicted: {predicted_count} JJs")
        
        if not predictions:
            print("❌ No predictions made")
            return None
        
        metrics = self._calculate_metrics(predictions)
        self.results[model_name] = {
            "predictions": predictions,
            "metrics": metrics
        }
        
        return metrics
    
    def _count_jumping_jacks(self, video_path, model, model_name, video_file):
        """
        Count jumping jacks using YOLO pose detection and export annotated video.
        Implements the same logic as yolo_pose_counter.py
        """
        print(f"    Running inference on {video_path}...")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"    ❌ Could not open video")
                return 0
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Create output video writer
            model_name_safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            video_name_without_ext = Path(video_file).stem
            output_path = EXPORT_FOLDER / f"{video_name_without_ext}_{model_name_safe}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
            
            # Initialize counter
            counter = JumpingJackCounter()
            frame_id = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO inference
                result = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, device=DEVICE, verbose=False)[0]
                
                overlay = frame.copy()
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

                        # Draw keypoints
                        for i in range(len(kpts)):
                            cv2.circle(overlay, tuple(kpts[i].astype(int)), 5, (0, 0, 255), -1)
                
                # Update counter
                o, c, state, count = counter.update(open_s, closed_s)
                
                # Draw UI text
                cv2.putText(overlay, f"Count: {count}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(overlay, f"State: {state}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(overlay, f"Model: {model_name}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overlay, f"Frame: {frame_id}", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                out.write(overlay)
                frame_id += 1
            
            cap.release()
            out.release()
            
            print(f"    ✅ Exported video to: {output_path}")
            
            return counter.count
            
        except Exception as e:
            print(f"    ❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _calculate_metrics(self, predictions):
        """Calculate evaluation metrics"""
        metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "tpr": 0,
            "fpr": 0,
            "mae": 0,
            "rmse": 0,
            "samples": len(predictions)
        }
        
        correct_predictions = 0
        total_predicted = 0
        total_actual = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        errors = []
        
        for video_file, data in predictions.items():
            correct = data["correct"]
            predicted = data["predicted"]
            
            errors.append(abs(correct - predicted))
            
            # Accuracy: exact match
            if correct == predicted:
                correct_predictions += 1
            
            total_actual += correct
            total_predicted += predicted
            
            # For TP/FP/FN calculation (counting as individual jumping jacks)
            if predicted > 0:
                # True positives: min of actual and predicted
                true_positives += min(correct, predicted)
                # False positives: overprediction
                if predicted > correct:
                    false_positives += (predicted - correct)
            
            # False negatives: underprediction
            if predicted < correct:
                false_negatives += (correct - predicted)
        
        # Calculate metrics
        metrics["accuracy"] = (correct_predictions / len(predictions)) * 100 if predictions else 0
        metrics["mae"] = np.mean(errors) if errors else 0
        metrics["rmse"] = np.sqrt(np.mean(np.array(errors)**2)) if errors else 0
        
        # Precision: TP / (TP + FP)
        if (true_positives + false_positives) > 0:
            metrics["precision"] = (true_positives / (true_positives + false_positives)) * 100
        
        # Recall: TP / (TP + FN)
        if (true_positives + false_negatives) > 0:
            metrics["recall"] = (true_positives / (true_positives + false_negatives)) * 100
        
        # TPR (same as recall)
        metrics["tpr"] = metrics["recall"]
        
        # FPR: FP / (FP + TN) - for this use case, TN is approximated
        # Using false negatives as a proxy for actual negatives
        if (false_positives + false_negatives) > 0:
            metrics["fpr"] = (false_positives / (false_positives + false_negatives)) * 100
        
        return metrics
    
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("❌ No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"{'EVALUATION RESULTS':^80}")
        print(f"{'='*80}\n")
        
        for model_name, data in self.results.items():
            metrics = data["metrics"]
            
            print(f"Model: {model_name}")
            print(f"{'-'*80}")
            print(f"  Samples Tested:        {metrics['samples']}")
            print(f"  Accuracy:              {metrics['accuracy']:.2f}%")
            print(f"  Precision:             {metrics['precision']:.2f}%")
            print(f"  Recall (Sensitivity):  {metrics['recall']:.2f}%")
            print(f"  True Positive Rate:    {metrics['tpr']:.2f}%")
            print(f"  False Positive Rate:   {metrics['fpr']:.2f}%")
            print(f"  Mean Absolute Error:   {metrics['mae']:.2f}")
            print(f"  RMSE:                  {metrics['rmse']:.2f}")
            print()
    
    def save_results_csv(self, output_file="evaluation_results.csv"):
        """Save detailed results to CSV"""
        output_path = Path(self.ground_truth_csv).parent / output_file
        
        rows = []
        for model_name, data in self.results.items():
            metrics = data["metrics"]
            for video_file, pred_data in data["predictions"].items():
                rows.append({
                    "model": model_name,
                    "video": video_file,
                    "correct_count": pred_data["correct"],
                    "predicted_count": pred_data["predicted"],
                    "error": abs(pred_data["correct"] - pred_data["predicted"])
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            print(f"✅ Results saved to {output_path}")
    
    def plot_results(self):
        """Create visualization of results"""
        if not self.results:
            print("❌ No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Jumping Jack Counter Model Evaluation', fontsize=16, fontweight='bold')
        
        model_names = list(self.results.keys())
        metrics_data = {name: data["metrics"] for name, data in self.results.items()}
        
        # Plot 1: Accuracy comparison
        accuracies = [metrics_data[name]["accuracy"] for name in model_names]
        axes[0, 0].bar(model_names, accuracies, color='skyblue')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylim([0, 100])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center')
        
        # Plot 2: Precision, Recall, TPR
        x = np.arange(len(model_names))
        width = 0.25
        precisions = [metrics_data[name]["precision"] for name in model_names]
        recalls = [metrics_data[name]["recall"] for name in model_names]
        tprs = [metrics_data[name]["tpr"] for name in model_names]
        
        axes[0, 1].bar(x - width, precisions, width, label='Precision', color='lightgreen')
        axes[0, 1].bar(x, recalls, width, label='Recall', color='lightcoral')
        axes[0, 1].bar(x + width, tprs, width, label='TPR', color='lightyellow')
        axes[0, 1].set_ylabel('Score (%)')
        axes[0, 1].set_title('Precision, Recall, and TPR Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 100])
        
        # Plot 3: MAE and RMSE
        maes = [metrics_data[name]["mae"] for name in model_names]
        rmses = [metrics_data[name]["rmse"] for name in model_names]
        
        axes[1, 0].bar(x - width/2, maes, width, label='MAE', color='orange')
        axes[1, 0].bar(x + width/2, rmses, width, label='RMSE', color='red')
        axes[1, 0].set_ylabel('Error (count)')
        axes[1, 0].set_title('Mean Absolute Error vs RMSE')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()
        
        # Plot 4: False Positive Rate
        fprs = [metrics_data[name]["fpr"] for name in model_names]
        axes[1, 1].bar(model_names, fprs, color='lightsteelblue')
        axes[1, 1].set_ylabel('False Positive Rate (%)')
        axes[1, 1].set_title('False Positive Rate Comparison')
        for i, v in enumerate(fprs):
            axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig(Path(self.ground_truth_csv).parent / 'evaluation_results.png', dpi=300)
        print("✅ Plot saved to evaluation_results.png")
        plt.show()


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("Jumping Jack Counter - Model Evaluation")
    print("="*60)
    print(f"Using device: {DEVICE} (torch.cuda.is_available()={torch.cuda.is_available()})")
    
    # Create evaluator
    evaluator = JumpingJackEvaluator(GROUND_TRUTH_CSV)
    
    # Evaluate each model
    for model_name, model_path in MODEL_PATHS.items():
        metrics = evaluator.evaluate_model(model_name, model_path)
    
    # Print and save results
    evaluator.print_results()
    evaluator.save_results_csv()
    evaluator.plot_results()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
