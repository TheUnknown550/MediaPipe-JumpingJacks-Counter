"""
Model Evaluation Script for Jumping Jack Counter
=================================================
Evaluates YOLO11n-Pose models based on their ability to count jumping jacks correctly.

Metrics calculated:
- Accuracy: Exact match on total count per video
- Precision/Recall/F1: Event-level across all videos
- FPS: Average inference frames per second
- Latency: Average per-frame inference time (ms)
- Size (MB): Model file size
- Params: Trainable parameters (if available)
- True Positive/False Positive/False Negative and confusion matrix
- Mean Absolute Error (MAE) & RMSE on counts
"""

import cv2
import torch
import numpy as np
import pandas as pd
import time
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
MODELS_DIR = PROJECT_ROOT / "models"
IMGSZ = 640
CONF_THRES = 0.25
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Define models to evaluate (PT + TFLite exports)
MODEL_PATHS = {
    "YOLO11n-Pose (PT)": PROJECT_ROOT / "optimization" / "yolo11n-pose.pt",
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
        total_frames = 0
        total_infer_time = 0.0
        
        for video_file in videos:
            if video_file not in self.ground_truth:
                print(f"⚠️  Skipping {video_file} - no ground truth label found")
                continue
            
            video_path = DATA_FOLDER / video_file
            correct_count = self.ground_truth[video_file]
            
            print(f"\nProcessing: {video_file} (Expected: {correct_count} JJs)")
            
            # Count jumping jacks and export video
            jj_stats = self._count_jumping_jacks(video_path, model, model_name, video_file)
            predicted_count = jj_stats["count"]
            total_frames += jj_stats["frames"]
            total_infer_time += jj_stats["infer_time"]
            predictions[video_file] = {
                "correct": correct_count,
                "predicted": predicted_count,
                "frames": jj_stats["frames"],
                "infer_time": jj_stats["infer_time"]
            }
            
            print(f"  → Predicted: {predicted_count} JJs")
        
        if not predictions:
            print("❌ No predictions made")
            return None
        
        metrics = self._calculate_metrics(
            predictions=predictions,
            total_frames=total_frames,
            total_infer_time=total_infer_time,
            model_path=model_path,
            model_obj=model,
        )
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
            
            # Initialize counter and timing
            counter = JumpingJackCounter()
            frame_id = 0
            total_infer_time = 0.0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO inference
                t0 = time.perf_counter()
                result = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, device=DEVICE, verbose=False)[0]
                total_infer_time += (time.perf_counter() - t0)
                
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
            
            return {
                "count": counter.count,
                "frames": frame_id,
                "infer_time": total_infer_time
            }
            
        except Exception as e:
            print(f"    ❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return {"count": 0, "frames": 0, "infer_time": 0.0}
    
    def _calculate_metrics(self, predictions, total_frames, total_infer_time, model_path, model_obj):
        """Calculate evaluation metrics and performance stats"""
        metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "tpr": 0,
            "fpr": 0,
            "mae": 0,
            "rmse": 0,
            "samples": len(predictions),
            "fps": 0,
            "latency_ms": 0,
            "size_mb": model_path.stat().st_size / 1e6 if model_path.exists() else 0,
            "params": None,
            "map": np.nan,
            "map50_95": np.nan,
            "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }
        
        correct_predictions = 0
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
        
        # F1 score
        if (metrics["precision"] + metrics["recall"]) > 0:
            p = metrics["precision"] / 100
            r = metrics["recall"] / 100
            metrics["f1"] = (2 * p * r / (p + r)) * 100
        
        # TPR (same as recall)
        metrics["tpr"] = metrics["recall"]
        
        # FPR: FP / (FP + TN) - for this use case, TN is approximated
        # Using false negatives as a proxy for actual negatives
        if (false_positives + false_negatives) > 0:
            metrics["fpr"] = (false_positives / (false_positives + false_negatives)) * 100

        # Performance stats
        if total_frames > 0 and total_infer_time > 0:
            metrics["fps"] = total_frames / total_infer_time
            metrics["latency_ms"] = (total_infer_time / total_frames) * 1000

        # Params (if available)
        try:
            if hasattr(model_obj, "model") and hasattr(model_obj.model, "parameters"):
                metrics["params"] = sum(p.numel() for p in model_obj.model.parameters())
        except Exception:
            metrics["params"] = None

        # Confusion matrix counts
        metrics["confusion"] = {
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "tn": 0  # TN not well-defined for count task
        }
        
        return metrics
    
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("❌ No results to display")
            return
        
        print(f"\n{'='*120}")
        print(f"{'COMPREHENSIVE MODEL COMPARISON':^120}")
        print(f"{'='*120}")
        header = (
            f"{'Model':<32}"
            f"{'FPS':>8}"
            f"{'Size(MB)':>10}"
            f"{'Latency(ms)':>13}"
            f"{'Params':>12}"
            f"{'mAP':>8}"
            f"{'mAP50-95':>10}"
            f"{'Prec%':>8}"
            f"{'Rec%':>8}"
            f"{'F1%':>8}"
            f"{'MAE':>7}"
            f"{'RMSE':>7}"
        )
        print(header)
        print("-" * len(header))

        for model_name, data in self.results.items():
            m = data["metrics"]
            params_m = (m["params"] / 1e6) if m["params"] else 0
            print(
                f"{model_name:<32}"
                f"{m['fps']:>8.1f}"
                f"{m['size_mb']:>10.2f}"
                f"{m['latency_ms']:>13.2f}"
                f"{params_m:>12.2f}"
                f"{(m['map'] if not np.isnan(m['map']) else 0):>8.1f}"
                f"{(m['map50_95'] if not np.isnan(m['map50_95']) else 0):>10.1f}"
                f"{m['precision']:>8.1f}"
                f"{m['recall']:>8.1f}"
                f"{m['f1']:>8.1f}"
                f"{m['mae']:>7.2f}"
                f"{m['rmse']:>7.2f}"
            )
            conf = m["confusion"]
            print(f"    Confusion Matrix (TP/FP/FN/TN): {conf['tp']}/{conf['fp']}/{conf['fn']}/{conf['tn']}")
        print("=" * len(header))
    
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
                    "error": abs(pred_data["correct"] - pred_data["predicted"]),
                    "frames": pred_data.get("frames", 0),
                    "infer_time_s": pred_data.get("infer_time", 0.0)
                })
            # summary row per model
            rows.append({
                "model": model_name,
                "video": "SUMMARY",
                "correct_count": "",
                "predicted_count": "",
                "error": "",
                "frames": "",
                "infer_time_s": "",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "fps": metrics["fps"],
                "latency_ms": metrics["latency_ms"],
                "size_mb": metrics["size_mb"],
                "params": metrics["params"],
                "tp": metrics["confusion"]["tp"],
                "fp": metrics["confusion"]["fp"],
                "fn": metrics["confusion"]["fn"],
                "tn": metrics["confusion"]["tn"],
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

        model_names = list(self.results.keys())
        metrics_data = {name: data["metrics"] for name, data in self.results.items()}
        x = np.arange(len(model_names))
        width = 0.25

        # Figure 1: Summary bars
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Jumping Jack Counter - Model Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy
        accuracies = [metrics_data[name]["accuracy"] for name in model_names]
        axes[0, 0].bar(model_names, accuracies, color='skyblue')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_ylim([0, 100])

        # Precision/Recall/F1
        precisions = [metrics_data[name]["precision"] for name in model_names]
        recalls = [metrics_data[name]["recall"] for name in model_names]
        f1s = [metrics_data[name]["f1"] for name in model_names]
        axes[0, 1].bar(x - width, precisions, width, label='Precision', color='lightgreen')
        axes[0, 1].bar(x, recalls, width, label='Recall', color='lightcoral')
        axes[0, 1].bar(x + width, f1s, width, label='F1', color='gold')
        axes[0, 1].set_ylabel('Score (%)')
        axes[0, 1].set_title('Precision / Recall / F1')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 100])

        # Errors
        maes = [metrics_data[name]["mae"] for name in model_names]
        rmses = [metrics_data[name]["rmse"] for name in model_names]
        axes[0, 2].bar(x - width/2, maes, width, label='MAE', color='orange')
        axes[0, 2].bar(x + width/2, rmses, width, label='RMSE', color='red')
        axes[0, 2].set_ylabel('Error (count)')
        axes[0, 2].set_title('MAE vs RMSE')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(model_names)
        axes[0, 2].legend()

        # FPS & Latency
        fps = [metrics_data[name]["fps"] for name in model_names]
        latency = [metrics_data[name]["latency_ms"] for name in model_names]
        axes[1, 0].bar(x - width/2, fps, width, label='FPS', color='steelblue')
        axes[1, 0].bar(x + width/2, latency, width, label='Latency (ms)', color='mediumseagreen')
        axes[1, 0].set_title('Throughput & Latency')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()

        # Model size
        sizes = [metrics_data[name]["size_mb"] for name in model_names]
        axes[1, 1].bar(model_names, sizes, color='plum')
        axes[1, 1].set_ylabel('Size (MB)')
        axes[1, 1].set_title('Model Size')

        # Params
        params = [((metrics_data[name]["params"] or 0) / 1e6) for name in model_names]
        axes[1, 2].bar(model_names, params, color='lightgray')
        axes[1, 2].set_ylabel('Parameters (M)')
        axes[1, 2].set_title('Parameter Count')

        plt.tight_layout()
        summary_path = Path(self.ground_truth_csv).parent / 'evaluation_results.png'
        plt.savefig(summary_path, dpi=300)
        print(f"✅ Summary plot saved to {summary_path}")

        # Figure 2: Confusion matrices
        fig_cm, ax_cm = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
        if len(model_names) == 1:
            ax_cm = [ax_cm]
        for idx, name in enumerate(model_names):
            conf = metrics_data[name]["confusion"]
            matrix = np.array([[conf["tp"], conf["fp"]], [conf["fn"], conf["tn"]]])
            im = ax_cm[idx].imshow(matrix, cmap='Blues')
            ax_cm[idx].set_title(f'Confusion: {name}')
            ax_cm[idx].set_xticks([0, 1])
            ax_cm[idx].set_yticks([0, 1])
            ax_cm[idx].set_xticklabels(['Pred Pos', 'Pred Neg'])
            ax_cm[idx].set_yticklabels(['Actual Pos', 'Actual Neg'])
            for (i, j), val in np.ndenumerate(matrix):
                ax_cm[idx].text(j, i, f"{val}", ha='center', va='center', color='black', fontsize=12)
        plt.tight_layout()
        cm_path = Path(self.ground_truth_csv).parent / 'confusion_matrices.png'
        plt.savefig(cm_path, dpi=300)
        print(f"✅ Confusion matrices saved to {cm_path}")
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
