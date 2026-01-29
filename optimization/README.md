# Model Optimization & Quantization

This directory contains tools and scripts for optimizing and quantizing the YOLO11n-Pose model for edge deployment.

## Overview

Quantization reduces model size and increases inference speed at the cost of slight accuracy loss. This is critical for Raspberry Pi and IoT devices.

## Files

### evaluate-model.py
Benchmarks and compares different model formats (PT, NCNN, TFLite) for:
- Model size (MB)
- Inference speed (FPS)
- Number of parameters
- Accuracy metrics

**Usage:**
```bash
python evaluate-model.py
```

### model-convert.py
Exports the original YOLO11n-Pose model to optimized formats:
- **NCNN (FP16)** - Best for Raspberry Pi 4 CPU
- **TFLite (INT8)** - Best for edge TPU (Coral)

**Usage:**
```bash
python model-convert.py
```

## Quantization Methods

| Format | Size | Speed | Accuracy | Best For |
|--------|------|-------|----------|----------|
| **PT (original)** | ~25MB | Baseline | 100% | Reference |
| **NCNN (FP16)** | ~12MB | 1.5-2x faster | 98-99% | RPi 4 CPU |
| **TFLite (INT8)** | ~6MB | 2-3x faster | 95-98% | RPi with Edge TPU |

## Getting Started

1. **Evaluate current models:**
   ```bash
   python evaluate-model.py
   ```

2. **Convert model to new formats:**
   ```bash
   python model-convert.py
   ```

3. **Choose the best format** based on your deployment hardware

## Notes

- INT8 quantization requires calibration data (sample images)
- FP16 quantization maintains better accuracy
- Always test on target hardware before production deployment

