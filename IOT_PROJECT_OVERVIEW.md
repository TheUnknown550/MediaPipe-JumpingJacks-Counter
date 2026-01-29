# 🍓 IoT Jumping Jacks Counter - Project Overview

## What Is This?

A **computer vision jumping jacks counter** designed specifically for **Raspberry Pi and IoT devices**. It uses lightweight AI models to detect body pose and count exercise repetitions in real-time.

---

## 🎯 Use Cases

### Fitness & Health
- Home gym workout counter
- Physical therapy progress tracking
- Group exercise monitoring
- Fitness app backend

### Education
- Computer vision learning project
- IoT edge computing demo
- AI on embedded systems showcase
- Pose estimation applications

### Industrial/Healthcare
- Physical therapy clinics
- Rehabilitation centers
- Fitness centers (smart equipment)
- Elderly care facilities

---

## 💡 Why Raspberry Pi?

### Cost-Effective
```
💰 RPi 4B (8GB): ~$60-75
🎥 USB Camera: ~$20-40
📷 CSI Camera: ~$30-50
Total: ~$100-150
```

### Power Efficient
```
⚡ RPi 4B: ~5W (normal) - ~10W (full load)
☀️ Can run on solar power
🔋 Portable deployment
💡 Low operational cost
```

### Community & Resources
```
📚 Huge community support
📖 Lots of tutorials
🔧 Easy to troubleshoot
🚀 Easy to extend
```

### Perfect for Edge Computing
```
🚀 Process data locally (privacy)
⚡ Real-time processing
🔌 No cloud dependency
📡 Optional cloud sync
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│      Raspberry Pi 4/5                   │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   Camera (USB or CSI)            │  │
│  │   ├─ Real-time video stream      │  │
│  │   └─ 30-60 FPS @ 1080p           │  │
│  └──────────────────────────────────┘  │
│              │                          │
│              ▼                          │
│  ┌──────────────────────────────────┐  │
│  │   Python Application             │  │
│  │   ├─ mediapipe_angle_counter.py  │  │
│  │   ├─ medapipe_distance_counter   │  │
│  │   └─ yolo_pose_counter.py        │  │
│  └──────────────────────────────────┘  │
│              │                          │
│              ├─► Local Display (optional)
│              ├─► CSV Logging
│              ├─► Cloud API (optional)
│              └─► Web Dashboard (optional)
└─────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Real-Time Processing
```
Frame Capture
    ↓
Pose Detection (MediaPipe/YOLO)
    ↓
Calculate Angles/Distances
    ↓
Check Thresholds
    ↓
Update Counter & State
    ↓
Log Data → Display / API / File
```

### Typical Latency
```
MediaPipe: 30-70ms (RPi 4)
YOLO: 80-200ms (RPi 4)
With Coral TPU: 30-40ms (any RPi)
```

---

## 📊 System Specifications

### Minimum Requirements
```
Hardware:
  • Raspberry Pi 4B (4GB RAM)
  • Power supply (5V/3A)
  • USB webcam or CSI camera
  • 32GB microSD card

Software:
  • Raspberry Pi OS (64-bit)
  • Python 3.9+
  • MediaPipe 0.10.9
  • OpenCV 4.8.0
```

### Recommended Setup
```
Hardware:
  • Raspberry Pi 5 (8GB RAM)
  • Good power supply (5V/4A)
  • CSI Camera Module 3
  • 64GB or larger SSD via USB 3.0
  • (Optional) Google Coral TPU

Software:
  • Raspberry Pi OS (latest)
  • Python 3.10 or 3.11
  • All dependencies
  • Systemd service for auto-start
```

---

## 🚀 Deployment Modes

### Mode 1: Interactive (Development)
```bash
python src/mediapipe_angle_counter.py
```
- Live display on screen
- Real-time adjustments
- Good for testing/tuning

### Mode 2: Headless (Production)
```bash
# Run in background
python src/mediapipe_angle_counter.py &

# Or as systemd service
sudo systemctl start jumpjack
```
- No display needed
- Save system resources
- Perfect for continuous operation

### Mode 3: Remote Monitoring
```python
# Send data to cloud/API
requests.post('https://your-api.com/data', 
              json={'count': counter, 'timestamp': now})
```
- Real-time monitoring from anywhere
- Dashboard integration
- Historical data analysis

---

## 📈 Performance Comparison

### MediaPipe vs YOLO on Raspberry Pi 4

```
┌─────────────────┬──────────────┬──────────────┐
│ Metric          │ MediaPipe    │ YOLO         │
├─────────────────┼──────────────┼──────────────┤
│ FPS             │ 15-20 ✅     │ 2-5 ⚠️       │
│ Latency         │ 50-70ms ✅   │ 200-500ms ⚠️ │
│ CPU Usage       │ 30-40% ✅    │ 95-100% ❌   │
│ Memory          │ 250MB ✅     │ 1GB+ ❌      │
│ Accuracy        │ 90% ✅       │ 95% ✅       │
│ Model Size      │ 7MB ✅       │ 250MB+ ⚠️    │
└─────────────────┴──────────────┴──────────────┘
```

### With Edge TPU Acceleration
```
YOLO + Coral TPU:
  FPS: 25-30 ✅ (vs 2-5)
  Latency: 30-40ms ✅ (vs 200-500ms)
  CPU: 20-30% ✅ (vs 95-100%)
  Cost: +$40-60
```

---

## 🛠️ Technology Stack

### Hardware
- **Processor:** ARM Cortex-A72 (RPi 4) / ARM Cortex-A78 (RPi 5)
- **Memory:** 4GB - 8GB LPDDR4
- **Storage:** microSD or USB SSD
- **Camera:** USB WebCam or CSI Camera Module
- **Optional:** Google Coral Edge TPU

### Software
- **OS:** Raspberry Pi OS (64-bit)
- **Language:** Python 3.9+
- **Core Libraries:**
  - OpenCV (computer vision)
  - MediaPipe (pose detection)
  - Ultralytics YOLO (object detection)
  - NumPy (numerical computing)

### Deployment
- **Service Management:** systemd
- **Cloud Integration:** REST API, MQTT (optional)
- **Monitoring:** Local logging, remote dashboards (optional)
- **Version Control:** Git

---

## 📋 Installation Summary

### Desktop/Laptop (for testing)
```bash
# 1. Clone repository
git clone <repo-url>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run script
python src/mediapipe_angle_counter.py
```

### Raspberry Pi (production)
```bash
# 1. Flash Raspberry Pi OS

# 2. Follow RASPBERRY_PI_SETUP.md

# 3. Install RPi-optimized dependencies
pip install -r requirements-rpi.txt

# 4. Set up as systemd service

# 5. Access remotely via API/logs
```

---

## 🔐 Security Considerations

### Local Operation
```
✅ All data processed locally
✅ No internet required for operation
✅ Full data privacy
❌ Cannot access remotely (by default)
```

### Optional Cloud Integration
```
⚠️ Secure the API endpoint
⚠️ Use HTTPS/TLS encryption
⚠️ Implement authentication
⚠️ Be aware of privacy regulations
```

### Best Practices
```
1. Keep Raspberry Pi OS updated
2. Use strong SSH passwords or keys
3. Disable SSH when not needed
4. Run as non-root user
5. Monitor logs for errors
6. Regular backups of configuration
```

---

## 💼 Real-World Applications

### Fitness Center
```
Setup: Install on Pi connected to gym equipment
Display: Show count on TV or app
Features: Sync with user accounts, track progress
Cloud: Save results for user access
```

### Physical Therapy
```
Setup: Clinic workstation
Display: Real-time form feedback
Logging: Detailed session logs
Compliance: Medical-grade data tracking
```

### Smart Home Gym
```
Setup: Home corner with Pi + camera
Integration: Connect to fitness app
Tracking: Personal progress dashboard
Motivation: Leaderboards, achievements
```

### Educational Demo
```
Setup: Classroom IoT demonstration
Features: Real-time ML on edge device
Teaching: Pose estimation, edge computing
Cost: Very affordable for schools
```

---

## 📊 Data Collection

### What's Tracked
```
Per Session:
  • Total repetitions
  • Time elapsed
  • Average FPS
  • Successful vs failed reps
  • Form quality (if angle-based)

Optional Tracking:
  • User identification
  • Historical progress
  • Device metrics (CPU, temp)
  • Network latency (if cloud-enabled)
```

### Storage Options
```
Local:
  • CSV files on RPi
  • Cloud-sync capable
  • ~1KB per repetition

Cloud (optional):
  • Remote dashboard
  • Multi-device tracking
  • Historical analysis
  • Share with coach/trainer
```

---

## 🎓 Learning Resources

### Included Documentation
- [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) - Step-by-step setup
- [IOT_DEPLOYMENT_GUIDE.md](IOT_DEPLOYMENT_GUIDE.md) - Full deployment guide
- [README.md](README.md) - Feature and configuration details
- [QUICKSTART.md](QUICKSTART.md) - Quick reference

### External Resources
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Raspberry Pi Official Guides](https://www.raspberrypi.org/documentation/)
- [Edge TPU Guides](https://coral.ai/docs/)

---

## 🚀 Getting Started

### Quick Path (Laptop First)
1. Read this document
2. Follow [QUICKSTART.md](QUICKSTART.md)
3. Run on your computer
4. Then deploy to RPi using [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)

### Direct RPi Path
1. Read this document
2. Follow [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)
3. Set up hardware
4. Install and run

---

## 📞 Support & Questions

- **Setup issues:** See [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)
- **How to run:** See [QUICKSTART.md](QUICKSTART.md)
- **Full docs:** See [README.md](README.md)
- **IoT details:** See [IOT_DEPLOYMENT_GUIDE.md](IOT_DEPLOYMENT_GUIDE.md)

---

## ✨ Key Takeaways

| Aspect | Benefit |
|--------|---------|
| **Cost** | ~$100-150 complete system |
| **Simplicity** | No coding required to use |
| **Performance** | 15-30 FPS real-time |
| **Privacy** | Process data locally |
| **Extensibility** | Easy to add features |
| **Reliability** | Proven technologies |
| **Community** | Large RPi community |
| **Scalability** | Run multiple Pis easily |

---

## 🎉 Ready to Deploy?

**Start here:** [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)

**Time required:** 30-45 minutes for complete setup

**Difficulty:** Beginner-friendly with detailed instructions

---

*🍓 Jumping Jacks Counter on Raspberry Pi - Professional IoT Project*

*Last updated: January 29, 2026*
