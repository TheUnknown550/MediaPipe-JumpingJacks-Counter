# 🍓 Raspberry Pi Quick Start Card

## Get Running in 45 Minutes

### What You Need
```
• Raspberry Pi 4B or 5 (with 4GB+ RAM)
• Power supply (5V/3A)
• USB camera or CSI camera
• microSD card (32GB+)
• Computer with SSH client
```

---

## 5-Minute Setup Overview

```
1. Flash Raspberry Pi OS (Imager)
2. SSH in and update system
3. Clone project and create venv
4. Install requirements-rpi.txt
5. Download model file
6. Run script
7. Test with jumping jack
```

---

## Command Cheat Sheet

### SSH Connection
```bash
ssh pi@raspberrypi.local
```

### Project Setup
```bash
# Clone (if from GitHub)
git clone <url>
cd MediaPipe-JackingJacks-Counter

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements-rpi.txt
```

### Download Model
```bash
mkdir -p models
cd models
wget https://storage.googleapis.com/mediapipe-models/vision_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
cd ..
```

### Run
```bash
# Activate venv
source .venv/bin/activate

# Run counter
python src/mediapipe_angle_counter.py

# Press 'q' to quit
```

### Setup Auto-Start
```bash
# Create service (see IOT_DEPLOYMENT_GUIDE.md)
sudo nano /etc/systemd/system/jumpjack.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable jumpjack
sudo systemctl start jumpjack

# Check status
sudo systemctl status jumpjack
```

---

## File Locations on RPi

```
/home/pi/
├── MediaPipe-JackingJacks-Counter/
│   ├── src/
│   │   ├── mediapipe_angle_counter.py
│   │   └── ...
│   ├── models/
│   │   └── pose_landmarker_lite.task
│   ├── data/
│   ├── .venv/
│   ├── requirements-rpi.txt
│   └── ...
```

---

## Performance Check

```bash
# While running in another terminal
watch -n 1 'free -h && echo "---" && top -bn1 | head -10'
```

Expected output (MediaPipe):
```
CPU: 30-40%
Memory: 250-350 MB
FPS: 15-20
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No camera | Check CSI connection or try `VideoCapture(1)` |
| Too slow | Reduce resolution or skip frames |
| Out of memory | Stop other services: `sudo systemctl stop cups` |
| Model not found | Download to `models/` folder |
| Module errors | Reinstall: `pip install -r requirements-rpi.txt` |

---

## Documentation Files

| File | Purpose |
|------|---------|
| IOT_PROJECT_OVERVIEW.md | Big picture overview |
| RASPBERRY_PI_SETUP.md | **START HERE** - Full setup |
| IOT_DEPLOYMENT_GUIDE.md | Advanced optimization |
| IOT_ENHANCEMENTS_SUMMARY.md | What's included |

---

## Performance Specs

```
Raspberry Pi 4B (8GB):
  ✅ MediaPipe: 15-20 FPS, 30-40% CPU
  ✅ With Edge TPU: 25-30 FPS, 20-30% CPU

Raspberry Pi 5 (8GB):
  ✅ MediaPipe: 20-30 FPS, 20-30% CPU
  ✅ With Edge TPU: 30+ FPS, 10-20% CPU
```

---

## Quick Options

### Option 1: Test on Laptop First
```bash
pip install -r requirements.txt
python src/mediapipe_angle_counter.py
# Test everything works
```

### Option 2: Straight to RPi
```bash
# Follow RASPBERRY_PI_SETUP.md exactly
# Use requirements-rpi.txt
```

### Option 3: With Edge TPU
```bash
# Install Coral runtime
# Use YOLO model with Edge TPU acceleration
# See IOT_DEPLOYMENT_GUIDE.md for details
```

---

## Remote Access

### SSH with Port Forwarding
```bash
# If behind firewall, use:
ssh -R 8888:localhost:22 user@your-server.com

# Or static IP + dynamic DNS
```

### Send Results to Cloud
```python
# Add to your script:
import requests

def send_data(count):
    requests.post('https://your-api.com/api/count',
                  json={'count': count, 'timestamp': now})
```

---

## Helpful Commands

```bash
# Check system
uname -a
vcgencmd measure_temp
free -h

# Monitor
htop
top

# Check storage
df -h
du -sh ~/

# Update
sudo apt update && sudo apt upgrade -y

# Logs
journalctl -u jumpjack.service -f
tail -f ~/jumpjack.log
```

---

## Cost Breakdown

```
Equipment:
  Raspberry Pi 4B (8GB)     ~$60-75
  USB Webcam              ~$20-40
  Power Supply            ~$10-15
  microSD Card (64GB)     ~$10-15
  ─────────────────────────────
  Total Basic             ~$100-145

Optional:
  CSI Camera Module       ~$30-50
  USB 3.0 SSD            ~$30-60
  Edge TPU (Coral)       ~$40-60
  Cooling Case           ~$10-20
```

---

## Success Checklist

- [ ] RPi OS installed and updated
- [ ] Python 3.9+ verified
- [ ] Virtual environment created
- [ ] requirements-rpi.txt installed
- [ ] Model file downloaded
- [ ] Camera tested
- [ ] Script runs
- [ ] Count increases correctly
- [ ] FPS is 15+ (smooth)
- [ ] CPU below 50%

---

## Need Help?

1. **Setup issues** → See RASPBERRY_PI_SETUP.md
2. **Performance** → See IOT_DEPLOYMENT_GUIDE.md
3. **Concepts** → See IOT_PROJECT_OVERVIEW.md
4. **Quick questions** → This card!

---

## 🚀 Ready?

**Next Step:** Read `RASPBERRY_PI_SETUP.md` (Complete guide)

**Time: 30-45 minutes to working system**

---

*🍓 Jumping Jacks Counter on Raspberry Pi - IoT Made Easy!*
