# 🍓 Raspberry Pi Setup - Step by Step

Complete guide for setting up Jumping Jacks Counter on Raspberry Pi.

---

## Phase 1: Hardware Setup (5 minutes)

### 1.1 What You Need
- Raspberry Pi 4B (4GB+) or Pi 5
- Power supply (5V/3A minimum)
- microSD card (32GB+)
- USB Webcam or CSI Camera
- Ethernet cable or WiFi

### 1.2 Flash Raspberry Pi OS

1. Download **Raspberry Pi Imager** from raspberrypi.org
2. Insert microSD card into computer
3. Open Imager:
   - Choose OS: **Raspberry Pi OS (64-bit)**
   - Choose Storage: Your microSD
   - Click Gear Icon for settings:
     - ✅ Set hostname: `jumpjack-pi`
     - ✅ Enable SSH (password auth)
     - ✅ Set username/password
     - ✅ Configure WiFi (if needed)
4. Click Write and wait (~10 minutes)

### 1.3 Connect Raspberry Pi

1. Insert microSD into Pi
2. Connect camera (if CSI)
3. Connect power
4. Wait 2-3 minutes for boot

---

## Phase 2: Initial Setup (10 minutes)

### 2.1 SSH into Your Pi

```bash
# From your computer
ssh pi@jumpjack-pi.local

# Enter password you set in Imager
```

### 2.2 Update System

```bash
# Update package lists
sudo apt update

# Upgrade all packages
sudo apt upgrade -y

# Remove unnecessary packages
sudo apt autoremove -y
```

### 2.3 Enable Camera (if using CSI)

```bash
# Open configuration
sudo raspi-config

# Navigation: Interface Options > Camera > Enable > Finish
# Reboot
sudo reboot
```

Wait for reboot and SSH back in.

---

## Phase 3: Project Setup (10 minutes)

### 3.1 Clone/Copy Project

```bash
# Clone from GitHub (if available)
git clone https://github.com/yourname/MediaPipe-JackingJacks-Counter.git
cd MediaPipe-JackingJacks-Counter

# OR manually copy your files to /home/pi/
```

### 3.2 Install Python & Dependencies

```bash
# Check Python version
python3 --version  # Should be 3.9+

# Install build tools
sudo apt install -y \
    build-essential \
    python3-pip \
    python3-venv \
    git \
    cmake

# Install media libraries
sudo apt install -y \
    libatlas-base-dev \
    libjasper-dev \
    libtiff5 \
    libjasper1 \
    libharfbuzz0b \
    libwebp6
```

### 3.3 Create Virtual Environment

```bash
# Navigate to project
cd ~/MediaPipe-JackingJacks-Counter

# Create venv
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3.4 Install Project Dependencies

```bash
# Install optimized for RPi
pip install --no-cache-dir \
    opencv-python==4.8.0.74 \
    mediapipe==0.10.9 \
    numpy==1.24.3

# Or use requirements file (if available)
pip install -r requirements-rpi.txt
```

⏱️ This takes 5-10 minutes on RPi 4

---

## Phase 4: Test Camera (5 minutes)

### 4.1 Test USB Camera

```bash
# List cameras
ls /dev/video*

# Simple test
python3 << 'EOF'
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    print("✅ Camera works! Resolution:", frame.shape)
else:
    print("❌ Camera not found. Try VideoCapture(1) or check connection")
EOF
```

### 4.2 Test CSI Camera

```bash
# Quick preview
libcamera-hello --duration 3000

# Should show camera preview for 3 seconds
```

### 4.3 Adjust Camera Index if Needed

In `src/mediapipe_angle_counter.py`, change:
```python
cap = cv2.VideoCapture(0)  # Try 0, 1, or 2
```

---

## Phase 5: Download Models (5 minutes)

### 5.1 MediaPipe Model

```bash
# Download and place in models/
mkdir -p models
cd models

# Download lite model (faster)
wget https://storage.googleapis.com/mediapipe-models/vision_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

cd ..
```

### 5.2 Verify Model

```bash
# Check if file exists and has size
ls -lh models/pose_landmarker_lite.task

# Should show file size ~6-7 MB
```

---

## Phase 6: First Run (3 minutes)

### 6.1 Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 6.2 Run the Script

```bash
# Run MediaPipe angle counter
python src/mediapipe_angle_counter.py
```

### Expected Output
```
Starting Jumping Jacks Counter...
Press 'q' to quit
--------------------------------------------------
(Display should show live video with skeleton)
```

### 6.3 Test It

1. Stand in front of camera
2. Do a jumping jack
3. You should see count increase
4. Press `q` to quit

**Success! 🎉**

---

## Phase 7: Optimize Performance (Optional, 10 minutes)

### 7.1 Check Resources

```bash
# Monitor while running in another terminal
watch -n 1 'free -h && echo "---" && top -bn1 | head -20'
```

### 7.2 Reduce Resolution (if slow)

Edit `src/mediapipe_angle_counter.py`:
```python
# After cv2.flip(frame, 1):
frame = cv2.resize(frame, (640, 480))  # Reduce resolution
```

### 7.3 Skip Frames (if still slow)

```python
frame_count = 0
while cap.isOpened():
    frame_count += 1
    
    if frame_count % 2 == 0:  # Process every 2nd frame
        # Process frame
```

### 7.4 Disable Display (for headless operation)

Comment out display lines:
```python
# cv2.imshow('Jumping Jack Counter', image)  # Comment this
# if cv2.waitKey(5) & 0xFF == ord('q'): break  # And this
```

---

## Phase 8: Setup Auto-Start (Optional, 5 minutes)

### 8.1 Create Service File

```bash
sudo nano /etc/systemd/system/jumpjack.service
```

Paste this:
```ini
[Unit]
Description=Jumping Jacks Counter
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/MediaPipe-JackingJacks-Counter
Environment="PATH=/home/pi/MediaPipe-JackingJacks-Counter/.venv/bin"
ExecStart=/home/pi/MediaPipe-JackingJacks-Counter/.venv/bin/python src/mediapipe_angle_counter.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Press `Ctrl+X`, then `Y`, then `Enter`

### 8.2 Enable Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable (auto-start on boot)
sudo systemctl enable jumpjack.service

# Start service
sudo systemctl start jumpjack.service

# Check status
sudo systemctl status jumpjack.service

# View logs
sudo journalctl -u jumpjack.service -f
```

### 8.3 Stop Service

```bash
sudo systemctl stop jumpjack.service
sudo systemctl disable jumpjack.service
```

---

## 🆘 Troubleshooting

### Problem: "No module named 'cv2'"

```bash
# Reactivate venv and reinstall
source .venv/bin/activate
pip install --upgrade opencv-python
```

### Problem: Camera shows black/no video

```bash
# Try different camera index
# Edit script and try VideoCapture(1) or VideoCapture(2)

# Disable GPU if using CSI
export LIBCAMERA_LOG_LEVELS=*:DEBUG
libcamera-hello
```

### Problem: Script runs but very slow (1-2 FPS)

```bash
# Reduce resolution in script (see Phase 7.2)
# Or skip frames (see Phase 7.3)
# Or disable display (see Phase 7.4)
```

### Problem: SSH drops/slow connection

```bash
# Use screen to keep session alive
screen

# Run script inside screen
python src/mediapipe_angle_counter.py

# Detach: Ctrl+A, then D
# Reattach: screen -r
```

### Problem: Out of memory errors

```bash
# Kill unnecessary services
sudo systemctl stop cups
sudo systemctl stop avahi-daemon

# Check memory
free -h

# Monitor during run
watch -n 1 free -h
```

---

## ✅ Verification Checklist

- [ ] Pi boots successfully
- [ ] SSH connection works
- [ ] Camera works (tested)
- [ ] Model file downloaded
- [ ] Virtual env created
- [ ] Dependencies installed
- [ ] Script runs without errors
- [ ] Jumping jack counted correctly
- [ ] (Optional) Service auto-starts

---

## 🚀 Next Steps

1. **Optimize:** Follow Phase 7 if performance needs improvement
2. **Deploy:** Use Phase 8 to auto-start on boot
3. **Monitor:** Set up logging to track usage
4. **Extend:** Add cloud integration, web dashboard, etc.

---

## 📞 Quick Commands Reference

```bash
# Activate environment
source .venv/bin/activate

# Run script
python src/mediapipe_angle_counter.py

# Check resources
free -h && top -bn1 | head -20

# Service commands
sudo systemctl status jumpjack.service
sudo systemctl restart jumpjack.service
sudo journalctl -u jumpjack.service -f

# Camera test
libcamera-hello --duration 3000

# Reboot
sudo reboot

# SSH
ssh pi@jumpjack-pi.local
```

---

**🍓 You're all set! Enjoy your IoT jumping jacks counter!**
