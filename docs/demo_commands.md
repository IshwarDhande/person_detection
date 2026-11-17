# Live Demonstration Commands

## Quick Start (Interactive Laptop Webcam)
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py
```
> *Press `q` or `Esc` inside the camera window to exit.*

## Explicit Options
```bash
/home/eshwar/code/interrupt/venv/bin/python3 /home/eshwar/code/detection/perception/detection/run_detection_demo.py \
  --camera /dev/video0 \
  --model /home/eshwar/code/detection/yolov8n.pt \
  --conf 0.5 \
  --width 1280 \
  --height 720 \
  --fps 30
```

## Headless Demonstration / Profiling Run (Bounded 100 Frames)
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py \
  --conf 0.5 \
  --max-frames 100 \
  --headless
```

## Automated Unit Tests
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```
