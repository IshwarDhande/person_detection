# Repository Demonstration and Execution Commands

> **Purpose:** This document is the centralized reference for executing live demonstrations, benchmarking utilities, calibration tools, and test suites across all project milestones. As subsequent milestones (M2–M10) are implemented, their operational commands will be documented here.

```text
+---------------------------------------------------------------------------------------------------+
|                           DEMO & EXECUTION COMMANDS DIRECTORY                                     |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  * M0: Calibration & Hardware Tools (camera_hardware_discovery.py, camera_capture_utility.py)      |
|  * M1: Basic Person Detection Live Demo & Tests (run_detection_demo.py)                           |
|  * M2: Reliable Single-Person Detection with EMA Smoothing (--ema-alpha)                          |
|  * M3: Single-Person Tracking with 8-D Kalman Filter & IoU Gating (run_tracking_demo.py)          |
|  * M4: Frame-Level Readiness Perception & Interaction Zone (run_readiness_demo.py)                |
|  * M5+: Temporal Approach & Final Readiness Decisions (Future Milestones)                         |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## Milestone M1 — Basic Person Detection Demo

The following commands launch and evaluate the Milestone M1 person detection runner.

### 1. Quick Start (Interactive Laptop Webcam)
Runs live person detection on `/dev/video0` with real-time bounding boxes, confidence labels, and HUD metrics (FPS and inference latency).
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py
```
> *Press `q` or `Esc` inside the camera window to exit.*

### 2. Full Parameter Launch (Explicit Options)
Runs live detection with explicit paths, camera device, resolution, and framerate requests. Works from any working directory.
```bash
/home/eshwar/code/interrupt/venv/bin/python3 /home/eshwar/code/detection/perception/detection/run_detection_demo.py \
  --camera /dev/video0 \
  --model /home/eshwar/code/detection/yolov8n.pt \
  --conf 0.5 \
  --width 1280 \
  --height 720 \
  --fps 30
```

### 3. Headless Profiling Run (Bounded 100 Frames)
Runs inference without opening a graphical window to benchmark CPU latency and application FPS directly in the terminal.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py \
  --conf 0.5 \
  --max-frames 100 \
  --headless
```

### 4. M1 Automated Unit Test Suite
Executes all 26 unit tests covering person-class filtering, threshold boundaries, and camera initialization seams.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```

---

## Milestone M2 — Reliable Single-Person Detection Demo

The following commands launch and evaluate the Milestone M2 person detection runner with EMA bounding-box coordinate smoothing.

### 1. Interactive Demo with EMA Smoothing
Runs live detection on `/dev/video0` with dual-bounding boxes (green smoothed primary box, subtle cyan raw detector box) and HUD smoothing status.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py \
  --camera /dev/video0 \
  --ema-alpha 0.65
```

### 2. Bypass Smoothing
Runs raw detection with EMA smoothing bypassed.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py \
  --no-smoothing
```

---

## Milestone M3 — Single-Person Tracking Demo

The following commands launch and evaluate the Milestone M3 single-person tracking runner with 8-D constant-velocity Kalman filtering, IoU association, prediction-uncertainty gating, and lifecycle management.

### 1. Quick Start (Interactive Laptop Webcam)
Runs live single-person tracking on `/dev/video0`. Displays temporary track ID, lifecycle state (TENTATIVE, CONFIRMED, LOST/COASTING), and explicit distinction between real observations and coasting predictions.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/tracking/run_tracking_demo.py
```
> *Press `q` or `Esc` inside the camera window to exit.*

### 2. Full Parameter Launch (Explicit Options)
Runs live tracking with custom parameters for IoU association threshold and retention timeout:
```bash
/home/eshwar/code/interrupt/venv/bin/python3 /home/eshwar/code/detection/perception/tracking/run_tracking_demo.py \
  --camera /dev/video0 \
  --model /home/eshwar/code/detection/yolov8n.pt \
  --conf 0.5 \
  --iou-thresh 0.3 \
  --timeout 2.0 \
  --width 1280 \
  --height 720 \
  --fps 30
```

### 3. Headless Profiling Run (Bounded 60 Frames)
Runs tracking inference without opening a graphical window to profile pipeline throughput and inference latency.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/tracking/run_tracking_demo.py \
  --max-frames 60 \
  --headless
```

### 4. Complete Repository Unit Test Suite (M1 + M2 + M3)
Executes all 81 unit tests across detection and tracking subsystems:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```

---

## Milestone M0 — Camera Hardware & Calibration Tools

### 1. Camera Hardware Discovery & Streaming Benchmark
Scans V4L2 devices, USB topology, formats, and executes a 100-frame streaming throughput benchmark.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 tools/calibration/camera_hardware_discovery.py
```


---

## Milestone M4 — Frame-Level Readiness Perception Demo

The following commands launch and evaluate the Milestone M4 frame-level readiness perception runner with person detection, bounding-box validity, MediaPipe face visibility, interaction-zone membership, and approximate head orientation.

### 1. Quick Start (Interactive Laptop Webcam)
Runs live readiness perception on `/dev/video0`. Displays the interaction-zone polygon, person bounding box (color-coded by zone membership), face bounding box, and diagnostic HUD showing all five observation conditions.
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/readiness/run_readiness_demo.py
```
> *Press `q` or `Esc` inside the camera window to exit.*

### 2. Full Parameter Launch (Explicit Options)
Runs live readiness perception with custom YAML configuration and device overrides:
```bash
/home/eshwar/code/interrupt/venv/bin/python3 /home/eshwar/code/detection/perception/readiness/run_readiness_demo.py \
  --config /home/eshwar/code/detection/configs/m4/stationary_readiness.yaml \
  --camera /dev/video0 \
  --model /home/eshwar/code/detection/yolov8n.pt \
  --face-model /home/eshwar/code/detection/models/face_landmarker.task \
  --conf 0.5 \
  --width 1280 \
  --height 720 \
  --fps 30
```

### 3. Headless Profiling Run (Bounded 90 Frames)
Runs perception inference without opening a graphical window to measure component and pipeline latencies directly in the terminal:
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/readiness/run_readiness_demo.py \
  --camera /dev/video0 \
  --headless \
  --max-frames 90
```

### 4. M4 Dedicated Unit Test Suite
Executes all 48 unit tests covering bounding-box validation, interaction-zone geometry, face analysis, orientation ratios, observation contract, and config schemas:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests/perception/readiness -t . -v
```

### 5. Complete Repository Regression Test Suite (M1 + M2 + M3 + M4)
Executes all 138 unit tests across the entire codebase:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```

