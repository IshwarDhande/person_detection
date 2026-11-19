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
|  * M5: Temporal Behaviour Estimation & Approach Tracking (run_temporal_demo.py)                   |
|  * M6: Readiness-Based Single-Person First Contact Pipeline (run_first_contact_demo.py)          |
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

### 5. Complete Repository Regression Test Suite (M1 through M5)
Executes all 214 unit tests across the entire codebase:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```

---

## Milestone M5 — Temporal Behaviour Estimation Demo

The following commands launch and evaluate the Milestone M5 temporal behaviour estimation runner, which integrates M1 YOLOv8n detection, M3 single-person tracking, M4 frame-level readiness perception, and M5 temporal estimators (velocity regression, 1.0s displacement, zone centroid approach, stationary predicate, and 2.0s stability timing with 200 ms noise tolerance).

### 1. Quick Start (Interactive Laptop Webcam)
Runs live temporal behaviour estimation on `/dev/video0`. Displays the interaction-zone polygon, centroid crosshair, person bounding box, center trajectory breadcrumbs, velocity vector arrow, distance-to-centroid line, and diagnostic HUD with real-time indicators:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 perception/temporal/run_temporal_demo.py
```
> *Press `q` or `Esc` inside the camera window to exit.*

### 2. Full Parameter Launch (Explicit Options)
Runs live temporal perception with custom configurations, compute device, and camera parameters:
```bash
PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 /home/eshwar/code/detection/perception/temporal/run_temporal_demo.py \
  --config /home/eshwar/code/detection/configs/m4/stationary_readiness.yaml \
  --temporal-config /home/eshwar/code/detection/configs/m5/temporal_behaviour.yaml \
  --camera /dev/video0 \
  --model /home/eshwar/code/detection/yolov8n.pt \
  --face-model /home/eshwar/code/detection/models/face_landmarker.task \
  --conf 0.5 \
  --device cpu
```

### 3. Headless Profiling Run (Bounded 100 Frames)
Runs perception and temporal estimation without graphical window display to benchmark pipeline latencies:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 perception/temporal/run_temporal_demo.py \
  --camera /dev/video0 \
  --headless \
  --max-frames 100
```

### 4. M5 Dedicated Unit Test Suite
Executes all 61 dedicated M5 unit tests covering polygon centroid calculations, observation history, linear regression velocity, displacement, stationary classification, centroid-directed approach, stability duration timing, noise tolerance, and synthetic sequences:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests/perception/temporal -t . -v
```

---

## Milestone M6 — Readiness-Based Single-Person First Contact Demo

The following commands launch and evaluate the Milestone M6 complete perception-to-first-contact pipeline.

### 1. Quick Start (Interactive Live Demo)
Runs the complete end-to-end first-contact pipeline on `/dev/video0` with real-time HUD, readiness state badge, predicate checklist, distance indicator, and one-shot event logger:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 perception/contact/run_first_contact_demo.py
```
> *Press `q` or `Esc` inside the camera window to exit.*

### 2. Full Parameter Launch (Explicit Options)
Runs live first contact with custom configurations, override mock distance, compute device, and camera parameters:
```bash
PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 /home/eshwar/code/detection/perception/contact/run_first_contact_demo.py \
  --config /home/eshwar/code/detection/configs/m4/stationary_readiness.yaml \
  --temporal-config /home/eshwar/code/detection/configs/m5/temporal_behaviour.yaml \
  --contact-config /home/eshwar/code/detection/configs/m6/first_contact.yaml \
  --camera /dev/video0 \
  --model /home/eshwar/code/detection/yolov8n.pt \
  --face-model /home/eshwar/code/detection/models/face_landmarker.task \
  --conf 0.5 \
  --distance 1.2 \
  --device cpu
```

### 3. Headless Profiling Run (Bounded Frames)
Runs the complete perception pipeline without graphical window display to benchmark component latencies and resource usage:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 perception/contact/run_first_contact_demo.py \
  --camera /dev/video0 \
  --headless \
  --max-frames 100
```

### 4. M6 Dedicated Unit Test Suite
Executes all 97 dedicated M6 unit tests covering predicate evaluation, continuous eligibility timer, readiness state machine, target locking, tracking-loss grace periods, reacquisition, and synthetic interaction scenarios:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests/perception/contact -t . -v
```

### 5. Full Repository Test Suite (M1–M6)
Executes all 328 unit tests across all milestones (M1–M6) to verify zero regressions:
```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```



