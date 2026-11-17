# Project Implementation Audit — Current State Assessment

This report provides a formal assessment of the current state of the **Vision-Based First Contact for a Reception Social Robot** project (`/home/eshwar/code/detection`). It assesses all completed work, hardware characterization benchmarks, and existing code against the reorganized roadmap ([IMPLEMENTATION_ROADMAP.md](file:///home/eshwar/code/detection/research/IMPLEMENTATION_ROADMAP.md)).

```text
+---------------------------------------------------------------------------------------------------+
|                                  PROJECT CURRENT STATE OVERVIEW                                   |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  STAGE A: Prototype Development (Laptop Webcam /dev/video0)                                      |
|  +--------------------+       +--------------------+       +-----------------------------------+  |
|  |  M0: HW Discovery  | ----> |   M1: Detection    | ----> | M2 - M9: Tracking, Spatial, State  |  |
|  |    & Acquisition   |       |   (YOLOv8 Live)    |       | FSM, Event Dispatch, Integration  |  |
|  |     [VERIFIED]     |       |    [NOT STARTED]   |       |           [NOT STARTED]           |  |
|  +--------------------+       +--------------------+       +-----------------------------------+  |
|            |                                                                                      |
|            v                                                                                      |
|  STAGE B: Robot Hardware Deployment                                                               |
|  +---------------------------------------------------------------------------------------------+  |
|  | M10: Robot Mount, ChArUco Calibration, Desk Occlusion Geometry, Physical Validation         |  |
|  |                                        [DEFERRED]                                           |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 1. Executive Summary

1. **Roadmap Realignment**: The project has adopted a two-stage strategy ([IMPLEMENTATION_ROADMAP.md](file:///home/eshwar/code/detection/research/IMPLEMENTATION_ROADMAP.md)):
   * **Stage A (M1–M9)**: Prototyping perception, tracking, pose-based localization, temporal movement filters, interaction readiness, and decision state machines using the workstation's built-in webcam (`/dev/video0`).
   * **Stage B (M10)**: Transferring and validating the pipeline on the physical reception robot (sensor mounting, ChArUco lens calibration, desk geometry, and environmental lighting).
2. **Current Implementation Scope**: All functional software implemented to date resides in `tools/calibration/`. No perception modules, tracking algorithms, pose estimators, or state machine runners exist in `perception/`, `spatial_understanding/`, or `behaviour/` yet.
3. **Hardware & Environment Status**: The development environment (`/home/eshwar/code/interrupt/venv`) is operational with PyTorch, Ultralytics YOLOv8, MediaPipe, and OpenCV. Camera `/dev/video0` has been characterized and benchmarked at 1280×720 @ 29.8 FPS under MJPG.

---

## 2. Existing Implementation Inventory

| Component | Status | Source File Paths | Main Functionality | Dependencies / Models | Stage | Automated Tests | Demonstrated |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Camera Hardware Discovery** | Fully Implemented & Verified | [camera_hardware_discovery.py](file:///home/eshwar/code/detection/tools/calibration/camera_hardware_discovery.py) | Discovers V4L2 devices, USB topology, formats/resolutions, and runs 100-frame streaming benchmarks | `opencv-python`, V4L2 sysfs, `v4l2-ctl` | Implemented | No | Yes (29.8 FPS benchmarked) |
| **Camera Capture Utility** | Fully Implemented & Verified | [camera_capture_utility.py](file:///home/eshwar/code/detection/tools/calibration/camera_capture_utility.py) | Acquires calibration/scene frames; supports Mode A (scene distances 0.0m–2.5m), Mode B (ChArUco), `--delay` countdown, and fail-closed `--consent` | `opencv-python`, `argparse`, JSON | Implemented | No | Yes (`scene_empty.png` captured & indexed) |
| **Person Detection** | Weights present; no module | [yolov8n.pt](file:///home/eshwar/code/detection/yolov8n.pt) (model weights only) | Model weights downloaded during ad-hoc validation; no perception module or live detection script written | Ultralytics YOLOv8 | Experimental / Planned | No | Ad-hoc CLI only |
| **Person Tracking** | Not Started | None | Multi-target association and trajectory management | Planned: ByteTrack / BoT-SORT | Planned (M3) | No | No |
| **Pose & Keypoint Estimation** | Not Started | None | Upper-body keypoint extraction for distance/orientation | Planned: YOLOv8-pose / MediaPipe | Planned (M4) | No | No |
| **Spatial Localization** | Analysis done; code not started | [m0_framing_and_calibration_report.md](file:///home/eshwar/code/detection/docs/calibration/m0_framing_and_calibration_report.md) | Optical ray-tracing and geometric analysis completed; calibration solver not implemented | Theoretical ray-tracing | Planned (M4, M10) | No | Theoretical proof only |
| **Movement / Approach Estimation** | Not Started | None | Temporal filtering (radial velocity $v_r$, trajectory smoothing) | Planned: Kalman Filter / Savitzky-Golay | Planned (M5) | No | No |
| **Stationary State Estimation** | Not Started | None | Dwelling/lingering verification at interaction threshold | Planned | Planned (M5) | No | No |
| **Orientation Estimation** | Not Started | None | Torso/face yaw relative to camera optical axis | Planned | Planned (M4) | No | No |
| **Readiness Evaluation** | Not Started | None | Multi-factor engagement scoring | Planned | Planned (M6) | No | No |
| **Candidate Management** | Not Started | None | Target selection and identity handover | Planned | Planned (M7) | No | No |
| **Lifecycle Management** | Not Started | None | State machine (APPROACHING, ENGAGED, LOST, TIMEOUT) | Planned | Planned (M8) | No | No |
| **Event Integration Interfaces** | Not Started | None | Event emission for speech/dialogue dispatch | Planned: JSON / ROS / ZeroMQ | Planned (M9) | No | No |
| **Runtime Pipeline Orchestration** | Not Started | None | Main frame loop coordinating detection, tracking, spatial, FSM | Planned | Planned (M8, M9) | No | No |
| **Automated Test Infrastructure** | Not Started | None | Unit tests and integration test suites | Planned: `pytest` | Planned | No | No |

---

## 3. Actual Files & Modules on Disk

The repository currently contains 31 files across 8 active directories:

```text
detection/
├── activate_env.sh                                            # Virtual environment activation script
├── configs/
│   └── runtime/
│       └── environment.yaml                                  # Environment & Python path configuration
├── data/
│   └── raw/
│       └── calibration/
│           ├── calibration_metadata.json                     # JSON index of captured calibration frames
│           └── scene_inspection/
│               └── scene_empty.png                           # Baseline empty background reference frame
├── docs/
│   ├── calibration/
│   │   ├── camera_capture_guide.md                           # Operator manual for capture utility
│   │   ├── m0_camera_hardware_inventory.json                 # Machine-readable hardware discovery inventory
│   │   ├── m0_camera_hardware_inventory.md                   # Human-readable hardware discovery report
│   │   └── m0_framing_and_calibration_report.md              # Optical framing & desk occlusion ray-tracing
│   └── project_implementation_audit.md                       # Current state assessment report (this file)
├── README.md                                                 # Project top-level documentation
├── research/                                                 # Research documents & governance files
│   ├── AGENTS.md
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   ├── ENGINEERING_STANDARDS.md
│   ├── From Camera to First Contact.png
│   ├── IMPLEMENTATION_ROADMAP.md                             # Incremental milestone specification M0–M10
│   ├── mermaid-diagram.png
│   ├── project_overview_implementation.md
│   ├── r2_ques.md
│   ├── r3_ques.md
│   ├── README.md
│   ├── REPOSITORY_MAP.md
│   ├── round_1.md
│   ├── round_2.md
│   ├── round_3.md
│   ├── STATUS.md
│   ├── Vision-Based First Contact System.png
│   └── WORKLOG.md
├── tools/
│   └── calibration/
│       ├── camera_capture_utility.py                         # Acquisition utility (Mode A/B, consent, delay)
│       └── camera_hardware_discovery.py                      # V4L2/UVC discovery and benchmark utility
└── yolov8n.pt                                                # Downloaded YOLOv8 nano model weights
```

---

## 4. Validation & Experimental Evidence

All completed hardware discovery and acquisition experiments were performed using `/home/eshwar/code/interrupt/venv`:

```text
+---------------------------------------------------------------------------------------------------+
|                                 HARDWARE BENCHMARK & STREAMING METRICS                            |
+---------------------------------------------------------------------------------------------------+
|  Device:                 /dev/video0 (Luxvisions Innotech Integrated RGB Camera, USB 0x30c9:0x00a8)|
|  Driver:                 uvcvideo (Linux kernel 6.8.0-52-generic)                                 |
|  Active Format:          MJPG (Motion-JPEG compressed)                                            |
|  Active Resolution:      1280 x 720 (16:9)                                                        |
|  Requested FPS:          30 FPS                                                                   |
|  Observed Streaming FPS: 29.80 FPS (over 100 benchmark frames)                                    |
|  Mean Frame Interval:    33.56 ms (standard deviation: 2.72 ms)                                   |
|  Frame Drops:            0 dropped frames detected                                                |
|  Channel Format & Dtype: 3 channels (BGR), uint8                                                  |
|  USB Bandwidth Note:     Uncompressed YUYV at 1280x720 throttles to 10.0 FPS due to USB 2.0 bus   |
|                          saturation. MJPG must be explicitly configured to achieve 30 FPS.        |
+---------------------------------------------------------------------------------------------------+
```

### Reference Capture Validation
* **Empty Scene Baseline**: Captured to [scene_empty.png](file:///home/eshwar/code/detection/data/raw/calibration/scene_inspection/scene_empty.png) and recorded in [calibration_metadata.json](file:///home/eshwar/code/detection/data/raw/calibration/calibration_metadata.json).
* **Detection Verification**: Evaluated with YOLOv8 nano inference; returned **0 person detections**, confirming a clean background baseline without phantom targets.

---

## 5. M0 Completion Assessment

### What M0 Accomplished
1. **Camera Discovery**: Completely characterized `/dev/video0` hardware, driver, UVC controls, and bus topology ([m0_camera_hardware_inventory.md](file:///home/eshwar/code/detection/docs/calibration/m0_camera_hardware_inventory.md)).
2. **Streaming Validation**: Confirmed stable 1280×720 @ 29.8 FPS streaming via MJPG with sub-3 ms frame jitter.
3. **Capture Utility**: Implemented [camera_capture_utility.py](file:///home/eshwar/code/detection/tools/calibration/camera_capture_utility.py) with operator countdown (`--delay`), fail-closed consent (`--consent`), and structured JSON metadata logging.
4. **Framing & Geometry Proof**: Completed optical ray-tracing analysis ([m0_framing_and_calibration_report.md](file:///home/eshwar/code/detection/docs/calibration/m0_framing_and_calibration_report.md)).
   * *Critical Finding*: At camera height $h = 115\text{ cm}$ and pitch $\theta = 0^\circ$, ground contact occurs at $d \approx 2.78\text{ m}$. At the 1.5 m first-contact boundary, feet are occluded below the frame and behind the reception desk.
   * *Architectural Conclusion*: Ground-plane homography using bounding-box bottom points is mathematically infeasible. Spatial localization must use upper-body keypoints (shoulders/head) rather than floor contact.

```text
+---------------------------------------------------------------------------------------------------+
|                         OPTICAL RAY-TRACING & OCCLUSION AT 1.5 METERS                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  h = 115 cm, pitch = 0 deg                                                                        |
|                                                                                                   |
|  Camera [O] ========================================> Principal Axis                              |
|       |   \                                                                                       |
|       |     \                                                                                     |
|       |       \  Ray to Lower Frame Boundary (-22.5 deg)                                          |
|       |         \                                                                                 |
|       |           \                                                                               |
|  Desk |-[Desk]-----\---------------------------------- [Floor Contact at 2.78 m]                  |
|       |             \                                                                             |
|       |              \                                                                            |
|  0 cm +---------------+------------------------------------------------------------ Ground Plane  |
|      0.0 m          1.0 m               1.5 m                             2.78 m                  |
|                                     [Person at 1.5 m]                                             |
|                                     * Head & Shoulders: IN FRAME                                  |
|                                     * Torso: IN FRAME                                             |
|                                     * Hips: CUT OFF BY LOWER FRAME (y > 720 px)                   |
|                                     * Feet: OCCLUDED BY DESK & OFF-SCREEN                         |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

### What Remains Open in M0
* **Intrinsic Matrix Estimation**: ChArUco / checkerboard pattern calibration has not been run because a physical printed calibration board has not yet been staged.
* **Robot Hardware Deployment**: Robot-mounted camera discovery and physical desk dimension measurements are deferred to **M10** under the two-stage roadmap.

---

## 6. M1–M10 Milestone Mapping Table

| Milestone | Roadmap Goal | Current Codebase State | Status | Evidence / Justification |
| :--- | :--- | :--- | :---: | :--- |
| **M0** | Hardware Discovery & Calibration | Discovery done, capture utility done, framing proof done; physical board pending | **Partially Complete** | [camera_hardware_discovery.py](file:///home/eshwar/code/detection/tools/calibration/camera_hardware_discovery.py), [camera_capture_utility.py](file:///home/eshwar/code/detection/tools/calibration/camera_capture_utility.py). Intrinsic board calibration pending; robot calibration deferred to M10. |
| **M1** | Basic Person Detection | Weights present (`yolov8n.pt`); no application module or live runner | **Not Started** | No files exist in `perception/detection/`. Model weights verified only through ad-hoc validation. |
| **M2** | Detection Optimization & Benchmarking | No benchmarking harness or TensorRT/ONNX export | **Not Started** | No benchmark scripts or optimized engine configs exist. |
| **M3** | Multi-Object Tracking (MOT) | No tracker module or association logic | **Not Started** | No files exist in `perception/tracking/`. |
| **M4** | 2D/3D Spatial & Pose Estimation | Ray-tracing theory documented; no pose or distance estimator | **Not Started** | Theoretical calculations exist in [m0_framing_and_calibration_report.md](file:///home/eshwar/code/detection/docs/calibration/m0_framing_and_calibration_report.md), but no runtime code exists in `spatial_understanding/`. |
| **M5** | Temporal Movement & Approach Estimation | No trajectory filter or approach detector | **Not Started** | No velocity computation or dwelling detector implemented. |
| **M6** | Interaction Readiness & Engagement | No readiness scoring module | **Not Started** | No readiness logic or engagement criteria implemented. |
| **M7** | Candidate Selection & Handover | No priority arbiter or selection logic | **Not Started** | Multi-person candidate management not implemented. |
| **M8** | State Machine & Pipeline Integration | No FSM or pipeline orchestrator | **Not Started** | No lifecycle states or central runner implemented. |
| **M9** | Event Interfaces & Downstream Handover | No event schema or dispatcher | **Not Started** | Downstream dispatch interfaces not implemented. |
| **M10** | Robot Deployment & On-Hardware Tuning | Deferred to physical robot arrival | **Deferred** | Explicitly scoped to Stage B per [IMPLEMENTATION_ROADMAP.md](file:///home/eshwar/code/detection/research/IMPLEMENTATION_ROADMAP.md). |

---

## 7. Outstanding Gaps & Blockers

1. **Missing Perception Code**: While directories and configurations are documented, the application source tree (`perception/`, `spatial_understanding/`, `behaviour/`) has no executable code.
2. **Absence of Automated Test Suite**: There are currently zero unit or regression tests (`pytest` has not been set up in `tests/`).
3. **Physical Calibration Target**: Acquiring an exact intrinsic camera matrix ($f_x, f_y, c_x, c_y$) requires printing or mounting a physical ChArUco/checkerboard grid (Mode B in capture utility).
4. **Keypoint-Based Localization Requirement**: Because feet are occluded at the critical 1.5 m threshold, Milestone M4 cannot rely on floor-plane homography and must be designed around torso/facial keypoint geometry.

---

## 8. Recommended Immediate Next Action

Proceed to **Milestone M1 — Basic Person Detection**:

```text
+---------------------------------------------------------------------------------------------------+
|                               RECOMMENDED IMMEDIATE WORKFLOW (M1)                                 |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. Implement Detection Wrapper:                                                                  |
|     `perception/detection/person_detector.py`                                                     |
|     - Encapsulate YOLOv8 nano inference                                                           |
|     - Filter class 0 ('person') exclusively                                                       |
|     - Expose confidence threshold parameter (e.g., conf >= 0.5)                                   |
|                                                                                                   |
|  2. Implement Standalone Live Runner:                                                              |
|     `perception/detection/run_detection_demo.py`                                                  |
|     - Stream live from `/dev/video0` (1280x720 @ 30 FPS MJPG)                                    |
|     - Render real-time bounding boxes, confidence scores, and actual FPS                          |
|     - Support clean exit on 'q' or ESC                                                            |
|                                                                                                   |
|  3. Validate M1 Exit Criteria:                                                                    |
|     - Verify sustained >= 25 FPS live inference on the development machine                        |
|     - Ensure zero unhandled exceptions on camera disconnect or window close                       |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

