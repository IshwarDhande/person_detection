# Milestone M1 — Basic Person Detection: Implementation Details

**Project:** Vision-Based First Contact for a Reception Social Robot  
**Milestone:** M1 — Basic Person Detection  
**Status:** Completed and Verified  
**Date:** 2026-11-17  
**Specification:** [m1_basic_detection.md](file:///home/eshwar/code/detection/docs/m1_basic_detection.md)  

```text
+---------------------------------------------------------------------------------------------------+
|                        MILESTONE M1: BASIC PERSON DETECTION COMPLETE                              |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Laptop Camera: /dev/video0] (1280x720 @ 30 FPS MJPG)                                            |
|       |                                                                                           |
|       v                                                                                           |
|  [Frame Acquisition] ---------------------------------------------+                               |
|       |                                                           |                               |
|       v (perf_counter start)                                      | Monotonic Loop Timestamps     |
|  [PersonDetector.detect()]                                        v                               |
|       | - YOLOv8n inference on frame                      [Application FPS]                       |
|       | - Strict COCO class_id == 0 filtering                     |                               |
|       | - Strict confidence >= threshold filtering                |                               |
|       v (perf_counter end -> Latency ms)                          |                               |
|  [Visual Rendering & HUD Overlay] <-------------------------------+                               |
|       | - Green bounding boxes (x1, y1) -> (x2, y2)                                               |
|       | - Confidence label: "Person: 93.1%"                                                       |
|       | - HUD: Instant & Avg FPS, Latency ms, Count, Resolution                                   |
|       v                                                                                           |
|  [Display & Event Loop]                                                                           |
|       | - 'q' / 'Esc' / Window Close -> Clean Exit & Resource Teardown                            |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## Quick Start — Live Demonstration Commands

### Option A: From Repository Root (Recommended)
```bash
cd /home/eshwar/code/detection
/home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py
```

### Option B: Single One-Line Command (Works from Any Terminal Directory)
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py --camera /dev/video0 --conf 0.5 --width 1280 --height 720 --fps 30
```
> *Press `q` or `Esc` inside the window to exit.*

### Option C: Headless Demonstration / Profiling Run (Bounded 100 Frames)
```bash
cd /home/eshwar/code/detection && /home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py --conf 0.5 --max-frames 100 --headless
```

---

## 1. Files Created and Modified

All files and directories follow strict lowercase naming conventions:

| File Path | Action | Description |
| :--- | :--- | :--- |
| [person_detector.py](file:///home/eshwar/code/detection/perception/detection/person_detector.py) | Created | Core reusable perception module: `PersonDetection` dataclass and `PersonDetector` class wrapping YOLOv8n. |
| [run_detection_demo.py](file:///home/eshwar/code/detection/perception/detection/run_detection_demo.py) | Created | Standalone live demonstration runner for `/dev/video0` with real-time HUD and metrics tracking. |
| [\_\_init\_\_.py (detection)](file:///home/eshwar/code/detection/perception/detection/__init__.py) | Created | Detection package initializer exposing `PersonDetection` and `PersonDetector`. |
| [\_\_init\_\_.py (perception)](file:///home/eshwar/code/detection/perception/__init__.py) | Created | Perception root package initializer. |
| [test_person_detector.py](file:///home/eshwar/code/detection/tests/perception/detection/test_person_detector.py) | Created | 19 automated unit tests for class filtering, confidence boundaries, data validation, and mock seams. |
| [test_detection_config.py](file:///home/eshwar/code/detection/tests/perception/detection/test_detection_config.py) | Created | 7 automated unit tests for CLI argument parsing, input boundary validation, and camera failure handling. |
| [\_\_init\_\_.py (tests packages)](file:///home/eshwar/code/detection/tests/perception/detection/__init__.py) | Created | Package initializers across `tests/`, `tests/perception/`, and `tests/perception/detection/`. |

---

## 2. Implemented Functionality

### A. Reusable `PersonDetector`
* **Model Loading & Validation**: Safely initializes YOLOv8 nano from [yolov8n.pt](file:///home/eshwar/code/detection/yolov8n.pt); validates weights existence (`FileNotFoundError`) and checks threshold interval `[0.0, 1.0]` (`ValueError`).
* **Strict Class Filtering**: Exclusively processes and returns COCO class `0` (`person`). Non-person classes (bicycles, vehicles, pets, furniture) are filtered out defensively in post-processing.
* **Configurable Confidence**: Filters detections below `confidence_threshold` (default `0.50`), with boundary condition `confidence >= threshold` verified.
* **Typed Data Representation**: Returns `list[PersonDetection]` records containing finite pixel coordinates `(x1, y1, x2, y2)`, float confidence, and `class_id = 0`. Disallows persistent tracking state.
* **Resource Management**: Exposes `close()` to release PyTorch/YOLO references cleanly.

### B. Live Demonstration Runner
* **Camera Streaming**: Captures from `/dev/video0` (configurable via `--camera`), negotiating 1280×720 @ 30 FPS over MJPG compression to avoid USB 2.0 bandwidth bottlenecks.
* **Real-Time Visuals**: Renders high-contrast green bounding boxes and background-plated confidence labels (`Person: 93.1%`).
* **On-Screen Display (HUD)**: Displays rolling application FPS, cumulative mean FPS, frame inference latency ($L_t$ in ms), active detection count, and negotiated resolution.
* **Graceful Teardown**: Binds `'q'`, `'Q'`, `Esc` (27), and window close button `[X]` to safe teardown paths in a `finally` block, ensuring OpenCV windows and V4L2 device file handles are never leaked.

---

## 3. Python Environment & Dependencies Used

* **Virtual Environment**: `/home/eshwar/code/interrupt/venv`
* **Python Version**: `3.12.3`
* **PyTorch**: `2.13.0+cu130` (CPU compute backend active)
* **Ultralytics**: `8.4.115` (API wrapping YOLOv8n)
* **OpenCV**: `5.0.0` (V4L2 video capture and rendering)
* **Weights File**: Reused existing [yolov8n.pt](file:///home/eshwar/code/detection/yolov8n.pt) in repository root.

---

## 4. Commands to Launch the Live Demo

### GUI Live Demonstration (Interactive Laptop Webcam)
```bash
/home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py \
  --camera /dev/video0 \
  --model yolov8n.pt \
  --conf 0.5 \
  --width 1280 \
  --height 720 \
  --fps 30
```
*Press `q` or `Esc` inside the window to exit.*

### Headless Demonstration / Profiling Run (Bounded 100 Frames)
```bash
/home/eshwar/code/interrupt/venv/bin/python3 perception/detection/run_detection_demo.py \
  --camera /dev/video0 \
  --conf 0.5 \
  --max-frames 100 \
  --headless
```

### Running Automated Unit Tests
```bash
/home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```

---

## 5. Automated Test Results

The test suite executed with zero errors and zero skipped tests:

```text
+---------------------------------------------------------------------------------------------------+
|                                 AUTOMATED TEST EXECUTION SUMMARY                                  |
+---------------------------------------------------------------------------------------------------+
|  Ran 26 tests in 0.637s                                                                           |
|  Result: OK (ALL 26 TESTS PASSED)                                                                 |
+---------------------------------------------------------------------------------------------------+
|  Suite 1: test_person_detector.py (19 tests)                                                      |
|    * test_valid_detection_instantiation                      [PASS]                                   |
|    * test_reject_non_person_class_id                         [PASS]                                   |
|    * test_reject_out_of_bound_confidence                     [PASS]                                   |
|    * test_reject_invalid_bbox_length                         [PASS]                                   |
|    * test_reject_non_finite_coordinates                      [PASS]                                   |
|    * test_reject_inverted_coordinates                        [PASS]                                   |
|    * test_missing_model_path_raises_file_not_found           [PASS]                                   |
|    * test_invalid_confidence_threshold_raises_value_error    [PASS]                                   |
|    * test_boundary_confidence_thresholds                     [PASS]                                   |
|    * test_model_init_failure_raises_runtime_error            [PASS]                                   |
|    * test_person_class_retained                              [PASS]                                   |
|    * test_non_person_classes_filtered_out                    [PASS]                                   |
|    * test_below_threshold_detections_filtered_out            [PASS]                                   |
|    * test_boundary_confidence_threshold_retained             [PASS]                                   |
|    * test_empty_model_results_produce_empty_list             [PASS]                                   |
|    * test_multiple_persons_detected                          [PASS]                                   |
|    * test_invalid_input_frame_raises_value_error             [PASS]                                   |
|    * test_inference_failure_raises_runtime_error             [PASS]                                   |
|    * test_close_detector_releases_resources                  [PASS]                                   |
|                                                                                                   |
|  Suite 2: test_detection_config.py (7 tests)                                                      |
|    * test_default_cli_arguments                              [PASS]                                   |
|    * test_custom_cli_arguments                               [PASS]                                   |
|    * test_reject_invalid_confidence_arg                      [PASS]                                   |
|    * test_reject_non_positive_dimensions                     [PASS]                                   |
|    * test_camera_open_failure_raises_runtime_error           [PASS]                                   |
|    * test_initial_frame_read_failure_releases_camera         [PASS]                                   |
|    * test_successful_camera_open                             [PASS]                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 6. Actual FPS and Inference Latency Measurements

A 60-frame live benchmark was executed using `/dev/video0` on the development workstation:

```text
+---------------------------------------------------------------------------------------------------+
|                             M1 PERFORMANCE BASELINE METRICS (CPU)                                 |
+---------------------------------------------------------------------------------------------------+
|  Camera Device:                 /dev/video0 (Luxvisions Innotech Integrated RGB)                  |
|  Negotiated Resolution:         1280 x 720 (16:9)                                                 |
|  Compression Codec:             MJPG                                                              |
|  Requested Camera FPS:          30.0 FPS                                                          |
|  Processed Frames:              60 frames                                                         |
|  Total Benchmark Duration:      7.42 seconds                                                      |
|                                                                                                   |
|  Inference Latency (CPU):                                                                         |
|    - Frame 1 (JIT / Cold Start): 1607.22 ms                                                       |
|    - Steady-State Min Latency:  72.14 ms                                                          |
|    - Median (P50) Latency:      88.47 ms                                                          |
|    - 95th Percentile (P95):     109.54 ms                                                         |
|    - Mean Latency (incl. init): 114.14 ms                                                         |
|                                                                                                   |
|  Observed Application FPS:                                                                        |
|    - Steady-State Loop FPS:     10.50 FPS                                                         |
|    - Overall Mean FPS:          7.96 FPS                                                          |
+---------------------------------------------------------------------------------------------------+
```

---

## 7. Live Demonstration Status

The implementation was validated against real image frames and live video:

1. **Empty Reception Scene Verification**:
   * Evaluated against [scene_empty.png](file:///home/eshwar/code/detection/data/raw/calibration/scene_inspection/scene_empty.png) and the 60-frame live camera stream.
   * **Result**: **0 person detections**, confirming zero false positives in an empty room.
2. **Person Entering Frame Verification**:
   * Evaluated against real frame capture containing the operator seated at the reception desk.
   * **Result**: Detected **1 person** with bounding box `(152.4, 26.5, 716.2, 569.1)` at **93.1% confidence** (`class_id=0`).
3. **Clean Shutdown**:
   * Tested window close and terminal interrupts; verified camera handle is closed and GUI windows destroyed.

---

## 8. Known Limitations

* **CPU Latency Ceiling**: Running unquantized PyTorch FP32 inference on CPU yields ~88 ms steady-state latency (~10–11 FPS throughput). Optimization (ONNX Runtime, OpenVINO, INT8) is explicitly deferred to Milestone M2.
* **No Temporal Continuity**: Frame-level bounding boxes fluctuate independently per frame; bounding boxes do not carry identity or tracking IDs across frames.
* **No Distance or Physical Metric**: Pixel coordinates are 2D image coordinates only; no metric distance or 3D localization is computed.

---

## 9. Milestone M1 Acceptance Criteria Checklist

| Acceptance Criterion | Status | Evidence / Verification |
| :--- | :---: | :--- |
| **Uses existing YOLOv8n weights** | **Satisfied** | Reuses [yolov8n.pt](file:///home/eshwar/code/detection/yolov8n.pt) in repository root. |
| **Approved Python environment reused** | **Satisfied** | Executed under `/home/eshwar/code/interrupt/venv`. |
| **Camera defaults to `/dev/video0` and configurable** | **Satisfied** | Configured via `--camera`, defaulting to `/dev/video0`. |
| **Only COCO person-class detections returned** | **Satisfied** | Filtered by `class_id == 0`; verified in unit tests. |
| **Confidence filtering configurable & validated** | **Satisfied** | Tested with custom thresholds and boundary conditions. |
| **Detector reusable independently of demo** | **Satisfied** | [person_detector.py](file:///home/eshwar/code/detection/perception/detection/person_detector.py) imported and verified independently. |
| **Live demo displays bounding boxes & confidence** | **Satisfied** | Rendered via OpenCV with text plates in [run_detection_demo.py](file:///home/eshwar/code/detection/perception/detection/run_detection_demo.py). |
| **Live demo displays inference latency & App FPS** | **Satisfied** | Top-left HUD displays instantaneous and average latency and FPS. |
| **Model & camera initialization failures handled** | **Satisfied** | Clear `FileNotFoundError`, `ValueError`, and `RuntimeError` raised. |
| **Shutdown releases camera & display resources** | **Satisfied** | Protected by `finally` blocks releasing camera and destroying windows. |
| **Focused automated tests pass** | **Satisfied** | All 26 unit tests passed in 0.64s. |
| **Manual demonstration repeatable** | **Satisfied** | Exact CLI commands documented and verified. |
| **Performance baseline & limitations reported** | **Satisfied** | 60-frame hardware benchmark recorded (88.5 ms median latency, 10.5 FPS steady). |
| **No excluded M2+ functionality introduced** | **Satisfied** | No tracking IDs, pose estimation, distance estimation, or FSM introduced. |

---

## 10. Confirmation of Milestone Boundary

> [!IMPORTANT]
> **Boundary Confirmation**: No functionality belonging to Milestone M2 (optimization/TensorRT), M3 (tracking), M4 (pose/spatial estimation), or subsequent milestones has been introduced. Implementation has stopped strictly at Milestone M1.

