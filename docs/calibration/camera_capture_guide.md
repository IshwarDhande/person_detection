# Camera Calibration Capture Utility — User Guide

> **Document:** `camera_capture_guide.md`  
> **Tool:** `tools/calibration/camera_capture_utility.py`  
> **Target Subsystem:** Spatial Understanding & Perception Calibration (M0 Milestone)  
> **Execution Environment:** `/home/eshwar/code/interrupt/venv` (Python 3.12.3, OpenCV 5.0.0)  

---

## 1. Overview & Hardware Topology

The `camera_capture_utility.py` script provides safe, deterministic camera acquisition and calibration image collection for the reception social robot.

```text
+------------------------------------------------------------------------------------+
|                         CAMERA CALIBRATION ACQUISITION SYSTEM                      |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  Tabletop Robot Head Mounting:                                                     |
|    - Lens Height: 115.0 cm above floor                                             |
|    - Optical Pitch: ~0.0° (Horizontal forward-facing optical axis)                  |
|    - Optical Yaw / Roll: ~0.0°                                                     |
|    - Physical Framing Observation: Feet/floor occluded by desk at <= 2.5 m         |
|                                                                                    |
|  Hardware Streaming Configuration:                                                 |
|    - Device Path: /dev/video0 (Luxvisions Innotech Integrated RGB Camera)          |
|    - Required Format: MJPG (Motion-JPEG)                                           |
|    - Verified Mode: 1280 × 720 @ 30 FPS (Observed ~29.7 FPS, 0 drops)              |
|                                                                                    |
|  Acquisition Modes:                                                                |
|    [ test ]      ==> Diagnostic verification & FPS benchmark (zero disk writes)    |
|    [ scene ]     ==> Mode A: Framing at empty, 1.0 m, 1.5 m, 2.0 m, 2.5 m          |
|    [ intrinsic ] ==> Mode B: Multi-pose Checkerboard / ChArUco pattern acquisition |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

---

## 2. Quickstart & Diagnostic Mode

Run a diagnostic test without persisting any files to disk:

```bash
# Activate runtime environment
source activate_env.sh

# Run streaming benchmark (30 frames)
python3 tools/calibration/camera_capture_utility.py --mode test
```

### Diagnostic Output Checks
The utility validates:
* Successful V4L2 opening of `/dev/video0`.
* Enforcement of `CAP_PROP_FOURCC = 'MJPG'`.
* Returned buffer dimensions: exactly `1280 × 720 × 3` (`uint8`).
* Measured framerate ($\ge 25.0$ FPS) and inter-frame timing jitter.
* Clean camera handle release upon completion.

---

## 3. Mode A — Scene Inspection Capture

Mode A captures standardized reference frames across the reception interaction zone.

### Supported Stages
1. `empty` — Background reference of the empty reception desk and foyer.
2. `person_1_0m` — Subject standing at measured 1.0 m mark (close proximity).
3. `person_1_5m` — Subject standing at measured 1.5 m mark (**first-contact decision boundary**).
4. `person_2_0m` — Subject standing at measured 2.0 m mark (approach zone).
5. `person_2_5m` — Subject standing at measured 2.5 m mark (detection boundary).

### Privacy & Consent Policy
Per project engineering standards:
* **Consent required:** All capture involving human subjects requires the `--consent` flag.
* **No biometric retention:** The pipeline does not store facial crops, embeddings, or biometric templates. Images are strictly used for bounding box and keypoint height validation.

### Commands

```bash
# Capture empty reception background:
python3 tools/calibration/camera_capture_utility.py \
  --mode scene \
  --stage empty \
  --save

# Capture 1.5 m first-contact boundary with consent:
python3 tools/calibration/camera_capture_utility.py \
  --mode scene \
  --stage person_1_5m \
  --consent \
  --notes "Subject standing centered facing robot at measured 1.50 m tape mark" \
  --save

# Execute all stages sequentially in dry-run mode (preview without saving):
python3 tools/calibration/camera_capture_utility.py \
  --mode scene
```

---

## 4. Mode B — Intrinsic Calibration Capture

Mode B captures planar calibration targets (Checkerboard or ChArUco) across multiple orientations to estimate camera matrix $K$ and lens distortion coefficients $D$.

### Calibration Target Specifications
* **Checkerboard (Default):** 9 × 6 inner corners ($10 \times 7$ squares), 25.0 mm square size.
* **ChArUco Board:** 9 × 6 grid, 25.0 mm square size, 15.0 mm marker size (`DICT_4X4_50`).

### Board Positioning Protocol
To avoid degenerate calibration solutions:
1. **Quadrant Coverage:** Capture images across all four quadrants (top-left, top-right, bottom-left, bottom-right) and center.
2. **Angular Variation:** Tilt the board along pitch ($\pm 20^\circ$) and yaw ($\pm 20^\circ$). Do not capture only perpendicular boards.
3. **Distance Variation:** Range from 0.5 m to 2.0 m from the lens.
4. **Motion Blur Filter:** The tool automatically computes Laplacian variance and rejects blurry or moving frames.

```text
+------------------------------------------------------------------------------------+
|                         INTRINSIC CALIBRATION COVERAGE MATRIX                      |
+------------------------------------------------------------------------------------+
|                                                                                    |
|          [Top-Left Quadrant]             |           [Top-Right Quadrant]          |
|          - Close range (0.6 m)           |           - Mid range (1.2 m)           |
|          - Pitch tilted +15 deg          |           - Yaw tilted -15 deg          |
|                                          |                                         |
|  ----------------------------------------+---------------------------------------  |
|                                                                                    |
|          [Bottom-Left Quadrant]          |          [Bottom-Right Quadrant]        |
|          - Mid range (1.0 m)             |           - Close range (0.7 m)         |
|          - Yaw tilted +15 deg            |           - Pitch tilted -15 deg        |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

### Commands

```bash
# Capture 15 checkerboard calibration frames with live preview:
python3 tools/calibration/camera_capture_utility.py \
  --mode intrinsic \
  --pattern checkerboard \
  --board-width 9 \
  --board-height 6 \
  --square-size 25.0 \
  --count 15 \
  --preview \
  --save

# Capture ChArUco board frames:
python3 tools/calibration/camera_capture_utility.py \
  --mode intrinsic \
  --pattern charuco \
  --board-width 9 \
  --board-height 6 \
  --square-size 25.0 \
  --marker-size 15.0 \
  --count 20 \
  --preview \
  --save
```

---

## 5. Output Directory Structure & Metadata

```text
data/raw/calibration/
├── calibration_metadata.json          <--- Master structured JSON index
├── scene_inspection/                  <--- Mode A output
│   ├── scene_empty.png
│   ├── scene_person_1_0m.png
│   ├── scene_person_1_5m.png
│   ├── scene_person_2_0m.png
│   └── scene_person_2_5m.png
└── intrinsics/                        <--- Mode B output
    ├── calib_checkerboard_001_center.png
    ├── calib_checkerboard_002_top_left.png
    └── ...
```

### Metadata JSON Schema
Each entry in `calibration_metadata.json` contains:
```json
{
  "file_name": "scene_person_1_5m.png",
  "relative_path": "raw/calibration/scene_inspection/scene_person_1_5m.png",
  "capture_mode": "scene_inspection",
  "stage": "person_1_5m",
  "nominal_distance_m": 1.5,
  "timestamp_utc": "2026-11-17T08:30:31.552050+00:00",
  "resolution": {"width": 1280, "height": 720, "channels": 3, "dtype": "uint8"},
  "sharpness": 32.5,
  "camera_geometry": {
    "lens_height_cm": 115.0,
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
    "feet_visibility": "not_visible"
  },
  "privacy_status": {
    "consent_verified": true,
    "biometric_identity_retained": false,
    "purpose": "First-contact framing and distance boundary calibration"
  }
}
```

---

## 6. Command-Line Reference

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--mode` | string | `test` | `test` (diagnostic), `scene` (Mode A), `intrinsic` (Mode B) |
| `--device` | string | `/dev/video0` | Video device path or persistent udev symlink |
| `--width` | integer | `1280` | Frame width |
| `--height` | integer | `720` | Frame height |
| `--fps` | integer | `30` | Frame rate |
| `--fourcc` | string | `MJPG` | Video compression fourcc (`MJPG` enforced) |
| `--save` | flag | `False` | Persist captured frames to disk (dry-run if omitted) |
| `--preview` | flag | `False` | Open live OpenCV preview window |
| `--stage` | string | `None` | Scene stage (`empty`, `person_1_0m`, `person_1_5m`, etc.) |
| `--distance` | float | `None` | Numeric distance in meters |
| `--consent` | flag | `False` | Explicit user consent certification for human subjects |
| `--delay` | float | `0.0` | Countdown delay (seconds) to allow operator positioning |
| `--pattern` | string | `checkerboard` | Calibration target: `checkerboard` or `charuco` |
| `--count` | integer | `15` | Number of intrinsic board images to collect |
| `--output-dir` | path | `data/raw/calibration` | Base image destination directory |
| `--metadata-file`| path | `.../calibration_metadata.json` | Path to master JSON metadata index |

