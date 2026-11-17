# Worklog

Add new meaningful engineering sessions at the top. Record facts, actual commands/tests, outcomes, and the next concrete action.

## 2026-11-17 — Repository Cleanup & Governance Synchronization

### Objective
Remove empty directories and synchronize repository governance documentation to reflect the active filesystem structure.

### Work performed
- Recursively identified and safely removed 101 genuinely empty directories using bottom-up directory deletion.
- Preserved all 16 active directories containing files and project assets (`configs/runtime`, `data/raw/calibration`, `docs/calibration`, `research`, `tools/calibration`).
- Confirmed zero files deleted and zero non-empty directories modified.
- Synchronized primary governance map `research/REPOSITORY_MAP.md` and repository overview `research/README.md` to document the active filesystem and clarify just-in-time introduction for planned subsystem packages.
- Updated `research/STATUS.md`.

### Results
Repository tree streamlined from 117 to 13 directories. Zero empty leaf directories remain.

### Open items
Proceed with intrinsic calibration and initial perception module implementations.

### Next action
Collect intrinsic calibration dataset with physical target board.

### Git reference
Not recorded.

## 2026-11-17 — M0: Camera Calibration Capture Utility Implemented & Framing Analyzed

### Objective
Implement safe camera calibration image acquisition tooling, establish metadata tracking, and document framing observations and geometry constraints.

### Work performed
- Implemented `tools/calibration/camera_capture_utility.py` supporting stream diagnostics, Mode A (scene inspection at empty, 1.0m, 1.5m, 2.0m, 2.5m), and Mode B (ChArUco / checkerboard multi-pose acquisition).
- Enforced explicit `MJPG` 1280×720 @ 30 FPS configuration, dry-run by default, fail-closed consent certification for human subjects, and clean signal-based resource release.
- Executed diagnostic validation against `/dev/video0`: verified 29.8 FPS, 30/30 frames in 1.007s, 2.7 ms jitter, exact `1280×720×3 uint8` format, and clean handle release.
- Validated Mode A acquisition: saved empty scene baseline `scene_empty.png` and generated persistent metadata index `data/raw/calibration/calibration_metadata.json`.
- Derived optical ray-tracing proof in `docs/calibration/m0_framing_and_calibration_report.md`: with $h = 1.15\text{ m}$ and $\theta = 0^\circ$, ground contact occurs at $\approx 2.78\text{ m}$, mathematically explaining why feet are occluded at the 1.5 m decision boundary. Formally ruled out feet-based ground homography in favor of upper-body/shoulder keypoint triangulation.
- Authored comprehensive user guide in `docs/calibration/camera_capture_guide.md`.

### Results
Tooling operational, metadata index active, framing constraints proved and architectural direction established.

### Open items
Acquire physical ChArUco / checkerboard pattern dataset; place floor tape marks at 1.0 m, 1.5 m, 2.0 m, 2.5 m for empirical distance function calibration.

### Next action
Run intrinsic calibration session once physical board is presented.

### Git reference
Not recorded.

## 2026-11-16 — M0 Batch 1: Embedded Camera Hardware Discovery Completed

### Objective
Identify and characterize the robot's embedded camera hardware and verify the 1280×720 @ 30 FPS operational assumption.

### Work performed
- Implemented `tools/calibration/camera_hardware_discovery.py` to inspect V4L2, USB, sysfs, and OpenCV interfaces.
- Executed hardware discovery and streaming benchmarks across all `/dev/video*` devices.
- Identified primary RGB camera as Luxvisions Innotech Limited Integrated RGB Camera (`0x30c9:0x00a8`) driven by Linux `uvcvideo` over internal USB 2.0.
- Classified `/dev/video0` as primary RGB capture device and `/dev/video2` as auxiliary 640×360 IR sensor; identified `/dev/video1` and `/dev/video3` as UVC metadata capture nodes.
- Conducted 5-second OpenCV streaming benchmark at 1280×720 @ 30 FPS: measured 29.69 FPS, 0 failed frames, 2.78 ms jitter under MJPG format. Discovered that uncompressed YUYV format is hardware-limited to 10 FPS at 720p due to USB 2.0 bandwidth.
- Generated comprehensive Markdown report `docs/calibration/m0_camera_hardware_inventory.md` and structured JSON inventory `docs/calibration/m0_camera_hardware_inventory.json`.

### Results
Camera hardware identified and 1280×720 @ 30 FPS verified under MJPG compression. Verdict: PASS.

### Open items
M0 Batch 2: Physical desk mounting measurements ($h$, $\theta_{pitch}$) and camera intrinsic calibration (focal length, optical center, distortion).

### Next action
Collect ChArUco / checkerboard frames for intrinsic camera matrix estimation.

### Git reference
Not recorded.

## 2026-11-16 — Python runtime environment configured

### Objective
Select and configure the Python execution environment for the project.

### Work performed
- Audited available virtual environments across pyenv and workspace projects.
- Selected `/home/eshwar/code/interrupt/venv` (Python 3.12.3) containing required dependencies: Ultralytics (8.4.115), MediaPipe (0.10.14), PyTorch (2.13.0), OpenCV (5.0.0), and SciPy (1.18.0).
- Created runtime configuration in `configs/runtime/environment.yaml`.
- Created shell activation script `activate_env.sh` and root `.env`.

### Results
Environment imports verified without errors. Project runtime bound to `/home/eshwar/code/interrupt/venv`.

### Open items
Proceed with pipeline component interface definitions and camera calibration data collection.

### Next action
Define typed observation and tracking schemas.

### Git reference
Not recorded.

## 2026-11-16 — Governance baseline prepared

### Objective
Establish repository documentation and engineering workflow.

### Work performed
- Consolidated project scope and current behavioural requirements.
- Prepared project-entry documents, AI-agent instructions, architecture, repository map, and engineering standards.
- Recorded initial engineering decisions.

### Results
Documentation baseline prepared. No implementation code, runtime dependencies, or model choices are asserted.

### Open items
Confirm actual repository tree; resolve camera calibration, HOLD/timer semantics, and event lifecycle details.

### Next action
Integrate these documents without overwriting existing requirements or research records.

### Git reference
Not recorded.
