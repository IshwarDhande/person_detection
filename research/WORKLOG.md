# Worklog

Add meaningful engineering sessions at the top when a worklog update is requested or required by the project convention. Record facts, actual commands/tests, outcomes, and the next concrete action. Do not invent execution results.

## 2026-11-17 — Implementation strategy reorganized into incremental milestones

### Objective
Reorder implementation to produce small, demonstrable increments, starting with basic person detection on the laptop webcam and increasing complexity progressively.

### Decisions recorded
- Use the laptop built-in webcam for initial software development.
- Defer robot camera integration and robot-specific physical calibration until the deployment milestone.
- Implement one milestone at a time and require a working demo, relevant tests, evaluation evidence, and documented limitations before marking it complete.
- Sequence: M1 basic person detection; M2 reliable single-person detection; M3 single-person tracking; M4 frame-level readiness perception; M5 temporal behaviour estimation; M6 complete single-person pipeline; M7 multi-person detection/tracking; M8 multi-person readiness/selection; M9 lifecycle integration; M10 robot deployment/calibration.
- Split readiness development into frame-level Part A and temporal Part B.
- Do not automatically proceed to the next milestone without review/authorization.

### Immediate next action
Implement M1 only: live laptop webcam person detection with bounding box, confidence, FPS, inference latency, and clean shutdown.

### Validation
This entry records a planning decision only. It does not claim that M1 or any later milestone has been implemented or tested.

## 2026-11-17 — M0: Camera Capture Utility and Framing Inspection

### Objective
Implement safe camera image acquisition tooling and inspect the development camera framing.

### Work performed (as previously reported)
- Implemented `tools/calibration/camera_capture_utility.py` supporting stream diagnostics, scene inspection, and intrinsic calibration pattern capture.
- Configured capture for the reported development-camera mode: MJPG, 1280×720, approximately 30 FPS.
- Reported a diagnostic stream test and an empty-scene capture.
- Prepared calibration usage and framing documentation.

### Qualification
The camera discovery/capture results refer to the camera used for laptop-based development. They do not establish the identity, stream capabilities, or geometry of the eventual robot camera. The reported feet-visibility observation applies only to the inspected development setup. Do not infer robot distance accuracy or a precise floor visibility cutoff from it.

### Open items
- Capture a person at controlled distances using the development webcam if useful for software prototyping.
- Collect calibration-target images if intrinsic calibration is needed for the development camera.
- Do not treat laptop calibration as a substitute for M10 robot calibration.

## 2026-11-16 — Camera hardware discovery

### Objective
Identify and characterize the camera used during initial development and verify the nominal stream configuration.

### Work performed (as previously reported)
- Implemented `tools/calibration/camera_hardware_discovery.py`.
- Reported the primary RGB capture device as a Luxvisions Innotech Integrated RGB Camera.
- Reported a 1280×720 MJPG stream at approximately 30 FPS.
- Reported that YUYV at 720p was limited to approximately 10 FPS.

### Qualification
These are development-camera findings from the prior discovery session. Robot hardware remains to be independently identified and tested.

## 2026-11-16 — Python runtime environment configured

### Objective
Select and configure the Python execution environment for the project.

### Work performed (as previously reported)
- Selected `/home/eshwar/code/interrupt/venv` (Python 3.12.3) containing required dependencies: Ultralytics, MediaPipe, PyTorch, OpenCV, and SciPy.
- Created runtime configuration and activation helpers.

### Qualification
Verify that this environment path and dependency versions remain valid before relying on them in a new implementation task.

## 2026-11-16 — Governance baseline prepared

### Objective
Establish repository documentation and engineering workflow.

### Work performed
- Consolidated project scope and behavioural requirements.
- Prepared project-entry documents, AI-agent instructions, architecture, repository map, and engineering standards.
- Recorded initial engineering decisions.
