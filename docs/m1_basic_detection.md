# M1 --- Basic Person Detection

## Formal Design Specification

**Project:** Vision-Based First Contact for a Reception Social Robot\
**Milestone:** M1 --- Basic Person Detection\
**Development stage:** Stage A --- Laptop webcam prototype\
**Status:** Design approved; implementation not yet started\
**Document purpose:** Define the scope, interfaces, runtime behaviour,
validation strategy, and exit criteria for the first executable
person-detection milestone.

------------------------------------------------------------------------

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

------------------------------------------------------------------------

## 1. Purpose

M1 establishes the first working perception capability in the project:
detecting people in a live video stream from the development laptop's
built-in webcam.

The milestone converts the existing YOLOv8n model weights into a small,
reusable detection component and a standalone live demonstration. It
establishes a measurable baseline for detection correctness, inference
latency, and end-to-end throughput.

M1 is deliberately limited to frame-level person detection. It does not
attempt to understand a person's identity, movement, intent, or
readiness to interact.

## 2. Context and Existing Assets

The implementation audit reports the following existing assets:

  ---------------------------------------------------------------------------------------------------
  Asset                                              Existing state           M1 treatment
  -------------------------------------------------- ------------------------ -----------------------
  `tools/calibration/camera_hardware_discovery.py`   Camera discovery and     Retain; do not rewrite
                                                     streaming benchmark      
                                                     utility                  

  `tools/calibration/camera_capture_utility.py`      Scene/calibration frame  Retain; do not rewrite
                                                     acquisition utility      

  `yolov8n.pt`                                       YOLOv8 nano weights      Reuse
                                                     present in repository    
                                                     root                     

  `/dev/video0`                                      Laptop development       Default camera device
                                                     camera                   

  `/home/eshwar/code/interrupt/venv`                 Environment used by      Reuse, subject to
                                                     prior camera/model       dependency verification
                                                     validation               

  Camera inventory and calibration reports           Existing                 Retain as reference
                                                     documentation/evidence   
  ---------------------------------------------------------------------------------------------------

The reported camera benchmark is 1280×720 MJPG at approximately 29.8
FPS. This is a camera-streaming measurement, not a guarantee of detector
throughput. M1 must measure actual application performance
independently.

The laptop camera is a development device. Its settings and observations
must not be treated as robot-camera calibration.

## 3. Approved Design Decisions

The following decisions were confirmed for M1:

1.  **Python environment:** Reuse `/home/eshwar/code/interrupt/venv`.
    Verify that required packages are available before implementation;
    do not create another environment as part of M1.
2.  **Detection model:** Reuse the repository's `yolov8n.pt` weights
    through the Ultralytics API. Do not conduct a model bake-off or
    replace the model during this milestone.
3.  **Camera selection:** Default to `/dev/video0`, while allowing the
    camera device to be configured at runtime. Avoid hardcoding the
    device in the detector implementation.
4.  **Visual output:** Display person bounding boxes, confidence values,
    measured FPS, and inference latency.
5.  **Performance policy:** Prioritize correctness and establish a
    reproducible performance baseline. Do not impose a hard 25/30 FPS
    acceptance threshold or undertake optimization in M1.

## 4. Scope

### 4.1 Included

-   Open a configurable camera device; default to `/dev/video0`.
-   Acquire frames continuously.
-   Run YOLOv8n inference on each processed frame.
-   Retain detections belonging only to the COCO `person` class (class
    ID `0`).
-   Apply a configurable confidence threshold.
-   Represent detections using a stable, documented frame-level data
    structure.
-   Render bounding boxes and confidence labels.
-   Display measured FPS and inference latency.
-   Support clean shutdown using `q`, `Esc`, or a normal window-close
    event where supported.
-   Handle model-loading and camera-initialization failures with clear
    errors.
-   Add focused automated tests for detection post-processing and
    configuration behaviour.
-   Provide a repeatable manual live-camera demonstration.

### 4.2 Explicitly excluded

M1 must not implement or imply completion of:

-   Multi-object or single-object tracking; no persistent `track_id`.
-   Trajectory/history storage or temporal smoothing.
-   Pose estimation, face detection, or keypoint extraction.
-   Person orientation or robot-facing estimation.
-   Metric distance estimation, camera calibration, or interaction-zone
    geometry.
-   Approach, movement, stationary, dwell, or readiness estimation.
-   Candidate queues, target selection, or arbitration.
-   Lifecycle state machines or downstream event dispatch.
-   ROS, ZeroMQ, speech, dialogue, or robot actuation integration.
-   Model optimization, ONNX/TensorRT export, or hardware-specific
    acceleration work.

## 5. Proposed Component Design

Keep the implementation small and aligned with the repository's actual
structure. Before creating files, inspect the current tree and follow
existing naming and packaging conventions. The following paths are
proposed targets, not permission to create unrelated architecture or
placeholder directories.

``` text
perception/
└── detection/
    ├── __init__.py
    ├── person_detector.py
    └── run_detection_demo.py

tests/
└── perception/
    └── detection/
        ├── test_person_detector.py
        └── test_detection_config.py
```

If the repository already has a more appropriate package or test layout,
use it rather than creating a competing structure.

### 5.1 `PersonDetector`

**Responsibility:** Encapsulate model loading, inference, person-class
filtering, and conversion into the project's frame-level detection
representation.

Suggested public interface:

``` python
class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        ...

    def detect(self, frame) -> list[PersonDetection]:
        ...

    def close(self) -> None:
        ...
```

Design notes:

-   Use the Ultralytics YOLO API with the existing `yolov8n.pt` weights.
-   Keep model construction separate from frame acquisition and
    visualization.
-   Validate that the model path exists and that the confidence
    threshold is in the valid interval `[0, 1]`.
-   Use class ID `0` for COCO `person`; do not rely on display-name
    matching alone.
-   Apply the configured confidence threshold consistently.
-   Return an empty list when no person is detected.
-   Do not silently convert inference or model errors into an empty
    detection list. Surface errors with useful context.
-   Do not introduce tracking IDs or temporal state into this class.
-   Keep device selection configurable, but do not add device
    auto-tuning or optimization logic in M1.

### 5.2 Frame-level detection representation

Use a small typed structure (for example, a dataclass) with the
following semantic fields:

  -----------------------------------------------------------------------
  Field                               Meaning
  ----------------------------------- -----------------------------------
  `bbox_xyxy`                         Bounding box `(x1, y1, x2, y2)` in
                                      input-frame pixel coordinates

  `confidence`                        Model confidence for the person
                                      detection

  `class_id`                          COCO class ID; must be `0` for
                                      returned records
  -----------------------------------------------------------------------

Requirements:

-   Coordinates must be finite and expressed in the coordinate system of
    the input frame.
-   Confidence must be finite and within `[0, 1]`.
-   The representation is per-frame only. It must not contain a
    persistent identity or history.
-   Keep conversion from the Ultralytics result object isolated so it
    can be unit-tested without a camera.

### 5.3 Live demonstration runner

**Responsibility:** Own the camera and display loop; call the detector
and render its outputs.

Suggested command-line options:

  -----------------------------------------------------------------------
  Option                  Default                 Purpose
  ----------------------- ----------------------- -----------------------
  `--camera`              `/dev/video0`           Camera device path or
                                                  supported OpenCV camera
                                                  index

  `--model`               Repository `yolov8n.pt` Model weights path

  `--conf`                `0.5`                   Person confidence
                                                  threshold

  `--width`               `1280`                  Requested capture width

  `--height`              `720`                   Requested capture
                                                  height

  `--fps`                 `30`                    Requested camera FPS
  -----------------------------------------------------------------------

The capture width, height, and FPS are requests to the camera backend,
not guarantees. Log or display the actual frame dimensions where useful.
Do not fail solely because the camera negotiates a different FPS.

The runner should:

1.  Parse and validate configuration.
2.  Load the detector.
3.  Open the requested camera and verify that frames can be read.
4.  Read frames continuously.
5.  Invoke `PersonDetector.detect(frame)`.
6.  Draw each returned bounding box and confidence value.
7.  Display FPS and inference latency.
8.  Exit cleanly on `q`, `Esc`, or window closure.
9.  Release the camera and destroy display windows in a `finally`-style
    cleanup path.

The runner should not contain model-specific parsing logic; that belongs
in `PersonDetector`.

## 6. Runtime Metrics and Definitions

M1 records two distinct performance quantities.

### 6.1 Inference latency

For frame (t), measure the elapsed wall-clock time spent inside the
detector call:

\[ L_t = t\^{`\mathrm{end}`{=tex}}\_t - t\^{`\mathrm{start}`{=tex}}\_t
\]

Report latency in milliseconds. Use a monotonic high-resolution clock.
Be explicit about whether the selected inference API includes
preprocessing and postprocessing; the reported value should cover the
complete `detect()` call.

### 6.2 Application FPS

Measure the observed processing-loop throughput rather than displaying
the camera's requested FPS:

\[ `\mathrm{FPS}`{=tex} = `\frac{N-1}{\tau_N-\tau_1}`{=tex} \]

where (`\tau`{=tex}\_i) are monotonic timestamps for completed loop
iterations. Use a rolling measurement window or smoothed display value
to avoid misleading frame-to-frame fluctuations. Do not label requested
camera FPS as measured application FPS.

### 6.3 Baseline evidence

Record at least:

-   Model name and weights path.
-   Confidence threshold.
-   Camera device and negotiated frame dimensions.
-   Runtime environment and relevant package versions.
-   Number of processed frames and test duration.
-   Mean and percentile inference latency where enough samples exist.
-   Mean observed application FPS.
-   Any camera read failures, inference errors, or observed instability.

M1 is a baseline, not an optimization exercise. Report measurements as
observed on the development machine and test conditions.

## 7. Error Handling and Resource Management

  -----------------------------------------------------------------------
  Failure condition                   Required behaviour
  ----------------------------------- -----------------------------------
  Model weights missing or unreadable Fail at initialization with a clear
                                      message

  Ultralytics/model initialization    Surface a contextual initialization
  failure                             error

  Camera cannot be opened             Fail clearly; do not enter an
                                      infinite read loop

  Camera read fails                   Report the failure and terminate or
                                      use a small, explicit bounded retry
                                      policy; never spin indefinitely

  Invalid confidence threshold        Reject configuration before
                                      inference

  Inference failure                   Surface/log the error; do not
                                      misrepresent it as zero detections

  User exits or display closes        Release camera and close windows

  Exception during loop               Execute resource cleanup before
                                      propagating/reporting the exception
  -----------------------------------------------------------------------

Avoid broad exception swallowing. Resource cleanup must be reliable even
when inference or rendering fails.

## 8. Testing and Validation Strategy

M1 has two complementary validation paths.

### 8.1 Automated tests

Tests should not require a physical webcam or download model weights
during execution.

At minimum, cover:

-   Person-class detections are retained.
-   Non-person classes are filtered out.
-   Detections below the confidence threshold are filtered out.
-   Threshold boundary behaviour is defined and tested.
-   Empty model results produce an empty detection list.
-   Bounding-box coordinates and confidence values are converted
    correctly.
-   Invalid threshold values are rejected.
-   Missing model path is reported clearly.
-   Camera initialization failure is handled by the runner's camera
    abstraction or a focused test seam, if practical.

Use mocked/fake Ultralytics outputs for post-processing tests. Do not
make tests depend on nondeterministic live inference.

### 8.2 Manual live-camera demonstration

Run the standalone demonstration using the laptop's built-in webcam.

Demonstrate:

1.  Empty scene: no person box is drawn.
2.  One person enters: a bounding box and confidence label appear.
3.  Person moves within the frame: the box follows the current
    frame-level detection.
4.  Person leaves: the box disappears when no person is detected.
5.  Exit using `q` or `Esc`: camera and display resources are released.

The demonstration does not establish tracking. A box following a moving
person across successive frames is simply repeated frame-level
detection.

### 8.3 Performance baseline

Measure actual inference latency and application FPS under the
development environment. No hard FPS threshold is an M1 acceptance
condition. If performance is unexpectedly poor, document the result and
likely measurement conditions; do not expand M1 into model optimization
without a separate decision.

## 9. M1 Acceptance Criteria

M1 is complete only when all of the following are satisfied:

-   [ ] The implementation uses the existing YOLOv8n weights.
-   [ ] The approved Python environment is reused and dependencies are
    verified.
-   [ ] The camera defaults to `/dev/video0` and can be configured.
-   [ ] Only COCO person-class detections are returned.
-   [ ] Confidence filtering is configurable and validated.
-   [ ] The detector is reusable independently of the live demo.
-   [ ] The live demo displays bounding boxes and confidence values.
-   [ ] The live demo displays measured inference latency and
    application FPS.
-   [ ] Model and camera initialization failures are handled clearly.
-   [ ] Shutdown releases camera and display resources.
-   [ ] Focused automated tests pass.
-   [ ] The manual demonstration is repeatable.
-   [ ] Performance baseline and known limitations are reported.
-   [ ] No excluded M2+ functionality has been introduced.

A source file existing is not sufficient evidence of completion.
Completion requires working behaviour, tests, a repeatable
demonstration, and recorded measurements.

## 10. Known Limitations

-   Detection quality depends on the pretrained model, camera image
    quality, lighting, occlusion, pose, and distance.
-   M1 does not establish reliable detection across all reception
    conditions.
-   Frame-level detections do not provide persistent person identity.
-   Bounding boxes do not provide calibrated physical distance.
-   No claim is made about robot-facing orientation or first-contact
    readiness.
-   The laptop webcam benchmark does not establish performance on the
    robot's eventual camera or compute hardware.

## 11. Implementation Constraints

-   Inspect the actual repository tree and current Git working-tree
    state before editing.
-   Reuse existing camera utilities; do not rewrite them.
-   Reuse the existing Python environment; do not create a second
    environment.
-   Keep the change set limited to M1 source, focused tests, and only
    essential run instructions.
-   Do not conduct a repository-wide audit or refactor.
-   Do not modify unrelated governance documents or create broad
    planning reports.
-   Do not delete or overwrite existing assets.
-   Do not commit changes.
-   Do not begin M2 or any later milestone.
-   At completion, report changed files, commands run, test results,
    demo instructions, measured performance, and known limitations.
-   Stop after M1 and wait for review before proceeding.

------------------------------------------------------------------------

## 12. Definition of Done

M1 delivers a minimal, reusable, tested person-detection capability and
a live laptop-webcam demonstration. It establishes trustworthy
frame-level detection and performance evidence while leaving tracking,
spatial reasoning, temporal behaviour, readiness, multi-person
selection, and robot deployment for their explicitly assigned later
milestones.
