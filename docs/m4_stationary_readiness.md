# M4 — Stationary Readiness Perception Specification

**Filename:** `m4_stationary_readiness.md`  
**Project:** Vision-Based First Contact  
**Milestone:** M4 — Frame-Level Readiness Perception  
**Status:** DESIGN FROZEN — READY FOR IMPLEMENTATION

---

## 1. Objective

M4 implements a frame-level perception module that extracts relevant person-related observations from an individual RGB camera frame.

The module determines:

1. Whether a person is detected.
2. Whether the person's bounding box is geometrically valid.
3. Whether the person's face is sufficiently visible.
4. Whether the person is inside the configured image-space interaction zone.
5. Whether the person's head is approximately oriented towards the robot.

M4 produces structured observations with confidence, validity, and uncertainty information.

**Important scope distinction:** Despite the filename, M4 does not determine whether a person is temporally stationary, approaching, or ready for interaction. Those require temporal observations and belong to subsequent milestones.

## 2. Scope and Boundaries

### 2.1 Included

- Current-frame person detection.
- Person bounding-box validation.
- Face detection and visibility assessment.
- Facial landmark extraction.
- Approximate head-orientation classification.
- Image-space interaction-zone membership.
- Structured observation output.
- Confidence and validity reporting.
- Unit testing and performance evaluation.

### 2.2 Explicitly excluded

- Body-pose estimation.
- Face recognition or identity matching.
- Appearance embeddings or person re-identification.
- Facial expression or emotion recognition.
- Dedicated eye-gaze estimation.
- Mouth-opening or speaking detection.
- Temporal approach or stationary behaviour estimation.
- Final first-contact readiness decision.
- Metric distance estimation or physical interaction-zone calibration.
- Multi-person target selection and lifecycle management.

M4 consumes the existing person detection pipeline and does not replace M1, M2, or M3.

## 3. Input and Camera Assumptions

### Input

- One RGB camera frame.
- Current-frame YOLO person detection results.
- Configured interaction-zone polygon.
- Configured perception thresholds.

### Development camera

| Property | Value |
|---|---|
| Resolution | 1280 × 720 |
| Expected frame rate | 30 FPS |
| Camera type | RGB |
| Development camera | Laptop webcam |
| Camera movement | Fixed |
| Final deployment camera | Embedded robot camera |

The interaction-zone configuration must be recalibrated when the actual robot camera geometry or framing changes.

## 4. Perception Architecture

```text
                  RGB Frame
                      |
                      v
             YOLO Person Detector
                      |
                      v
             Person Detection Result
                      |
             +--------+--------+
             |                 |
             v                 v
       BBox Validation    Person Crop
             |                 |
             v                 v
       Interaction Zone  MediaPipe Face
          Membership       Landmarker
                               |
                    +----------+----------+
                    |                     |
                    v                     v
              Face Visibility       Facial Landmarks
                    |                     |
                    |                     v
                    |              Head Orientation
                    |                     |
                    +----------+----------+
                               |
                               v
                     Observation Aggregation
                               |
                               v
                   Structured M4 Observation
```

### 4.1 Person Detection

Use the existing YOLO person detector.

- Person class: existing class ID 0.
- Initial detection confidence threshold: existing M1/M2 configuration (0.5).
- Use current-frame detections only.
- Do not use a Kalman-predicted or coasting M3 bounding box as evidence of current person presence.
- Do not require an M3 `CONFIRMED` track for M4 person presence.

### 4.2 Face Detection and Landmark Extraction

Use **MediaPipe Face Landmarker** as the single additional face-analysis component.

It provides face detection and facial landmark information.

Requirements:

- Run face analysis on the detected person's image crop.
- Run on every frame in which YOLO provides an eligible person detection.
- Do not introduce a separate face detector or separate landmark model.
- Use facial landmarks for approximate orientation estimation.
- Do not perform face recognition or identity matching.

The initial minimum face dimensions for quality assessment are:

- Width: 40 pixels.
- Height: 40 pixels.

The MediaPipe confidence threshold must be configurable. Use the selected model's default threshold as the initial value unless explicitly configured otherwise.

## 5. Observation Conditions

### Condition 1 — Person Presence

Determine whether a person is detected in the current frame.

States:

| Value | Meaning |
|---|---|
| `DETECTED` | At least one valid current-frame person detection passes the configured confidence threshold. |
| `NOT_DETECTED` | Detection completed successfully but no eligible person was detected. |
| `UNKNOWN` | Detection was unavailable or failed. |

A predicted M3 track does not count as current-frame person detection.

### Condition 2 — Person Bounding-Box Validity

Validate the detected bounding box geometrically.

Input:

\[
B=(x_1,y_1,x_2,y_2)
\]

Validation requirements:

1. All coordinates must be finite.
2. \(x_2>x_1\).
3. \(y_2>y_1\).
4. The bounding box must intersect the image frame.

Calculate:

\[
W_b=x_2-x_1
\]

\[
H_b=y_2-y_1
\]

\[
A_b=W_bH_b
\]

\[
A_{\text{ratio}}=\frac{A_b}{W_{\text{image}}H_{\text{image}}}
\]

Preserve:

- Original detector bounding box.
- Clipped bounding box for image-based calculations, when required.
- Bounding-box area ratio.

Do not introduce arbitrary minimum or maximum person-box size thresholds in this milestone.

A geometrically valid bounding box does not establish that the entire person is visible.

### Condition 3 — Face Visibility and Quality

Use MediaPipe Face Landmarker to assess whether the person's face is sufficiently observable.

States:

| Value | Meaning |
|---|---|
| `SUFFICIENT` | Face detected with adequate quality for the intended observations. |
| `INSUFFICIENT` | Face analysis completed successfully, but no sufficiently observable face was found. |
| `UNKNOWN` | Face analysis could not be evaluated reliably. |

Initial face-quality requirements:

- Minimum face dimensions: 40 × 40 pixels.
- MediaPipe detection confidence.
- Landmark availability and quality.
- Geometric consistency of available landmarks.

Do not treat face detection alone as proof of sufficient landmark quality.

### Condition 4 — Image-Space Interaction-Zone Membership

Define a configurable polygon representing the image-space interaction region.

The polygon is stored in a YAML configuration file using normalized coordinates.

For image width \(W\) and height \(H\):

\[
x_n=\frac{x}{W},\qquad y_n=\frac{y}{H}
\]

Example configuration:

```yaml
interaction_zone:
  enabled: true
  coordinate_system: normalized
  vertices:
    - [0.23, 0.28]
    - [0.77, 0.28]
    - [0.90, 0.90]
    - [0.10, 0.90]
```

The coordinates above are illustrative and must be calibrated using the actual development camera image.

#### Membership rule

A person is considered inside the interaction zone only when the entire person bounding box lies inside the configured polygon.

\[
\text{INSIDE}\iff B\subseteq P
\]

\[
\text{OUTSIDE}\iff B\nsubseteq P
\]

States:

| Value | Meaning |
|---|---|
| `INSIDE` | Entire person bounding box lies inside the polygon. |
| `OUTSIDE` | Bounding box is not completely contained within the polygon. |
| `UNKNOWN` | Bounding box or interaction-zone configuration is invalid or unavailable. |

Do not use only the bounding-box centre or face centre for this condition.

The containment calculation must correctly evaluate the complete rectangle against the configured polygon.

Important:

- This is image-space membership, not metric distance.
- No physical distance estimation is performed.
- A person partially entering the polygon remains `OUTSIDE`.
- Initial polygon configuration is manual.
- Physical robot-camera calibration is deferred.

### Condition 5 — Robot-Facing Head Orientation

Estimate approximate head orientation using facial landmarks from MediaPipe Face Landmarker.

No separate head-pose estimation model will be introduced.

The objective is to distinguish approximately frontal faces from clearly turned faces.

States:

| Value | Meaning |
|---|---|
| `FACING_ROBOT` | Facial landmark geometry indicates an approximately frontal face. |
| `NOT_FACING_ROBOT` | Geometry indicates a clearly turned face. |
| `UNKNOWN` | Face or landmark evidence is insufficient or ambiguous. |

Requirements:

- Use facial landmark geometry.
- Preserve uncertainty.
- Do not force a binary decision when evidence is ambiguous.
- Determine orientation thresholds experimentally.
- Keep thresholds configurable.
- Do not interpret this as a precise head-pose measurement.

The classification is an approximate orientation indicator, not a calibrated estimate of head yaw.

The fixed robot-mounted camera's viewing direction serves as the initial reference for approximate robot-facing orientation.

## 6. Common Observation Contract

Every observation must expose three common fields:

| Field | Description |
|---|---|
| `value` | Estimated property value. |
| `confidence` | Model-provided confidence when available. |
| `status` | Evaluation validity and uncertainty. |

### 6.1 Status values

| Status | Meaning |
|---|---|
| `VALID` | Property was evaluated successfully with sufficient evidence. |
| `UNCERTAIN` | Evaluation succeeded, but available evidence is ambiguous. |
| `INVALID` | Property could not be evaluated. |

`UNKNOWN` is a property value, not a replacement for the status field.

### 6.2 Confidence semantics

- Use model-provided confidence wherever available.
- Use `null` when a meaningful model confidence is unavailable.
- Do not invent probability values for deterministic geometric calculations.
- Do not claim that heuristic orientation scores are calibrated probabilities.

Confidence and validity must remain separate.

### 6.3 Examples

Successful person detection:

```python
{
    "value": "DETECTED",
    "confidence": 0.91,
    "status": "VALID"
}
```

Successful face inference, but no face found:

```python
{
    "value": "INSUFFICIENT",
    "confidence": None,
    "status": "VALID"
}
```

Face inference failure:

```python
{
    "value": "UNKNOWN",
    "confidence": None,
    "status": "INVALID"
}
```

Ambiguous orientation:

```python
{
    "value": "UNKNOWN",
    "confidence": None,
    "status": "UNCERTAIN"
}
```

## 7. Failure and Uncertainty Handling

The implementation must distinguish between:

1. A valid negative observation.
2. An ambiguous observation.
3. An observation that could not be evaluated.

Examples:

| Situation | Expected result |
|---|---|
| Detector runs, no person found | `NOT_DETECTED`, `VALID` |
| Person detector unavailable | `UNKNOWN`, `INVALID` |
| Face model runs, no face found | `INSUFFICIENT`, `VALID` |
| Face model execution fails | `UNKNOWN`, `INVALID` |
| Face detected but too small | `UNKNOWN` or insufficient face quality, according to the face observation contract |
| Landmarks ambiguous | `UNKNOWN`, `UNCERTAIN` |
| Invalid interaction polygon | `UNKNOWN`, `INVALID` |
| Invalid bounding box | Bounding-box observation `INVALID`; dependent observations must not produce fabricated results |

Failures in one observation should not automatically invalidate unrelated observations when those observations can still be evaluated independently.

## 8. Output Structure

Use a nested structured observation rather than one flat dictionary or an unnecessarily large hierarchy of classes.

The logical structure should group observations into:

```text
FrameObservation
|
+-- person
|   +-- presence
|   +-- bounding_box
|   +-- interaction_zone
|
+-- face
|   +-- visibility
|   +-- quality
|
+-- orientation
|   +-- head_orientation
|
+-- frame_metadata
    +-- timestamp
    +-- image_width
    +-- image_height
```

Each observation follows the common `value`, `confidence`, and `status` contract.

The final Python schema and exact field names must be defined consistently during implementation.

The output must preserve the distinction between:

- No detection.
- Invalid input.
- Uncertain inference.
- Valid negative observation.

## 9. Configuration Management

Use YAML for configurable parameters.

The configuration should cover:

- Person detection threshold, reusing existing configuration where possible.
- MediaPipe model settings.
- MediaPipe confidence threshold.
- Minimum face dimensions (initially 40 × 40 pixels).
- Orientation classification thresholds.
- Interaction-zone polygon.
- Relevant image-coordinate settings.

Avoid duplicating existing configuration values unnecessarily.

All thresholds should have documented units and meanings.

## 10. Testing Requirements

M4 testing is limited to unit tests using synthetic and mocked inputs.

### 10.1 Person detection and bounding-box tests

Test:

- Person detected.
- No person detected.
- Detector unavailable.
- Valid bounding box.
- Non-finite coordinates.
- Zero or negative width/height.
- Bounding box outside the frame.
- Bounding box partially intersecting the frame.
- Original and clipped bounding-box preservation.
- Area-ratio calculation.

### 10.2 Face observation tests

Test:

- Face detected.
- No face detected.
- Face model failure.
- Face smaller than 40 × 40 pixels.
- Valid face landmarks.
- Missing or unusable landmarks.
- Insufficient face quality.
- Uncertain face observation.

### 10.3 Orientation tests

Test:

- Approximately frontal face.
- Clearly turned face.
- Ambiguous orientation.
- Missing landmarks.
- Insufficient face quality.
- Configurable threshold behaviour.

### 10.4 Interaction-zone tests

Test:

- Entire bounding box inside polygon.
- Bounding box outside polygon.
- Bounding box crossing polygon boundary.
- Bounding box partially inside polygon.
- Invalid polygon.
- Missing polygon configuration.
- Normalized coordinate conversion.
- Different supported image dimensions.

### 10.5 Observation-contract tests

Test:

- Valid observation.
- Uncertain observation.
- Invalid observation.
- Unknown property value.
- Null confidence.
- Correct distinction between valid negative results and execution failures.

## 11. Performance Evaluation

Measure the performance of individual components and the complete M4 pipeline.

Record:

- YOLO inference latency.
- MediaPipe Face Landmarker latency.
- Bounding-box and interaction-zone calculation latency.
- Orientation post-processing latency.
- Total pipeline latency.
- p50, p95 and p99 latency.
- Application FPS.
- CPU utilization.
- Memory consumption.

Use the same input frames and test environment when comparing component and pipeline measurements.

Do not assume that the expected 30 FPS camera rate is achieved merely because the camera supplies 30 FPS.

No specific latency target is frozen in this specification. Establish the baseline through measurements before optimization.

## 12. Demonstration Requirements

Provide a demonstration that displays:

- Person bounding box.
- Face bounding box, when available.
- Configured interaction polygon.
- Person's interaction-zone state.
- Face visibility state.
- Head-orientation state.
- Confidence and validity information.

Use clear visual indicators for `VALID`, `UNCERTAIN`, and `INVALID`.

The demonstration must not imply that an image-space zone represents physical distance.

## 13. Acceptance Criteria

M4 is complete when:

1. The five agreed observations are implemented.
2. Existing YOLO detection is reused.
3. MediaPipe Face Landmarker provides face detection and landmarks.
4. Face analysis runs on every eligible person crop.
5. Bounding-box validation follows the defined geometric rules.
6. Interaction-zone membership requires the entire bounding box to be inside the configured polygon.
7. Approximate head orientation supports all three states.
8. Every observation follows the common output contract.
9. Confidence, validity and uncertainty are preserved.
10. Configuration is maintained in YAML.
11. Unit tests cover valid, invalid and uncertain scenarios.
12. Component-level and total pipeline performance measurements are available.
13. The demonstration displays the resulting frame-level observations.

## 14. Implementation Constraints

- Reuse existing M1, M2 and M3 components.
- Do not modify frozen predecessor milestone behaviour.
- Do not add body-pose estimation.
- Do not add face recognition, identity matching or appearance embeddings.
- Do not add dedicated gaze or facial-expression models.
- Do not implement temporal approach or stationary behaviour.
- Do not implement final first-contact readiness logic.
- Avoid unnecessary dependencies and architectural changes.
- Keep face analysis modular.
- Preserve the distinction between raw detections and derived observations.
- Do not optimize before establishing performance measurements.

## 15. Milestone Boundary

M4 produces a structured frame-level observation.

It does not decide whether the robot should initiate first contact.

The output will provide the perception inputs required by subsequent milestones, where temporal behaviour and readiness decisions can be developed separately.

---

**Final status: M4 design frozen and ready for implementation.**