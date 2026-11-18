# M2 --- Reliable Single-Person Detection

**Project:** Vision-Based First Contact for a Reception Social Robot\
**Milestone:** M2\
**Status:** Design specification\
**Predecessor:** M1 --- Basic Person Detection\
**Next milestone:** M3 --- Single-Person Tracking

------------------------------------------------------------------------

## 1. Objective

M2 extends the M1 YOLOv8n-based person detector with short-term
perception handling and bounding-box stabilization.

The objective is to make single-person detection outputs more usable and
visually stable under ordinary changes in scene conditions, while
preserving a clear separation between frame-level detection and person
tracking.

M2 remains a **single-person perception milestone**. It does not
introduce persistent person identity, trajectory estimation, or
multi-object tracking.

## 2. Agreed Design Decisions

  -----------------------------------------------------------------------
  ID                      Design area             Agreed decision
  ----------------------- ----------------------- -----------------------
  D1                      Meaning of reliability  Perception reliability:
                                                  improve frame-level
                                                  detection usability and
                                                  add short-term handling
                                                  of detection misses and
                                                  bounding-box stability.

  D2                      Temporary detection     Immediately report the
                          failures                person as not detected
                                                  when a frame has no
                                                  valid person detection.
                                                  Do not retain a
                                                  previous detection
                                                  through a grace period.

  D3                      Bounding-box stability  Apply lightweight
                                                  exponential moving
                                                  average (EMA) smoothing
                                                  to bounding-box
                                                  coordinates. Preserve
                                                  access to raw detector
                                                  output.

  D4                      Confidence threshold    Keep the existing fixed
                                                  confidence threshold of
                                                  **0.5**.

  D5                      Evaluation              No dedicated evaluation
                                                  study or evaluation
                                                  campaign is included in
                                                  M2.

  D6                      Person count            Detect all people
                                                  returned by the model,
                                                  but design and exercise
                                                  M2's temporal
                                                  perception behaviour
                                                  for a single-person
                                                  scenario. Multi-person
                                                  association is out of
                                                  scope.
  -----------------------------------------------------------------------

These decisions define the scope. Do not silently replace them with
alternative behaviours during implementation.

## 3. Scope

### 3.1 Included

1.  Reuse the existing M1 YOLOv8n detector and camera pipeline.
2.  Continue frame-by-frame person detection.
3.  Preserve raw detection results.
4.  Add EMA smoothing for bounding-box coordinates in the single-person
    case.
5.  Keep the confidence threshold fixed at 0.5.
6.  Report the current frame's detection result directly.
7.  Maintain existing FPS and inference-latency reporting.
8.  Handle invalid frames, camera read failures, and inference errors
    safely.
9.  Preserve reliable resource cleanup on normal shutdown and error
    paths.
10. Add deterministic unit tests for the new behaviour.

### 3.2 Explicitly excluded

-   Detection-miss grace periods or temporal retention of stale
    detections.
-   DETECTED / TEMPORARILY_MISSING / LOST state machine.
-   Persistent or temporary person IDs.
-   Person tracking or data association.
-   Person re-identification.
-   Trajectory or velocity estimation.
-   Approach, stationary, or departure classification.
-   Pose/keypoint estimation.
-   Robot-facing orientation estimation.
-   Metric distance estimation.
-   Interaction-zone reasoning.
-   First-contact readiness logic.
-   Candidate queues or target selection.
-   Dedicated benchmark/evaluation campaign or claims of measured
    robustness.
-   Multi-person temporal association.

These capabilities belong to later milestones or are otherwise outside
the agreed M2 scope.

## 4. Functional Requirements

### 4.1 Frame-level detection

For each valid input frame:

1.  Run the existing YOLOv8n person detector.
2.  Retain detections belonging to the person class.
3.  Apply the existing confidence threshold of 0.5.
4.  Return the current frame's raw detections.
5.  Do not carry a detection forward when the current frame contains no
    valid person detection.

A frame with no valid person detection must be represented as no
detection for that frame. M2 must not infer that a person remains
present based on an earlier frame.

### 4.2 Bounding-box smoothing

Implement a lightweight EMA over bounding-box coordinates for the
single-person use case.

For raw box b_t and previous smoothed box b_hat\_(t-1):

**b_hat_t = alpha \* b_t + (1 - alpha) \* b_hat\_(t-1)**

where: - `b_t = (x1, y1, x2, y2)` is the current raw bounding box. -
`b_hat_t` is the smoothed bounding box. - `alpha` is a smoothing
coefficient in `(0, 1]`.

Implementation requirements:

-   Make the coefficient explicit and configurable in the smoothing
    component or its configuration; do not change the detector
    confidence threshold.
-   Initialize the smoothed box from the first valid raw detection.
-   Preserve the raw bounding box separately; never overwrite the
    detector's raw output.
-   Apply smoothing only when there is an unambiguous single-person
    detection.
-   If there are zero detections, return no current detection and reset
    smoothing state.
-   If there are multiple person detections, do not arbitrarily
    associate one with a previous box. Preserve the raw detections and
    bypass single-person smoothing for that frame; reset any
    single-person smoothing state.
-   Ensure the output box remains finite and has valid coordinate
    ordering.
-   Do not add track IDs or imply that EMA smoothing performs identity
    association.

The coefficient's default value should be selected conservatively and
documented in code/configuration. Do not introduce a filter framework or
additional dependency.

### 4.3 Confidence handling

-   Keep the confidence threshold at 0.5, consistent with M1.
-   Do not add threshold tuning, threshold sweeps, adaptive thresholds,
    or separate acquisition and retention thresholds.
-   Preserve each detection's confidence value in the output.

### 4.4 Runtime behaviour

-   Preserve the existing camera configuration and live demo interface.
-   Preserve inference-latency and application-FPS reporting.
-   Keep inference errors visible; do not silently swallow exceptions.
-   Handle invalid frames and camera read failures safely.
-   Ensure camera, detector, and GUI resources are cleaned up on both
    normal and exceptional exits.

## 5. Data and Interface Expectations

Reuse the existing M1 `PersonDetection` representation and detector
interface wherever practical.

The perception output should make the distinction between: - **Raw
detection:** the direct result from the current YOLO inference. -
**Smoothed detection:** the EMA-filtered bounding box, when
single-person smoothing is applicable.

Do not make the smoothed box look like an independent detector
observation. Keep confidence and class information grounded in the
current raw detection.

Do not introduce a person identity field, track history, or temporal
identity semantics in M2.

If the existing M1 interfaces need a small extension to expose raw and
smoothed boxes, preserve backward compatibility where practical and keep
the change localized.

## 6. Testing Requirements

Add deterministic unit tests for the M2 additions.

### 6.1 EMA smoothing

-   First detection initializes the smoothed box.
-   A subsequent detection produces the expected EMA result.
-   Raw bounding-box coordinates remain unchanged.
-   Invalid smoothing coefficients are rejected.
-   Output coordinates remain finite and ordered.

### 6.2 Detection-count behaviour

-   Zero detections produce no current detection and reset smoothing
    state.
-   One detection uses EMA smoothing.
-   Multiple detections preserve raw outputs and bypass single-person
    smoothing.
-   Smoothing does not associate detections across ambiguous
    multi-person frames.

### 6.3 Existing runtime behaviour

-   The confidence threshold remains 0.5.
-   Invalid input frames are handled as specified by the existing
    detector contract.
-   Camera read failures remain visible and trigger cleanup.
-   Inference errors remain visible and trigger cleanup.
-   Normal shutdown releases resources.

Use mocked model outputs for deterministic tests. Do not make unit tests
depend on a physical person being present in front of the webcam.

## 7. Implementation Constraints

-   Inspect the actual repository structure and current Git status
    before editing.
-   Read the M1 specification and inspect the existing M1
    implementation.
-   Reuse working M1 components rather than rewriting them.
-   Modify only files necessary for M2 and directly relevant tests.
-   Avoid broad refactoring and unrelated cleanup.
-   Do not add unnecessary dependencies.
-   Do not modify unrelated governance documentation.
-   Do not commit changes.
-   Do not begin M3.

## 8. Completion Criteria

M2 is ready for review when:

1.  The existing M1 detection pipeline remains functional.
2.  Raw detector outputs remain accessible and unmodified.
3.  EMA smoothing works for unambiguous single-person detections.
4.  Smoothing resets on zero detections and is bypassed on multiple
    detections.
5.  A missed detection is reported immediately; no stale detection is
    retained.
6.  The confidence threshold remains fixed at 0.5.
7.  Existing runtime metrics and cleanup behaviour are preserved.
8.  New and existing relevant tests pass.
9.  The implementation report lists changed files, test
    commands/results, and known limitations.
10. No out-of-scope tracking or readiness functionality has been
    introduced.

## 9. Implementation Handoff

Implement only the requirements in this document. If an implementation
detail appears to require changing an agreed decision or expanding
scope, stop and ask for clarification rather than making that change
implicitly.

After implementation and tests, provide a concise completion report and
stop for review. Do not proceed to M3 without explicit approval.
