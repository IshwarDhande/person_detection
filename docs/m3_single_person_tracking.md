# M3 --- Single-Person Tracking

**Project:** Vision-Based First Contact for a Reception Social Robot\
**Milestone:** M3\
**Status:** Approved design specification\
**Predecessor:** M2 --- Reliable Single-Person Detection\
**Next milestone:** M4 --- Frame-Level Readiness Perception

## 1. Objective

M3 introduces temporal tracking for a single person using the existing
M1/M2 person-detection pipeline. The tracker maintains a temporary
`track_id`, estimates bounding-box motion, tolerates short detection
gaps, and attempts to associate returning detections with the existing
track.

M3 is lightweight and motion-based. It does not perform face
recognition, appearance-based re-identification, or multi-person
tracking.

**Detection** means a person is detected in the current frame.
**Tracking** means a current detection is associated with a previously
created temporary track. **Re-identification** after track expiration is
out of scope.

## 2. Agreed Design Decisions

  ------------------------------------------------------------------------
  ID                      Design area             Agreed decision
  ----------------------- ----------------------- ------------------------
  D1                      Tracking algorithm      Constant-velocity Kalman
                                                  filter + IoU-based
                                                  detection association.

  D2                      Detection-loss handling Retain and predict the
                                                  track during detection
                                                  gaps for at most **2
                                                  seconds**.

  D3                      Track confirmation      Require **3 consecutive
                                                  matched detections**
                                                  before confirmation.

  D4                      Reacquisition           Retain the original
                                                  `track_id` if a
                                                  returning detection
                                                  matches the prediction
                                                  within 2 seconds.
                                                  Motion-based matching
                                                  only.

  D5                      Track data/history      Minimal tracking state
                                                  and a bounded history of
                                                  **30 observations**.

  D6                      Track deletion/new      Delete the old track
                          creation                after timeout. Create a
                                                  new track only when a
                                                  new valid detection
                                                  arrives, assigning a new
                                                  ID.

  D7                      IoU threshold           Require **IoU \>= 0.3**.
                                                  Keep configurable.

  D8                      Kalman state            Eight-dimensional state:
                                                  centre position,
                                                  bounding-box size, and
                                                  their velocities.

  D9                      Prediction timing       Predict on every
                                                  incoming frame using
                                                  actual timestamps and
                                                  elapsed time.

  D10                     Association reliability Require IoU \>= 0.3 and
                                                  an acceptable Kalman
                                                  prediction-uncertainty
                                                  check. No appearance
                                                  embeddings.

  D11                     Visualization           Minimal: bounding box,
                                                  temporary `track_id`,
                                                  and lifecycle state. No
                                                  trajectory trail.

  D12                     Verification            Functional unit tests
                                                  with mocked detections
                                                  plus a live webcam smoke
                                                  test. No dedicated
                                                  quantitative tracking
                                                  evaluation campaign.
  ------------------------------------------------------------------------

These decisions are authoritative. Do not silently replace them with
alternative algorithms or behaviours.

## 3. Scope

### Included

1.  Reuse the existing YOLOv8n detector and M2 perception pipeline.
2.  Introduce a single-person tracking component and temporary
    `track_id`.
3.  Use an eight-dimensional constant-velocity Kalman filter.
4.  Associate detections using IoU and a prediction-uncertainty check.
5.  Confirm tracks after three consecutive matched detections.
6.  Predict on every incoming frame using actual timestamps.
7.  Retain temporarily unobserved tracks for up to 2 seconds.
8.  Reacquire the original ID when association succeeds within the
    timeout.
9.  Maintain minimal track metadata and a 30-observation bounded
    history.
10. Delete expired tracks and assign new IDs to subsequently created
    tracks.
11. Display bounding box, track ID, and lifecycle state.
12. Add deterministic unit tests and a live webcam smoke test.

### Explicitly excluded

-   Multi-person tracking or simultaneous multiple-track management.
-   Face recognition, face embeddings, appearance embeddings, DeepSORT
    or ReID networks.
-   Persistent identity across track deletion or separate visits.
-   Approach, departure, or stationary classification.
-   Pose/keypoint or robot-facing orientation estimation.
-   Metric distance or interaction-zone estimation.
-   First-contact readiness, candidate queues, or target selection.
-   Dedicated quantitative tracking benchmarks or annotated-video
    evaluation.
-   Additional neural networks or GPU acceleration solely for tracking.

## 4. Tracking Architecture

``` text
Camera Frame
    |
    v
Existing YOLOv8n Person Detector
    |
    v
Current Frame's Person Detections
    |
    v
Kalman Prediction (current timestamp)
    |
    v
IoU Association + Prediction-Uncertainty Check
    |
    +---- Valid match ----> Update Kalman state; increment consecutive hits;
    |                       observed = True
    |
    +---- No valid match -> observed = False; retain predicted state;
    |                       start/continue loss timer
    |
    v
Track Lifecycle Management
    |
    +---- 3 consecutive matches -> CONFIRMED
    +---- Detection missing ----> LOST / COASTING
    +---- Missing for 2 seconds -> DELETED
    |
    v
Tracking Output + Minimal Visualization
```

Keep the tracker separate from the YOLO detector so detection and
tracking can be tested independently.

## 5. Kalman Filter Design

### 5.1 State vector

Use:

\[ `\mathbf{x}`{=tex}\_t = \[c_x, c_y, w, h, v_x, v_y, v_w, v_h\]\^T \]

where `cx, cy` are bounding-box centre coordinates, `w, h` are width and
height, and the four velocity terms describe their rates of change.

### 5.2 Time-dependent prediction

Calculate elapsed time from monotonic timestamps:

\[ `\Delta `{=tex}t = t\_{current} - t\_{previous} \]

Use a constant-velocity transition model. For each position/size
component `q` and its corresponding velocity:

\[
```{=tex}
\begin{bmatrix} q_t \\ v_{q,t} \end{bmatrix}
```
=
```{=tex}
\begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} q_{t-1} \\ v_{q,t-1} \end{bmatrix}
```
\]

The implementation must use actual timestamps, not assume a fixed camera
FPS.

### 5.3 Measurement update

A matched detection supplies:

\[ `\mathbf{z}`{=tex}\_t = \[c_x, c_y, w, h\]\^T \]

Use it to update the predicted Kalman state and covariance. Use a small,
well-tested linear Kalman filter; do not add a neural network or
heavyweight tracking dependency.

### 5.4 Prediction uncertainty

Maintain the Kalman covariance and use it to assess prediction
uncertainty. An association must satisfy both the IoU threshold and an
acceptable uncertainty limit.

The exact uncertainty gate and process/measurement noise parameters are
implementation parameters that must be documented. Do not claim they are
empirically optimal. If the uncertainty gate cannot be implemented
without changing the agreed behaviour, stop and request clarification.

## 6. Detection Association

For predicted box `Bp` and detected box `Bd`:

\[ IoU(B_p,B_d) = `\frac{|B_p \cap B_d|}{|B_p \cup B_d|}`{=tex} \]

A candidate must satisfy:

\[ IoU(B_p,B_d) `\geq 0.3`{=tex} \]

and pass the Kalman uncertainty check.

-   Use motion prediction and bounding-box overlap only.
-   Do not compute face or appearance embeddings.
-   Do not perform biometric recognition.
-   If multiple candidate detections pass both gates for the one active
    track, choose the candidate with the highest IoU.
-   Do not associate a detection that fails either gate.
-   Keep association logic independently unit-testable.
-   M3 is designed for a single-person scenario; crowded multi-person
    behaviour is not an acceptance target.

## 7. Track Lifecycle

### 7.1 TENTATIVE

-   A valid detection initializes a track and assigns a new temporary
    `track_id`.
-   Initialize the Kalman state from that detection.
-   Require three consecutive matched detections for confirmation.
-   A missed or unmatched frame breaks the consecutive-hit sequence.

### 7.2 CONFIRMED

-   Enter after three consecutive matched detections.
-   Update the Kalman filter on every matched detection.
-   Mark `observed=True` when matched in the current frame.

### 7.3 LOST / COASTING

-   Enter when the active track has no valid associated detection.
-   Continue Kalman prediction on every incoming frame.
-   Set `observed=False`.
-   Never present a predicted box as a real detector observation.
-   Continue attempting motion-based association.
-   Retain for no more than 2 seconds from the last successful matched
    detection.
-   A successful association within the timeout restores the original
    ID. A previously confirmed track returns to CONFIRMED. A tentative
    track remains subject to its three-consecutive-hit requirement.

### 7.4 DELETED

-   Delete when elapsed time since the last successful match exceeds 2
    seconds.
-   Remove Kalman state, covariance, and bounded observation history.
-   Do not preserve identity information for later re-identification.
-   A subsequent valid detection creates a new track with a new ID.

## 8. Track Data Model

Keep only necessary tracking information:

  -----------------------------------------------------------------------
  Field                               Purpose
  ----------------------------------- -----------------------------------
  `track_id`                          Temporary identity

  `bbox_xyxy`                         Current observed or predicted box,
                                      with observation status explicit

  `center`                            Bounding-box centre

  `velocity`                          Estimated image-space motion

  `timestamp`                         Relevant update timestamp

  `age`                               Time since track creation

  `last_seen_timestamp`               Last successful detection
                                      association

  `hit_count`                         Consecutive successful matches

  `miss_count`                        Consecutive frames without a
                                      successful match

  `state`                             TENTATIVE, CONFIRMED, or LOST /
                                      COASTING

  `observed`                          Whether matched to a real detection
                                      in the current frame

  `history`                           At most 30 compact observations
  -----------------------------------------------------------------------

`age = current_time - creation_time`.

`miss_duration = current_time - last_seen_timestamp`.

Keep at most 30 observations. Store compact numerical data and
timestamps only. Do not store image crops, face images, embeddings, or
full frames. Clear history on deletion.

## 9. Tracking Output and Visualization

Expose a structured tracking result containing at least: - `track_id` -
`bbox_xyxy` - `observed` - `state` - `age` - `miss_duration`

A predicted box during LOST / COASTING must be explicitly marked as
predicted (`observed=False`), never as a current YOLO detection.

The live demo displays only: - Bounding box. - Temporary `track_id`. -
Lifecycle state.

Do not add trajectory trails, appearance information, or elaborate
diagnostic overlays.

## 10. Runtime and Resource Constraints

-   Reuse M1/M2 camera and detection components.
-   Track only one person.
-   Keep the Kalman filter computationally lightweight.
-   Keep the tracker independent of the camera backend.
-   Use monotonic timestamps for elapsed-time and timeout calculations.
-   Handle invalid timestamps and detections safely.
-   Ensure deterministic cleanup on normal shutdown and exceptions.
-   Do not allow expired tracks to remain in memory.

## 11. Testing and Verification

Use functional verification, not a dedicated quantitative tracking
evaluation campaign.

### 11.1 Deterministic unit tests

Use mocked detections and controlled timestamps. Cover at least:

1.  New detection creates a tentative track with a new ID.
2.  First and second consecutive matches leave it tentative.
3.  Third consecutive match confirms it.
4.  A missed frame breaks the consecutive-hit sequence.
5.  Kalman prediction uses actual elapsed time.
6.  Matched detections update the Kalman state.
7.  IoU exactly at threshold is accepted.
8.  IoU below 0.3 is rejected.
9.  An association failing the uncertainty gate is rejected.
10. A missing detection marks the track unobserved.
11. Predicted boxes are distinguishable from observed detections.
12. A matching detection within 2 seconds retains the original ID.
13. A track is deleted after the 2-second timeout.
14. A detection after deletion creates a new ID.
15. History never exceeds 30 observations.
16. Track age and miss duration are calculated independently.
17. Invalid detections or timestamps do not corrupt tracker state.
18. Normal shutdown and exceptional paths release resources.

### 11.2 Live webcam smoke test

If camera access is available, manually verify: 1. Person enters frame.
2. Person remains visible long enough for confirmation. 3. Person moves
within the camera view. 4. Person briefly leaves or becomes occluded. 5.
Person returns within 2 seconds. 6. Person remains absent beyond 2
seconds. 7. A new detection appears and receives a new ID. 8. Demo exits
cleanly.

Report whether the live test was actually performed. If unavailable,
mark it NOT VERIFIED.

Do not claim tracking accuracy, IDF1, HOTA, or other benchmark results;
quantitative evaluation is excluded from M3.

## 12. Implementation Constraints

-   Inspect actual repository structure and current Git status before
    editing.
-   Read the M2 specification and inspect existing M1/M2 code.
-   Reuse working components.
-   Modify only files necessary for M3 and directly relevant tests.
-   Avoid broad refactoring and unrelated cleanup.
-   Do not add unnecessary dependencies.
-   Do not modify unrelated governance documentation.
-   Do not commit changes.
-   Do not begin M4.
-   If implementation conflicts with an agreed decision or requires
    scope expansion, stop and ask for clarification.

## 13. Completion Criteria

M3 is ready for review when: 1. A single-person track is initialized
with a temporary ID. 2. Three consecutive matched detections are
required for confirmation. 3. The Kalman filter uses the agreed
eight-dimensional state. 4. Prediction runs on every incoming frame
using actual timestamps. 5. Association requires IoU \>= 0.3 and an
acceptable uncertainty check. 6. A missed detection marks the track
unobserved and triggers prediction. 7. A matching detection within 2
seconds retains the original ID. 8. A track is deleted after the
2-second retention timeout. 9. A later valid detection creates a new
track with a new ID. 10. History is bounded to 30 observations. 11.
Predicted boxes are distinguished from real detections. 12. Minimal
visualization shows bounding box, ID, and lifecycle state. 13. Relevant
unit tests pass. 14. Live webcam smoke testing is performed if
available, with unavailable tests explicitly identified. 15. No face
embeddings, appearance models, multi-person tracking, or readiness logic
are introduced.

## 14. Implementation Handoff

Implement only this specification.

After implementation and testing, create
`docs/m3_implementation_details.md`. The report must describe the actual
implementation, architecture, Kalman state and parameters, association
and uncertainty gates, lifecycle behaviour, changed files, test commands
and results, live webcam verification status, limitations, and a PASS /
FAIL / NOT VERIFIED completion checklist.

Stop after M3 implementation and report generation. Wait for review and
explicit approval before proceeding to M4.
