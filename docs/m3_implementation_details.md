# Milestone M3 Implementation Details: Single-Person Tracking

**Project:** Vision-Based First Contact for a Reception Social Robot  
**Milestone:** M3 — Single-Person Tracking  
**Status:** Implementation Complete & Verified  
**Authoritative Design Specification:** [m3_single_person_tracking.md](file:///home/eshwar/code/detection/docs/m3_single_person_tracking.md)  
**Predecessor:** Milestone M2 — Reliable Single-Person Detection ([m2_reliable_single_person_detection.md](file:///home/eshwar/code/detection/docs/m2_reliable_single_person_detection.md))  
**Successor:** Milestone M4 — Frame-Level Readiness Perception (Part A)  

---

## 1. Implementation Summary

Milestone M3 introduces motion-based temporal tracking for a single person using the existing YOLOv8n person detector from M1/M2. The tracking subsystem maintains a temporary numeric identifier (`track_id`), predicts bounding-box motion across frames using an 8-dimensional linear Kalman filter, associates detections via Intersection-over-Union (IoU $\ge 0.3$) and innovation Mahalanobis uncertainty gating, confirms tracks after three consecutive detections, and bridges temporary detection gaps for up to 2.0 seconds with motion coasting.

```text
+---------------------------------------------------------------------------------------------------+
|                           M3 SINGLE-PERSON TRACKING SUMMARY                                       |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  * Motion Model:         8-D Constant-Velocity Kalman Filter (cx, cy, w, h, vx, vy, vw, vh)       |
|  * Timing:               Monotonic elapsed time dt (no assumed fixed framerate)                   |
|  * Association Gating:   IoU >= 0.30 AND Mahalanobis Innovation Gate (d_M^2 <= 16.0)              |
|  * Candidate Selection:  Highest-IoU candidate among those passing both gates                     |
|  * Track Confirmation:   Strictly 3 consecutive matched detections (TENTATIVE -> CONFIRMED)       |
|  * Gap Tolerance:        LOST / COASTING state up to 2.0s retaining original track_id             |
|  * Deletion Semantics:   Timeout > 2.0s triggers full state wipe; subsequent detection gets new ID|
|  * Observation History:  Strictly bounded to 30 compact geometric observations (no crops/images)  |
|  * Visual Feedback:      Minimal HUD; distinct colors and explicit [PREDICTED] coasting tags      |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Architecture and Data Flow

The tracking pipeline is designed as an independent downstream module consuming detections from the existing `PersonDetector` and emitting a rich `Track` representation.

```text
Camera Frame (/dev/video0)
       |
       v
+-----------------------------------+
|   Ultralytics YOLOv8n (M1)        |
+-----------------------------------+
       |
       v Raw Detections (class_id=0, conf>=0.50)
+-----------------------------------+
|   SinglePersonTracker (M3)        |
+-----------------------------------+
       |
       +---> 1. Time Update: Predict Kalman state x_pred with monotonic dt
       |
       +---> 2. Gating:
       |        - IoU(B_pred, B_det) >= 0.30
       |        - Mahalanobis d_M^2 <= 16.0
       |
       +---> 3. Candidate Matching:
       |        - If match: Update Kalman, hit_count += 1, state evaluation
       |        - If no match: hit_count = 0, state = LOST, check 2.0s timeout
       |
       v
+-----------------------------------+
|   Track Data Structure            |
|   (track_id, state, bbox, obs)    |
+-----------------------------------+
       |
       v
+-----------------------------------+
|   Minimal HUD & Rendering         |
|   (Confirmed / Tentative / Lost)  |
+-----------------------------------+
```

### Component Responsibilities

1. **`PersonDetector` (M1/M2 Reuse)**:
   - Responsible strictly for per-frame image inference, class filtering (`cls == 0`), and confidence thresholding (`conf >= 0.50`).
   - Tracking consumes raw detections via `detect_raw()` to ensure Kalman filtering operates directly on physical observations rather than pre-smoothed EMA coordinates.

2. **`ConstantVelocityKalmanFilter8D`**:
   - Encapsulates the 8-D state vector, transition matrix $F(\Delta t)$, continuous white noise acceleration $Q(\Delta t)$, measurement matrix $H$, measurement noise covariance $R$, and Joseph-form posterior covariance update.

3. **`SinglePersonTracker`**:
   - Manages track lifecycle (`TENTATIVE`, `CONFIRMED`, `LOST`, `DELETED`).
   - Enforces 3-consecutive-hit confirmation, 2.0-second retention timeout, and monotonically incrementing temporary `track_id` allocation.
   - Executes IoU and Mahalanobis uncertainty gating.

---

## 3. Kalman Filter Implementation

### State and Measurement Vectors

The linear Kalman filter models 2D bounding-box center coordinates, dimensions, and their linear rates of change in pixel space:

$$\mathbf{x}_t = \begin{bmatrix} c_x & c_y & w & h & v_x & v_y & v_w & v_h \end{bmatrix}^T \in \mathbb{R}^8$$

$$\mathbf{z}_t = \begin{bmatrix} c_x & c_y & w & h \end{bmatrix}^T \in \mathbb{R}^4$$

### Time-Dependent Prediction Model

Given monotonic timestamps $t_{current}$ and $t_{previous}$, the elapsed duration is $\Delta t = t_{current} - t_{previous}$. The state transition matrix $F(\Delta t)$ is:

$$F(\Delta t) = \begin{bmatrix} I_{4\times 4} & \Delta t \cdot I_{4\times 4} \\ 0_{4\times 4} & I_{4\times 4} \end{bmatrix}$$

$$\mathbf{x}^- = F(\Delta t) \mathbf{x}_{t-1}$$

$$P^- = F(\Delta t) P_{t-1} F(\Delta t)^T + Q(\Delta t)$$

### Process and Measurement Noise

The process noise matrix $Q(\Delta t)$ follows the Continuous White Noise Acceleration (CWNA) model:

$$Q_1(\Delta t) = \sigma_a^2 \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$

- **Position acceleration noise:** $\sigma_{a, pos} = 80.0\text{ px/s}^2$
- **Size acceleration noise:** $\sigma_{a, size} = 20.0\text{ px/s}^2$
- **Measurement noise matrix:** $R = \text{diag}([10^2, 10^2, 20^2, 20^2])$

> [!NOTE]
> These noise parameters represent baseline values established for desktop webcam framing. They are not claimed to be empirically optimal for the final robot hardware.

### Posterior Measurement Update

When an association match occurs:

$$\mathbf{y} = \mathbf{z} - H \mathbf{x}^-$$

$$S = H P^- H^T + R$$

$$K = P^- H^T S^{-1}$$

$$\mathbf{x} = \mathbf{x}^- + K \mathbf{y}$$

Posterior covariance is updated using the numerically stable **Joseph form**:

$$P = (I - K H) P^- (I - K H)^T + K R K^T$$

---

## 4. Detection Association

Detection association pairs incoming YOLO detections with the single active track using spatial overlap and statistical uncertainty gating.

```text
               +----------------------------------+
               |  Candidate Detection Box B_det   |
               +-----------------+----------------+
                                 |
                                 v
               +----------------------------------+
               |   IoU(B_pred, B_det) >= 0.30?    |
               +-----------------+----------------+
                                 |
                 +---------------+---------------+
                 | YES                           | NO
                 v                               v
+----------------------------------+   +-------------------+
|  Mahalanobis Innovation Check    |   | Reject Candidate  |
|      d_M^2 <= 16.0?              |   +-------------------+
+----------------+-----------------+
                 |
   +-------------+-------------+
   | YES                       | NO
   v                           v
+-------------------+   +-------------------+
| Candidate Passes  |   | Reject Candidate  |
| Both Gates        |   +-------------------+
+-------------------+
```

1. **IoU Threshold**: $\text{IoU}(B_p, B_d) \ge 0.30$ (configurable via `--iou-thresh`).
2. **Prediction Uncertainty Gate**:
   $$d_M^2 = \mathbf{y}^T S^{-1} \mathbf{y} \le 16.0$$
   Under a 4-degree-of-freedom $\chi^2$ distribution, a critical value of $16.0$ encompasses $>99.5\%$ of valid detection noise, rejecting outliers and spurious misdetections.
3. **Multi-Candidate Selection**:
   If more than one candidate detection satisfies both gates, the candidate with the highest IoU is selected. Unselected detections are dropped.

---

## 5. Track Lifecycle

```text
       [First Valid Detection]
                  |
                  v
       +--------------------+
       |     TENTATIVE      | <-----------------------+
       |   (hit_count: 1)   |                         |
       +---------+----------+                         |
                 |                                    |
     [3 consecutive matches]                          |
                 |                                    |
                 v                                    |
       +--------------------+                         |
+----> |     CONFIRMED      |                         |
|      |  (observed=True)   |                         |
|      +---------+----------+                         |
|                |                                    |
|         [Detection Missed]                          |
|                |                                    |
|                v                                    |
|      +--------------------+                         |
|      |  LOST / COASTING   |                         |
|      |  (observed=False)  |                         |
|      +----+----------+----+                         |
|           |          |                              |
|           |   [Reacquired within <= 2.0s]           |
|           |          +--- If previously confirmed --+
|           |          +--- If was tentative ---------+
|           |
|    [Timeout > 2.0s]
|           |
|           v
|      +--------------------+
+----- |      DELETED       |
       |  (Wipe state & ID) |
       +--------------------+
```

### Lifecycle State Machine Details

1. **`TENTATIVE`**:
   - Initial state upon receiving the first valid person detection.
   - Assigns a temporary integer `track_id` (e.g. `1`).
   - Requires 3 consecutive frames with successful association before transitioning to `CONFIRMED`.
   - A single missed frame resets `hit_count = 0` and sends the track to `LOST`.

2. **`CONFIRMED`**:
   - Reached after 3 consecutive successful matches.
   - `observed = True` when matched in the current frame.
   - Retains verified status across brief occlusions if reacquired within 2.0 seconds.

3. **`LOST / COASTING`**:
   - Entered whenever no candidate detection matches the active track.
   - `observed = False`.
   - The Kalman filter advances motion prediction continuously on every incoming frame.
   - Bounding box is rendered with explicit `[PREDICTED]` tags.
   - Reacquisition within 2.0 seconds restores the track to its prior state with the original `track_id` intact.

4. **`DELETED`**:
   - Triggered when `miss_duration = t_current - last_seen_timestamp > 2.0` seconds.
   - Kalman state, covariance matrices, and observation history are completely deallocated.
   - No biometric or appearance data is retained for re-identification.
   - A new detection arriving later initializes a new track with an incremented ID (e.g. `2`).

---

## 6. Track Data Model and Memory Management

### Data Structures

```python
@dataclass(frozen=True)
class TrackObservation:
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]
    observed: bool

@dataclass
class Track:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]
    velocity: tuple[float, float, float, float]
    timestamp: float
    creation_timestamp: float
    last_seen_timestamp: float
    hit_count: int
    miss_count: int
    state: TrackState
    observed: bool
    history: deque[TrackObservation]  # maxlen=30
```

### Memory and History Boundaries

- Observation history is stored in a `collections.deque(maxlen=30)`. Once 30 observations are recorded, oldest observations are discarded in $O(1)$ time.
- Only lightweight numerical floats and booleans are stored (timestamp, bounding box coordinates, center coordinate, observed flag).
- No raw images, image crops, face patches, or feature embeddings are ever stored.
- On track expiration or `reset()`, all track references are dereferenced.

---

## 7. Code Changes

### Files Created

| File Path | Description |
| :--- | :--- |
| [kalman_filter.py](file:///home/eshwar/code/detection/perception/tracking/kalman_filter.py) | 8-D constant-velocity Kalman filter with CWNA process noise, Joseph-form update, and Mahalanobis gating |
| [single_person_tracker.py](file:///home/eshwar/code/detection/perception/tracking/single_person_tracker.py) | Track data model, lifecycle state machine, IoU association, and timeout management |
| [run_tracking_demo.py](file:///home/eshwar/code/detection/perception/tracking/run_tracking_demo.py) | Standalone live webcam demonstration runner with minimal visual overlays and performance HUD |
| [__init__.py](file:///home/eshwar/code/detection/perception/tracking/__init__.py) | Exports tracker classes and public API symbols |
| [test_kalman_filter.py](file:///home/eshwar/code/detection/tests/perception/tracking/test_kalman_filter.py) | Unit tests for Kalman filter matrix operations, prediction, update, and defensive checks |
| [test_single_person_tracker.py](file:///home/eshwar/code/detection/tests/perception/tracking/test_single_person_tracker.py) | Unit tests for track lifecycle, 3-hit confirmation, association, and 2-second timeout |
| [test_tracking_config.py](file:///home/eshwar/code/detection/tests/perception/tracking/test_tracking_config.py) | Unit tests for CLI parsing and guaranteed cleanup seams |
| [__init__.py](file:///home/eshwar/code/detection/tests/perception/tracking/__init__.py) | Package initialization for tracking unit tests |
| [m3_implementation_details.md](file:///home/eshwar/code/detection/docs/m3_implementation_details.md) | This comprehensive Milestone M3 implementation and verification report |

### Files Modified

| File Path | Modification Summary |
| :--- | :--- |
| [demo_commands.md](file:///home/eshwar/code/detection/docs/demo_commands.md) | Documented M2 and M3 live demonstration, profiling, and testing execution commands |

---

## 8. Testing Results

### Test Execution Command

```bash
cd /home/eshwar/code/detection && PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```

### Test Suite Execution Output Summary

```text
Ran 90 tests in 0.596s

OK
```

### Test Breakdown by Subsystem

| Test Suite File | Tests | Passed | Failed | Description |
| :--- | :---: | :---: | :---: | :--- |
| [test_person_detector.py](file:///home/eshwar/code/detection/tests/perception/detection/test_person_detector.py) | 20 | 20 | 0 | M1 regression: class filtering, confidence boundaries, model errors |
| [test_detection_config.py](file:///home/eshwar/code/detection/tests/perception/detection/test_detection_config.py) | 15 | 15 | 0 | M1/M2 regression: CLI argument parsing, camera initialization seams |
| [test_single_person_smoother.py](file:///home/eshwar/code/detection/tests/perception/detection/test_single_person_smoother.py) | 14 | 14 | 0 | M2 regression: EMA math, count branching, ordering invariants |
| [test_kalman_filter.py](file:///home/eshwar/code/detection/tests/perception/tracking/test_kalman_filter.py) | 13 | 13 | 0 | M3: 8-D state, CWNA noise blocks, F/H/S matrices, Joseph form, Mahalanobis gating |
| [test_single_person_tracker.py](file:///home/eshwar/code/detection/tests/perception/tracking/test_single_person_tracker.py) | 21 | 21 | 0 | M3: 12-field contract, Cases A/B/C lifecycle, IoU gating, timeout, bounded history |
| [test_tracking_config.py](file:///home/eshwar/code/detection/tests/perception/tracking/test_tracking_config.py) | 7 | 7 | 0 | M3: CLI parsing, FPS steady state, cleanup on normal/error shutdown |
| **Total** | **90** | **90** | **0** | **100% Pass Rate Across All Milestones** |

---

## 9. Runtime Verification

### Bounded Live Webcam Verification

A bounded live webcam smoke test was executed against `/dev/video0` using the standalone runner:

```bash
/home/eshwar/code/interrupt/venv/bin/python3 perception/tracking/run_tracking_demo.py \
  --camera /dev/video0 \
  --model yolov8n.pt \
  --conf 0.5 \
  --iou-thresh 0.3 \
  --timeout 2.0 \
  --max-frames 60 \
  --headless
```

### Observed Runtime Metrics

```text
==============================================================================
 Performance & Execution Summary
==============================================================================
 Total frames processed:      60
 Total runtime:               6.79 s
 Steady-state Application FPS: 8.79 FPS
 YOLO Latency (Mean):          106.11 ms
 YOLO Latency (Median):        54.15 ms
 YOLO Latency (Min / Max):     43.43 ms / 2856.47 ms
==============================================================================
Camera capture released.
PersonDetector closed.
SinglePersonTracker reset.
OpenCV windows destroyed.
```

### Runtime Verification Checklist

To eliminate ambiguity, runtime verification is partitioned into automated hardware checks versus human-operator physical interaction:

| Verification Category | Status | Details & Evidence |
| :--- | :---: | :--- |
| **1. Camera Stream Initialization** | **VERIFIED** | Successfully negotiated V4L2 MJPG stream (1280x720 @ 30.0 FPS) on `/dev/video0`. Initial test frame read validated. |
| **2. Bounded Pipeline Execution** | **VERIFIED** | Processed 60 real video frames continuously through YOLOv8n detector and Kalman tracker. Steady-state throughput measured at 8.79 FPS with median YOLO latency of 54.15 ms on CPU. |
| **3. Live Stream Tracking Validation** | **VERIFIED** | Live person detected in stream (confidence > 0.94); tracker initialized `track_id=1`, transitioned from `TENTATIVE` (hits 1, 2) to `CONFIRMED` (hit 3), and maintained track across 20 frames. |
| **4. Unit-Tested Lifecycle Behaviour** | **VERIFIED** | All lifecycle state transitions (Cases A, B, and C), 2-second timeout deletion, and reacquisition are deterministically proven in `test_single_person_tracker.py` (21/21 passing). |
| **5. Guaranteed Resource Teardown** | **VERIFIED** | Clean release of camera capture, detector closure, tracker reset, and window destruction verified on both normal exit and mock-injected crash conditions. |
| **6. Interactive Human Occlusion Test** | **NOT VERIFIED** | Requires an actual human operator to physically step out of camera view for <2s (reacquisition) and >2s (track deletion) in an interactive GUI session. |

---

## 10. Verification Findings and Mathematical Proofs

### A. Tracking Output Contract Verification

The public `Track` class was inspected and verified to expose all required contract attributes:

1. `track_id`: Monotonically incrementing integer assigned on track creation.
2. `bbox_xyxy`: Float 4-tuple $(x_1, y_1, x_2, y_2)$ in pixel coordinates.
3. `center`: Center coordinate $(c_x, c_y)$.
4. `velocity`: Image-space velocity $(v_x, v_y, v_w, v_h)$ in pixels/sec.
5. `timestamp`: Monotonic update timestamp.
6. `creation_timestamp`: Monotonic creation timestamp.
7. `last_seen_timestamp`: Monotonic timestamp of last successful detection association.
8. `age`: Derived property: `max(0.0, timestamp - creation_timestamp)`.
9. `miss_duration`: Derived property: `max(0.0, timestamp - last_seen_timestamp)`.
10. `hit_count`: Consecutive match count.
11. `miss_count`: Consecutive unobserved frame count.
12. `state`: `TrackState` enum (`TENTATIVE`, `CONFIRMED`, `LOST`, `DELETED`).
13. `observed`: Explicit boolean (`True` on real detection match; `False` on coasting prediction).
14. `history`: Bounded `deque[TrackObservation]` strictly capped at 30 compact observations.

Status: **VERIFIED** (Automated proof in `test_tracking_output_contract`).

### B. Lifecycle Edge Case Verification

1. **Case A (Tentative Interruption)**:
   A tentative track receiving hits on frames 1 and 2 that misses frame 3 immediately transitions to `LOST` with `hit_count = 0`. Upon reacquisition on frame 4, state returns to `TENTATIVE` with `hit_count = 1`. Subsequent hits 2 and 3 advance it to `CONFIRMED`.
   Status: **VERIFIED** (Automated proof in `test_case_a_tentative_interrupted_by_missed_frame_full_sequence`).

2. **Case B (Unmatched Detection While LOST)**:
   While an active track is in `LOST / COASTING` (within its 2.0s timeout), an incoming candidate detection that fails IoU or uncertainty gating does NOT replace the coasting track or spawn a second track. The single existing track is retained in `LOST`, output coordinates reflect the Kalman prediction, and no identity mixing occurs.
   Status: **VERIFIED** (Automated proof in `test_case_b_unmatched_detection_while_track_is_lost`).

3. **Case C (Reacquisition Semantics)**:
   - A confirmed track lost for $< 2.0$s returns directly to `CONFIRMED` upon valid reacquisition with original `track_id`.
   - A tentative track lost for $< 2.0$s returns to `TENTATIVE` with original `track_id` and must re-earn 3 consecutive hits.
   - An expired track ($> 2.0$s) is deleted (`DELETED`); a subsequent detection creates a new track with an incremented `track_id`.
   Status: **VERIFIED** (Automated proof in `test_case_c_reacquisition_rules`).

### C. Kalman Filter Mathematical Correctness

1. **State-Transition Pairing**:
   The transition matrix $F(\Delta t)$ has block form $\begin{bmatrix} I_4 & \Delta t I_4 \\ 0 & I_4 \end{bmatrix}$. Verified state pairings: $(c_x \leftrightarrow v_x)$ at indices $(0, 4)$, $(c_y \leftrightarrow v_y)$ at $(1, 5)$, $(w \leftrightarrow v_w)$ at $(2, 6)$, and $(h \leftrightarrow v_h)$ at $(3, 7)$.
   Status: **VERIFIED** (Automated proof in `test_state_transition_matrix_structure`).

2. **Process Noise Matrix $Q(\Delta t)$**:
   CWNA blocks $\begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$ are verified to be placed at indices $(0, 4)$, $(1, 5)$ for position noise ($\sigma_a = 80.0\text{ px/s}^2$), and $(2, 6)$, $(3, 7)$ for size noise ($\sigma_a = 20.0\text{ px/s}^2$). $Q(\Delta t)$ is proven symmetric positive-semidefinite.
   Status: **VERIFIED** (Automated proof in `test_process_noise_matrix_structure_and_cwna_blocks`).

3. **Measurement Matrix & Innovation Covariance**:
   $H = [I_4, 0_4]$ correctly extracts $[c_x, c_y, w, h]$. Innovation covariance $S = H P^- H^T + R$ is proven symmetric positive-definite.
   Status: **VERIFIED** (Automated proof in `test_measurement_matrix_and_innovation_covariance`).

4. **Mahalanobis Uncertainty Gate Consistency**:
   $d_M^2 = y^T S^{-1} y$ computed by `mahalanobis_distance_squared` is proven to strictly use the exact same innovation covariance $S$ as the posterior update.
   Status: **VERIFIED** (Automated proof in `test_mahalanobis_gate_uses_same_innovation_covariance`).

5. **Covariance Numerical Stability**:
   Joseph-form covariance updates ensure $P = (I - KH)P^-(I - KH)^T + KRK^T$ maintains positive eigenvalues and exact symmetry across 100 continuous iterations without numerical degradation.
   Status: **VERIFIED** (Automated proof in `test_covariance_numerical_stability_across_multiple_steps`).

---

## 11. Limitations and Known Issues

1. **Single-Person Scope**:
   - The tracker maintains at most one active track. When two or more people are in the camera view, the candidate with the highest IoU to the existing track is associated, and any secondary person is ignored. Multi-person tracking is intentionally deferred to Milestone M7.
2. **Motion-Only Association**:
   - Reacquisition relies strictly on constant-velocity Kalman prediction and bounding-box spatial overlap (IoU $\ge 0.3$). If a person makes an erratic, rapid movement while occluded and reappears far from the predicted box, association will fail, leading to track expiration after 2.0 seconds.
3. **No Biometric Re-Identification**:
   - By explicit design, no appearance embeddings or facial features are stored. Once a track expires after 2.0 seconds, returning visitors receive a new `track_id`.

---

## 12. M3 Completion Checklist

All completion criteria specified in `docs/m3_single_person_tracking.md` are evaluated below:

| ID | Completion Criterion | Verification Evidence | Status |
| :--- | :--- | :--- | :---: |
| **C1** | Single-person track initialized with temporary ID | Verified in `test_new_detection_creates_tentative_track` | **PASS / VERIFIED** |
| **C2** | Three consecutive matched detections required for confirmation | Verified in `test_confirmation_requires_three_consecutive_hits` & `test_case_a` | **PASS / VERIFIED** |
| **C3** | Kalman filter uses agreed eight-dimensional state | Verified in `test_state_vector_initialization` & `test_state_transition_matrix_structure` | **PASS / VERIFIED** |
| **C4** | Prediction runs on every incoming frame using actual timestamps | Verified in `test_constant_velocity_prediction` and pipeline loop | **PASS / VERIFIED** |
| **C5** | Association requires IoU $\ge 0.3$ and acceptable uncertainty check | Verified in `test_iou_above_and_at_threshold_accepted` and `test_uncertainty_gate_rejection` | **PASS / VERIFIED** |
| **C6** | Missed detection marks track unobserved and triggers prediction | Verified in `test_missed_frame_breaks_consecutive_hits` & `test_case_b` | **PASS / VERIFIED** |
| **C7** | Matching detection within 2 seconds retains original ID | Verified in `test_reacquisition_within_two_seconds_retains_id` & `test_case_c` | **PASS / VERIFIED** |
| **C8** | Track deleted after 2-second retention timeout | Verified in `test_expiration_after_two_seconds_deletes_track_and_assigns_new_id` | **PASS / VERIFIED** |
| **C9** | Subsequent valid detection creates new track with new ID | Verified in `test_case_c_reacquisition_rules` | **PASS / VERIFIED** |
| **C10** | History is bounded to 30 observations | Verified in `test_history_bounded_to_thirty_observations` | **PASS / VERIFIED** |
| **C11** | Predicted boxes distinguished from real detections | Verified via `observed` flag, magenta color, and `[PREDICTED]` tag | **PASS / VERIFIED** |
| **C12** | Minimal visualization shows bounding box, ID, and lifecycle state | Verified in `render_tracking` implementation | **PASS / VERIFIED** |
| **C13** | Relevant unit tests pass | 90/90 unit tests passing across repository | **PASS / VERIFIED** |
| **C14a** | Hardware camera stream initialization & bounded 60-frame run | Successfully captured 60 frames on `/dev/video0` at 1280x720 | **PASS / VERIFIED** |
| **C14b** | Deterministic lifecycle & occlusion verification | Verified in `test_case_a`, `test_case_b`, `test_case_c` | **PASS / VERIFIED** |
| **C14c** | Interactive live physical human occlusion test | Requires live human operator presence in front of camera | **NOT VERIFIED** |
| **C15** | No face embeddings, appearance models, multi-person tracking, or readiness logic | Strictly omitted per M3 scope | **PASS / VERIFIED** |

---

## 13. Milestone Recommendation

- **Implementation Integrity:** Complete and fully compliant with `docs/m3_single_person_tracking.md`.
- **Test Coverage:** Comprehensive (90/90 tests passing; 32 dedicated tracking tests).
- **Mathematical Correctness:** Verified for CWNA process noise, transition matrix state pairs, innovation covariance, and Joseph-form covariance updates.
- **Freeze Recommendation:** **M3 READY TO FREEZE** (Software implementation, mathematical models, and deterministic test suite are complete and verified; interactive physical human occlusion remains marked NOT VERIFIED as expected for automated verification).


