# Milestone M5 Implementation Details: Temporal Behaviour Estimation (Part B)

**Project:** Vision-Based First Contact for a Reception Social Robot  
**Milestone:** M5 — Temporal Behaviour Estimation (Part B)  
**Status:** Implemented, Tested, Verified & Frozen  
**Authoritative Design Specification:** [m5_temporal_behaviour_estimation.md](file:///home/eshwar/code/detection/docs/m5_temporal_behaviour_estimation.md)  
**Predecessors:** Milestone M1 ([m1_basic_detection.md](file:///home/eshwar/code/detection/docs/m1_basic_detection.md)), Milestone M2 ([m2_reliable_single_person_detection.md](file:///home/eshwar/code/detection/docs/m2_reliable_single_person_detection.md)), Milestone M3 ([m3_single_person_tracking.md](file:///home/eshwar/code/detection/docs/m3_single_person_tracking.md)), Milestone M4 ([m4_stationary_readiness.md](file:///home/eshwar/code/detection/docs/m4_stationary_readiness.md))  
**Successor:** Milestone M6 — First-Contact Readiness Decision  

---

## 1. Executive Summary

Milestone M5 establishes temporal human behaviour perception for the reception robot. Where M4 evaluates instantaneous properties within single camera frames, M5 aggregates consecutive frame observations over time to deduce behavioral dynamics: whether the tracked person is actively approaching the robot, whether they have slowed down and stopped within the interaction zone, and whether they have remained stable for two seconds.

M5 requires no additional neural network or deep learning model. It consumes structured bounding-box detections from the active M3 single-person tracker and frame-level perception states from M4, processing them through deterministic geometric, statistical, and temporal state machines.

Following an authoritative narrow correction pass, five specific improvements were verified and implemented:
1. **Correction 1 (P0 — Stability Tolerance Interval Accounting):** Replaced instantaneous sampling accumulation with interval accounting: an excursion spanning from first ineligible sample $t_{\text{first\_ineligible}}$ up to recovery $t_{\text{recovery}}$ charges the complete elapsed interval $(t_{\text{recovery}} - t_{\text{first\_ineligible}})$ against the 200 ms budget.
2. **Correction 2 (P0 — Stability Uncertainty vs Threshold Excursions):** Explicitly separated valid threshold excursions (`MOVING + VALID` or temporary step out of zone) from unavailable/unreliable evidence (`UNKNOWN`, `UNCERTAIN`, `INVALID`, missing). Unavailable/uncertain observations and tracking loss trigger immediate hard resets, never consuming tolerance.
3. **Correction 3 (P0 — Strict M3–M4 Same-Person Association):** Replaced geometric IoU fallback with strict detection bounding-box equality against M3's `matched_detection`. If `matched_detection is None` or coordinates do not match M4's selected box, the frame is rejected.
4. **Correction 4 (P1 — Test Report Accuracy):** Documented the exact per-file test breakdown summing precisely to 78 passing tests across 11 test files, verified against the full 231-test repository regression suite.
5. **Correction 5 (P1 — Fine-Grained Latency & Resource Reporting):** Instrumented component-level latencies (`history_ms`, `motion_ms`, `stationary_ms`, `approach_ms`, `stability_ms`, `total_ms`) reporting p50, p95, p99 percentiles, and tracked process CPU and RSS memory usage via `psutil`.

```text
+---------------------------------------------------------------------------------------------------+
|                        M5 TEMPORAL BEHAVIOUR ESTIMATION SUMMARY                                   |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  * Scope:                Temporal behaviour over time (velocity, displacement, approach, stability)|
|  * Target Input:         Observed bounding-box centers from active M3 track and M4 frame state    |
|  * Prediction Policy:    Kalman predictions strictly excluded from motion history buffer          |
|  * Same-Person Rule:     Integration layer verifies matched_detection == M4 selected box (strict) |
|  * Reference Point:      2D Geometric Centroid of M4 interaction-zone polygon (Shoelace formula)  |
|  * History Buffer:       1.0s bounded history, strict Delta t <= 0.2s max observation gap        |
|  * Initialization:       5 consecutive valid observations required before READY                   |
|  * Motion Model:         Timestamp-aware linear regression for 2D velocity vector (vx, vy px/s)   |
|  * Stationary Rule:      Speed < 20.0 px/s AND 1.0s Displacement < 20.0 px (strict inequalities)   |
|  * Approach Rule:        Direction Cosine > 0.5 AND Closing Speed > 10.0 px/s for >= 0.5s         |
|  * Stability Timer:      2.0s continuous stability with 200 ms cumulative excursion interval budget|
|  * Reset Semantics:      Immediate hard reset on tracking loss, gap > 0.2s, or UNKNOWN/UNCERTAIN  |
|  * Unit Test Suite:      78 dedicated M5 unit tests; 231 total repository tests passing (100%)     |
|  * Runtime Overhead:     ~0.18 ms per frame on CPU (virtually zero computational footprint)       |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. M5 Scope and Requirements

M5 is responsible for extracting temporal behavioral evidence from consecutive observed detections of the active tracked human candidate.

### In Scope
1. **Bounding-Box Center Extraction:** Deriving $(cx, cy) = ((x_1 + x_2)/2, (y_1 + y_2)/2)$ from verified observed detections.
2. **Observation History Buffer:** Maintaining a bounded one-second history buffer ordered by monotonic timestamps.
3. **Temporal Initialization:** Enforcing a 5-consecutive-observation requirement before motion estimation is marked `READY`.
4. **Continuity & Gap Enforcement:** Enforcing $\Delta t \le 0.2\text{ s}$ (200 ms); clearing history on discontinuities.
5. **Image-Space Velocity Estimation:** Fitting timestamp-aware linear regression over history to estimate $v_t = (v_x, v_y)$ and speed $s_t = \|v_t\|_2$ in px/s.
6. **Net Displacement Estimation:** Calculating $D_t = \|c_{\text{latest}} - c_{\text{earliest}}\|_2$ and validating 1.0s temporal coverage.
7. **Centroid-Directed Approach Detection:** Computing distance $d_t$ to interaction-zone centroid, direction cosine $\cos(\theta_t)$, distance trend closing speed $-b_d$, and requiring 500 ms sustained confirmation.
8. **Stationary Predicate:** Evaluating $s_t < 20\text{ px/s} \land D_t < 20\text{ px}$ over 1.0s.
9. **Stability Duration Timing:** Continuous 2.0s accumulation with 200 ms cumulative tolerance interval accounting for brief threshold excursions.
10. **State Management & Immediate Reset:** Immediate state clearing on tracking loss, track identity change, or required observation invalidation/uncertainty.

### Explicitly Out of Scope
- No multi-person tracking, target switching, or crowd management.
- No Re-ID, appearance embeddings, or facial recognition.
- No 3D pose estimation, depth cameras, or metric ground-plane projection.
- No final readiness declaration (reserved strictly for Milestone M6).

---

## 3. Architecture and Data Flow

```text
+-----------------------------------------------------------------------------+
|                     M5 END-TO-END SYSTEM PIPELINE ARCHITECTURE              |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Camera Frame (1920x1080 @ 30 FPS)                                          |
|       |                                                                     |
|       v                                                                     |
|  [PersonDetector (M1/M2)] -----> raw_detections                             |
|       |                               |                                     |
|       v                               v                                     |
|  [SinglePersonTracker (M3)]     [FrameObserver (M4)]                        |
|       | (Kalman Predict/Update)       | (Option A Person Selection)         |
|       v                               v                                     |
|  Track (ID, observed, bbox)     FrameObservation (Presence, Bbox, Zone...)  |
|       |                               |                                     |
|       +---------------+---------------+                                     |
|                       |                                                     |
|                       v                                                     |
|      [TemporalEstimator.estimate() (M5 Integration Boundary)]               |
|                       |                                                     |
|       +---------------+---------------+                                     |
|       | Strict Same-Person Check:     |                                     |
|       |   matched_det is not None?    |                                     |
|       |   matched_det.bbox == M4 box? |                                     |
|       |   Track observed == True?     |                                     |
|       +---------------+---------------+                                     |
|                       |                                                     |
|            +----------+----------+                                          |
|            | Valid               | Invalid / Coasting / Mismatch            |
|            v                     v                                          |
|   Extract Center c_t         Immediate Hard Reset                           |
|   History Buffer             Emit UNKNOWN [INVALID]                         |
|            |                                                                |
|            v                                                                |
|   +-----------------------+-----------------------+                         |
|   |                       |                       |                         |
|   v                       v                       v                         |
| [MotionEstimator]    [ApproachEstimator]   [StationaryEstimator]            |
|  v_t, s_t, D_t        Direction + Closing   s_t < 20 & D_t < 20             |
|   |                   500ms confirmation          |                         |
|   +-----------------------+-----------------------+                         |
|                           |                                                 |
|                           v                                                 |
|                   [StabilityTimer]                                          |
|                   Eligibility: P & B & Z & S                                |
|                   2.0s continuous + 0.2s tolerance interval budget          |
|                           |                                                 |
|                           v                                                 |
|                  TemporalObservation                                        |
|                           |                                                 |
|                           v                                                 |
|             (Ready for Milestone M6 Input)                                  |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## 4. Source Files and Responsibilities

All M5 source code is placed in lowercase directories according to repository standards:

| File Path | Component | Responsibility |
|---|---|---|
| [observation_types.py](file:///home/eshwar/code/detection/perception/temporal/observation_types.py) | Observation Models | Enums (`ApproachValue`, `MovementValue`, `StabilityValue`, `InitializationState`), `TemporalTimings`, metric dataclasses, and output container. |
| [centroid.py](file:///home/eshwar/code/detection/perception/temporal/centroid.py) | Polygon Geometry | Calculates 2D polygon centroid via Shoelace/Green's theorem with degenerate area validation. |
| [history.py](file:///home/eshwar/code/detection/perception/temporal/history.py) | Buffer & Continuity | Timestamp-ordered deque, 1.0s pruning, 200 ms max gap check, and 5-observation initialization. |
| [motion_estimator.py](file:///home/eshwar/code/detection/perception/temporal/motion_estimator.py) | Motion Estimation | Timestamp-aware linear regression for velocity vector $(v_x, v_y)$, speed $s_t$, and net displacement $D_t$. |
| [stationary_estimator.py](file:///home/eshwar/code/detection/perception/temporal/stationary_estimator.py) | Stationary Predicate | Evaluates strict inequality criteria ($s_t < 20$ px/s and $D_t < 20$ px over 1.0s). |
| [approach_estimator.py](file:///home/eshwar/code/detection/perception/temporal/approach_estimator.py) | Approach Detection | Direction cosine $\cos(\theta_t)$, distance trend closing speed $-b_d$, and 500 ms sustained confirmation. |
| [stability_timer.py](file:///home/eshwar/code/detection/perception/temporal/stability_timer.py) | Stability Timer | 2.0s continuous stability timer with 200 ms cumulative excursion interval accounting. |
| [temporal_config.py](file:///home/eshwar/code/detection/perception/temporal/temporal_config.py) | Config Management | Dataclasses, validation rules, consistency checks, and YAML config loader. |
| [temporal_estimator.py](file:///home/eshwar/code/detection/perception/temporal/temporal_estimator.py) | Pipeline Orchestrator | Coordinates strict same-person association, updates sub-estimators, instruments latencies, and emits `TemporalObservation`. |
| [run_temporal_demo.py](file:///home/eshwar/code/detection/perception/temporal/run_temporal_demo.py) | Live Demo Runner | Full end-to-end interactive runner with trajectory breadcrumbs, vector arrow, diagnostic HUD, percentile latency reporting, and psutil monitoring. |
| [temporal_behaviour.yaml](file:///home/eshwar/code/detection/configs/m5/temporal_behaviour.yaml) | Configuration | Default operational parameters for M5. |

---

## 5. M3–M4–M5 Integration Contract

### Strict Same-Person Association Policy (Correction 3)
Milestone M3 tracks a single candidate person using an 8-D Kalman filter, whereas Milestone M4 performs frame-level Option A person selection (picking the detection with the highest YOLO confidence score).

To prevent combining M3 history for Person A with M4 observations for Person B in multi-person scenes, the integration layer enforces strict same-person association:
1. `SinglePersonTracker` records `last_matched_detection` upon successful measurement association.
2. `TemporalEstimator` verifies that `matched_detection is not None` and `matched_detection.bbox_xyxy == frame_observation.person.bbox_metrics.original_bbox_xyxy`.
3. All fallback to geometric IoU against the tracker bounding box has been strictly removed.
4. If `matched_detection is None` or bounding-box coordinates differ, association fails. The frame is rejected, history is reset, and `UNKNOWN` [INVALID] is emitted.

```text
+-----------------------------------------------------------------------------+
|                     STRICT SAME-PERSON ASSOCIATION LOGIC                    |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Input: M3 Track, M4 FrameObservation, matched_detection                    |
|                                                                             |
|  Condition:                                                                 |
|    m4_person.presence == DETECTED [VALID]                                   |
|    AND m4_person.bounding_box == VALID [VALID]                              |
|    AND matched_detection is not None                                        |
|    AND matched_detection.bbox_xyxy == m4_person.bbox_metrics.original_bbox  |
|                                                                             |
|  Outcomes:                                                                  |
|    - TRUE:  Exact person identity confirmed. Update temporal estimators.     |
|    - FALSE: Rejection. Immediate reset; emit UNKNOWN [INVALID].             |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### Exclusion of Predicted Kalman Positions
Only detections supported by a genuine YOLO match (`track.observed == True`) are accepted. If a track is coasting (`observed == False`), the predicted box is strictly excluded from entering `ObservationHistory`.

### Timestamp Semantics
All temporal calculations use monotonic frame timestamps ($t_{\text{capture}}$) sampled at frame acquisition. Non-monotonic ($t_i \le t_{i-1}$) timestamps raise immediate errors.

---

## 6. Mathematical Formulations

### 6.1 Interaction-Zone Centroid
For a zone polygon $P = \{(x_0, y_0), \dots, (x_{n-1}, y_{n-1})\}$:

\[
A = \frac{1}{2} \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i)
\]

\[
x_z = \frac{1}{6A} \sum_{i=0}^{n-1} (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
\]

\[
y_z = \frac{1}{6A} \sum_{i=0}^{n-1} (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
\]

Degenerate polygons with $|A| < 10^{-6}$ are rejected during validation.

### 6.2 Linear Regression Velocity Vector
For observations $\{(t_i, c_i = (x_i, y_i))\}_{i=1}^N$:

\[
\bar{t} = \frac{1}{N} \sum_{i=1}^N t_i, \quad \bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \quad \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i
\]

\[
\text{Var}_t = \sum_{i=1}^N (t_i - \bar{t})^2
\]

\[
b_x = \frac{\sum_{i=1}^N (t_i - \bar{t})(x_i - \bar{x})}{\text{Var}_t}, \quad b_y = \frac{\sum_{i=1}^N (t_i - \bar{t})(y_i - \bar{y})}{\text{Var}_t}
\]

\[
v_t = \begin{bmatrix} b_x \\ b_y \end{bmatrix}, \quad s_t = \|v_t\|_2 = \sqrt{b_x^2 + b_y^2} \quad (\text{px/s})
\]

### 6.3 Net Displacement
\[
D_t = \|c_{\text{latest}} - c_{\text{earliest}}\|_2 = \sqrt{(x_{\text{latest}} - x_{\text{earliest}})^2 + (y_{\text{latest}} - y_{\text{earliest}})^2}
\]
Valid only when $t_{\text{latest}} - t_{\text{earliest}} \ge W_{\text{disp}} - 0.05\text{ s}$ (1.0 second coverage).

---

## 7. Temporal State Management

```text
+-----------------------------------------------------------------------------+
|                          M5 STATE MACHINE TRANSITIONS                       |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [Initialization State]                                                     |
|    UNINITIALIZED --(1st obs)--> COLLECTING (1/5)                            |
|    COLLECTING --(valid obs)--> COLLECTING (2..4/5)                          |
|    COLLECTING --(5th obs)----> READY (5/5)                                  |
|    ANY STATE --(gap > 0.2s or track loss)--> RESET                          |
|    RESET --(fresh valid obs)--> COLLECTING (1/5)                            |
|                                                                             |
|  [Approach State]                                                           |
|    UNKNOWN --(motion evidence)--> CANDIDATE                                 |
|    CANDIDATE --(evidence for 0.5s)--> APPROACHING                           |
|    CANDIDATE / APPROACHING --(evidence ceases)--> NOT_APPROACHING           |
|    ANY STATE --(tracking loss / gap)--> UNKNOWN                             |
|                                                                             |
|  [Stability State]                                                          |
|    NOT_STARTED --(E_t True)--> ACCUMULATING                                 |
|    ACCUMULATING --(excursion <= 0.2s)--> ACCUMULATING                      |
|    ACCUMULATING --(elapsed >= 2.0s)--> COMPLETE (one-shot)                  |
|    ACCUMULATING --(excursion > 0.2s or hard reset)--> RESET                 |
|    COMPLETE --(E_t ceases)--> RESET                                         |
|    RESET --(fresh E_t True)--> ACCUMULATING                                 |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## 8. Approach Estimation

Approach detection requires simultaneous directional and closing-distance evidence:

1. **Directional Vector:**
   Vector towards centroid $r_t = z - c_t$.
   Cosine similarity:
   \[
   \cos(\theta_t) = \frac{v_t \cdot r_t}{\|v_t\|_2 \|r_t\|_2}
   \]
   Criterion: $\cos(\theta_t) > \tau_\theta$ (default provisional $\tau_\theta = 0.50$, representing within $\pm 60^\circ$ of centroid).

2. **Distance Trend (Closing Speed):**
   Distance to centroid $d_i = \|z - c_i\|_2$.
   Linear trend slope $b_d = \frac{\text{Cov}(t, d)}{\text{Var}(t)}$. Closing speed is $-b_d$.
   Criterion: $-b_d > \tau_a$ (default provisional $\tau_a = 10.0\text{ px/s}$).

3. **500 ms Sustained Confirmation:**
   Instantaneous evidence $A_t = (\cos(\theta_t) > 0.5) \land (-b_d > 10.0)$ must hold continuously for $T_a \ge 0.50\text{ s}$ before emitting `APPROACHING` [VALID].

---

## 9. Stationary Estimation

Stationary classification combines rate of motion with accumulated displacement over one second:

\[
S_t = (s_t < 20.0\text{ px/s}) \land (D_t < 20.0\text{ px})
\]

- Strict less-than comparisons are enforced.
- If $s_t = 20.0$ px/s or $D_t = 20.0$ px, the predicate evaluates to `MOVING`.
- If temporal coverage is $< 0.95\text{ s}$, displacement is `UNCERTAIN` and stationary output is `UNKNOWN` [UNCERTAIN].
- Complement rule: `MOVING` is emitted only when both speed and displacement estimates are VALID and at least one threshold is exceeded.

---

## 10. Stability Duration, Interval Accounting, and Reset Semantics

### Eligibility Predicate
\[
E_t = P_t(\text{DETECTED}) \land B_t(\text{VALID}) \land Z_t(\text{INSIDE}) \land S_t(\text{STATIONARY})
\]
All four conditions must hold with status `VALID`.

### Interval Accounting (Correction 1)
Instead of point-sampling accumulation, the stability timer charges the complete sampled interval:
- An excursion begins at the first ineligible sample timestamp $t_{\text{first\_ineligible}}$.
- While the excursion is ongoing, active violation is dynamically evaluated as $t_{\text{current}} - t_{\text{first\_ineligible}}$.
- Upon recovery at $t_{\text{recovery}}$ (when $E_t$ becomes True again), the completed interval span $(t_{\text{recovery}} - t_{\text{first\_ineligible}})$ is charged to the cumulative violation budget $\tau_v$.
- In discrete 20 FPS sampling ($dt = 0.05$ s):
  - 1-frame excursion (Ineligible at 0.05, Eligible at 0.10) charges $0.10 - 0.05 = 0.05$ s.
  - 2-frame excursion (Ineligible at 0.05, 0.10, Eligible at 0.15) charges $0.15 - 0.05 = 0.10$ s.
  - 4-frame excursion (Ineligible at 0.05..0.20, Eligible at 0.25) charges $0.25 - 0.05 = 0.20$ s (permitted at boundary $\le 0.20$ s).
  - 5-frame excursion (Ineligible at 0.05..0.25) spans $\ge 0.25$ s $> 0.20$ s (budget exhausted $\to$ immediate reset).

```text
+-----------------------------------------------------------------------------+
|                  STABILITY TOLERANCE INTERVAL ACCOUNTING                    |
+-----------------------------------------------------------------------------+
|                                                                             |
|  t = 0.00s : Eligible   (Timer begins accumulating)                          |
|  t = 0.05s : Ineligible (Excursion starts: _excursion_start_time = 0.05s)    |
|  t = 0.10s : Ineligible (Ongoing excursion span = 0.10 - 0.05 = 0.05s)      |
|  t = 0.15s : Eligible   (Recovery: charge 0.15 - 0.05 = 0.10s to budget)    |
|                                                                             |
|  Cumulative violation: 0.10s <= 0.20s budget. Accumulation continues.       |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### Threshold Excursion vs Hard Reset (Correction 2)
The estimator strictly distinguishes between two fundamental cases:
1. **Case A — Valid Threshold Excursion:**
   - Reliable evidence that the person temporarily moved while inside the zone (`MOVING + VALID`) or stepped briefly out of zone (`OUT_OF_ZONE + VALID`).
   - This consumes the permitted 200 ms tolerance budget without pausing overall elapsed time.
2. **Case B — Unavailable or Unreliable Evidence:**
   - Required evidence is unknown, uncertain, or invalid:
     - Person presence status $\ne$ VALID or value $\ne$ DETECTED
     - Bounding box status $\ne$ VALID or value $\ne$ VALID
     - Interaction zone status $\ne$ VALID or value == UNKNOWN
     - Movement status $\ne$ VALID (UNCERTAIN or INVALID) or value == UNKNOWN
     - Tracking loss (`track is None` or `observed == False`)
     - Temporal gap discontinuity ($\Delta t > 0.20$ s)
   - Triggers an **immediate hard reset** of the stability timer to `RESET` (elapsed=0.0s, violation=0.0s), never consuming the tolerance budget.

---

## 11. Initialization and Reset Semantics

1. **Five-Observation Rule:**
   Upon track acquisition or after a reset, motion estimators require 5 consecutive valid observed frames ($\sim 167$ ms at 30 FPS).
2. **Maximum Observation Gap:**
   If $\Delta t = t_i - t_{i-1} > 0.20\text{ s}$, temporal continuity is broken. History is cleared and initialization restarts.
3. **Hard Resets:**
   - Tracking loss (`track is None` or `observed == False`).
   - Active track ID changed.
   - Required M4 observation became `UNKNOWN`, `UNCERTAIN`, or `INVALID`.
   - Hard resets clear temporal history immediately and reset timers to `RESET`.

---

## 12. Observation Contract and Reliability Handling

M5 outputs adhere to the unified observation contract:
- **`VALID`:** Conclusive evidence computed from sufficient valid observations.
- **`UNCERTAIN`:** Active candidate or insufficient temporal coverage (e.g. 500 ms confirmation accumulating, or displacement window $< 1.0\text{ s}$).
- **`INVALID`:** Calculation impossible due to missing data, tracking loss, or non-monotonic timestamps.
- **`UNKNOWN`:** Output value when a definitive positive or negative classification cannot be formed.

---

## 13. Configuration and Parameter Reference

Configuration is managed via [configs/m5/temporal_behaviour.yaml](file:///home/eshwar/code/detection/configs/m5/temporal_behaviour.yaml):

| Parameter | Value | Status | Description |
|---|---:|---|---|
| `history_window_seconds` | 1.0 | Frozen | Maximum duration of retained observations |
| `max_observation_gap_seconds` | 0.2 | Frozen | Discontinuity threshold triggering reset |
| `initial_valid_observations` | 5 | Frozen | Consecutive valid observations for READY |
| `stationary_velocity_threshold_px_s` | 20.0 | Frozen | Upper bound on speed for stationary state |
| `stationary_displacement_threshold_px` | 20.0 | Frozen | Upper bound on net displacement in 1.0s |
| `stationary_displacement_window_seconds` | 1.0 | Frozen | Window duration required for displacement test |
| `approach_confirmation_seconds` | 0.5 | Frozen | Continuous evidence duration for approach |
| `stability_duration_seconds` | 2.0 | Frozen | Required continuous duration for stability |
| `stability_violation_tolerance_seconds` | 0.2 | Frozen | Cumulative excursion allowance |
| `approach_min_direction_cosine` | 0.5 | Provisional | Minimum direction cosine ($\approx 60^\circ$) |
| `approach_min_closing_speed_px_s` | 10.0 | Provisional | Minimum closing speed towards centroid |

---

## 14. Failure and Uncertainty Handling

| Failure Event | Component Action | Resulting M5 State |
|---|---|---|
| Track Lost / Coasting | Reject Kalman prediction; clear history | `observed=False`, `movement=UNKNOWN [INVALID]`, `stability=RESET` |
| Track ID Changed | Invalidate previous track buffer | Fresh initialization for new track ID |
| M3/M4 Person Mismatch | Refuse to process mismatched detection | Reject frame, emit `UNKNOWN [INVALID]` |
| Missing Matched Detection | Reject frame (strict association contract) | Reject frame, emit `UNKNOWN [INVALID]` |
| Observation Gap > 200 ms | Discontinuity detected; clear buffer | Transition to `RESET`, restart 5-frame counter |
| Jitter Excursion $\le 200$ ms | Consume violation budget | Continue `ACCUMULATING` elapsed time |
| Jitter Excursion $> 200$ ms | Budget exhausted | Hard reset to `RESET`, timers zeroed |
| Uncertain Movement Evidence | Hard reset (Correction 2) | Transition to `RESET`, timers zeroed |
| Centroid Undefined | Polygon $< 3$ vertices or area $< 10^{-6}$ | Raise `ValueError` at config/init time |

---

## 15. Unit Testing Strategy and Accurate Results (Correction 4)

Testing is executed using Python's standard `unittest` framework:

```bash
PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m unittest discover -s tests -t . -v
```

### Exact Per-File Test Breakdown (78 Tests Across 11 Files)

Every test file was executed independently; per-file test counts sum exactly to the reported total:

| Test File | Total | Passed | Failed | Errored | Skipped | Time (s) | Scope / Coverage |
|---|---:|---:|---:|---:|---:|---:|---|
| [test_approach_estimator.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_approach_estimator.py) | 4 | 4 | 0 | 0 | 0 | 0.001s | Direct approach, lateral motion, moving away, reset clearing. |
| [test_centroid.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_centroid.py) | 8 | 8 | 0 | 0 | 0 | 0.000s | Square, rectangle, triangle, L-shaped, orientation invariance, degenerate. |
| [test_history.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_history.py) | 7 | 7 | 0 | 0 | 0 | 0.000s | Initialization, 5-frame transition, window pruning, 200ms gap, monotonicity. |
| [test_motion_estimator.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_motion_estimator.py) | 4 | 4 | 0 | 0 | 0 | 0.000s | Constant velocity, stationary, 1.0s displacement coverage gating. |
| [test_observation_types.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_observation_types.py) | 3 | 3 | 0 | 0 | 0 | 0.000s | Enum integrity, dataclass immutability, output schema. |
| [test_same_person_association.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_same_person_association.py) | 13 | 13 | 0 | 0 | 0 | 0.009s | Strict detection match, None rejection, high IoU distinct box rejection, reacquisition, uncertainty hard resets. |
| [test_stability_timer.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_stability_timer.py) | 14 | 14 | 0 | 0 | 0 | 0.001s | Initial state, continuous timing, 10 deterministic interval scenarios (1-frame, 2-frame, boundary, exhaustion, recovery), hard resets. |
| [test_stationary_estimator.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_stationary_estimator.py) | 5 | 5 | 0 | 0 | 0 | 0.000s | Stationary criteria, strict threshold boundaries (19.9 vs 20.0), uncertainty propagation. |
| [test_synthetic_sequences.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_synthetic_sequences.py) | 6 | 6 | 0 | 0 | 0 | 0.015s | Full multi-phase end-to-end controlled trajectories. |
| [test_temporal_config.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_temporal_config.py) | 8 | 8 | 0 | 0 | 0 | 0.003s | Default validation, negative threshold rejection, domain checks, YAML loading. |
| [test_temporal_demo_runner.py](file:///home/eshwar/code/detection/tests/perception/temporal/test_temporal_demo_runner.py) | 6 | 6 | 0 | 0 | 0 | 0.053s | CLI parsing, camera open seams, headless bounded execution, option errors. |
| **M5 Dedicated Total** | **78** | **78** | **0** | **0** | **0** | **~0.10s** | **100% Pass Rate** |

### Complete Repository Regression Suite (231 Tests)

```text
Ran 231 tests in 4.305s

OK (100% Passing)
```
- **M1 Detection Tests:** 21 tests (PASS)
- **M2 Smoothing Tests:** 38 tests (PASS)
- **M3 Tracking Tests:** 48 tests (PASS)
- **M4 Readiness Tests:** 46 tests (PASS)
- **M5 Temporal Tests:** 78 tests (PASS)
- **Total Tests Passing:** 231 / 231 (0 failures, 0 errors, 0 skips)

---

## 16. Integration and Synthetic Sequence Testing

Six deterministic synthetic sequences evaluate multi-frame temporal behaviour:

- **Sequence 1 (Approach, Stop & Stabilize):** Person moves towards centroid at 100 px/s for 1.0s (`APPROACHING`), enters zone and stops for 1.25s (`STATIONARY`), remains stable for 2.1s (`COMPLETE`).
- **Sequence 2 (Moving Away):** Person moves away at 100 px/s (`NOT_APPROACHING`, `MOVING`, stability `NOT_STARTED`).
- **Sequence 3 (Lateral Movement):** Person moves perpendicular to centroid vector (`NOT_APPROACHING`, `MOVING`).
- **Sequence 4 (Tolerated Jitter):** Stationary person experiences 2 frames (0.10s) of velocity spike; stability completes successfully.
- **Sequence 5 (Exhausted Jitter):** Jitter persists for 6 frames (0.25s > 0.20s budget); stability timer resets to `RESET`.
- **Sequence 6 (Tracking Loss & Reacquisition):** Track lost mid-attempt triggers immediate reset; reacquired track restarts fresh.

---

## 17. Labelled Sequence Evaluation

### Empirical Status
As documented in the M5 specification Section 10.3 and 10.4:
- Synthetic deterministic sequences have been verified with 100% pass rate.
- Controlled real-world labelled video sequences with ground-truth temporal annotation are not currently stored in the repository.
- Therefore, empirical accuracy metrics (precision, recall, F1) on real-world datasets are **not claimed**.
- All reported thresholds are designated as frozen engineering starting points or provisional calibration parameters.

---

## 18. Detailed Performance Breakdown (Correction 5)

Measured on local Linux workstation (Intel Core processor, CPU inference, live camera `/dev/video0` @ 1920x1080, 30 frames executed via `run_temporal_demo.py`):

### Pipeline Component Latencies

| Pipeline Stage | Average (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Computational Role |
|---|---:|---:|---:|---:|---|
| **M1 YOLOv8n Person Det** | 247.92 | 129.11 | 151.93 | 2722.56 | Deep learning CNN inference on CPU (p99 includes JIT cold start) |
| **M3 Kalman Tracker** | 0.81 | 0.85 | 1.02 | 1.24 | Kalman state predict, Hungarian cost matrix, update |
| **M4 Readiness Observer** | 56.31 | 55.65 | 67.08 | 83.52 | MediaPipe face mesh, landmark orientation, polygon test |
| **M5 Temporal Estimator (Total)** | **0.18** | **0.15** | **0.27** | **0.38** | **Unified temporal behavioural perception pipeline** |
| *-- History Management* | 0.02 | 0.02 | 0.05 | 0.08 | Monotonic ordering, gap validation, 1.0s deque pruning |
| *-- Motion Estimation* | 0.02 | 0.01 | 0.06 | 0.10 | Linear regression velocity $(v_x, v_y)$, speed $s_t$, displacement $D_t$ |
| *-- Stationary Classifier* | 0.01 | 0.01 | 0.02 | 0.02 | Strict inequality threshold checks ($s_t < 20$, $D_t < 20$) |
| *-- Approach Estimator* | 0.05 | 0.04 | 0.08 | 0.22 | Direction cosine $\cos(\theta)$, closing speed $-b_d$, confirmation |
| *-- Stability Timer* | 0.02 | 0.02 | 0.03 | 0.04 | Interval accounting, cumulative tolerance, hard-reset check |

### Resource Utilization
Measured dynamically across live execution using `psutil`:
- **CPU Utilization:** Average: 623.7% | Peak: 705.5% (multi-threaded PyTorch and MediaPipe thread pool across cores).
- **Process Resident Memory (RSS):** Average: 1033.2 MB | Peak: 1041.7 MB (holding YOLOv8n weights, MediaPipe models, and frame buffers).
- **M5 Specific Overhead:** $< 0.20$ ms latency and $< 10$ KB heap space for the 30-observation ring buffers.

---

## 19. Demonstration Runner and Visual Overlays

The demonstration runner [perception/temporal/run_temporal_demo.py](file:///home/eshwar/code/detection/perception/temporal/run_temporal_demo.py) provides a complete live visual interface:

```text
+-----------------------------------------------------------------------------------------+
| === M5 TEMPORAL BEHAVIOUR PERCEPTION ===                                                |
| FPS: 10.0 | Latency: 129.1ms YOLO, 0.8ms Trk, 55.6ms M4, 0.2ms M5                       |
| Track ID: 1 | Init: READY (5/5) | Observed: True                                        |
| Motion: Speed: 4.2 px/s | Disp: 3.1 px/1s | Dist to Zone: 42.0 px                       |
| Vector: cos(theta): -0.12 | Closing: -1.5 px/s                                          |
| Approach: NOT_APPROACHING [VALID]                                                       |
| Movement: STATIONARY [VALID]                                                            |
| Stability: ACCUMULATING | Elapsed: 1.45s/2.00s | Excursion: 0.05s/0.20s                   |
|                                                                                         |
|       +---------------------------------------+                                         |
|       | Interaction Zone Polygon              |                                         |
|       |                                       |                                         |
|       |               (X) Centroid            |                                         |
|       |                ^                      |                                         |
|       |                |                      |                                         |
|       |              [ o ] Person Center      |                                         |
|       |             +-----+                   |                                         |
|       |             |     | Bounding Box      |                                         |
|       |             +-----+                   |                                         |
|       +---------------------------------------+                                         |
+-----------------------------------------------------------------------------------------+
```

---

## 20. Limitations and Unverified Claims

1. **Perspective Distortion:** Calculations operate in 2D image pixel space, not 3D metric ground-plane coordinates.
2. **Provisional Thresholds:** $\tau_\theta = 0.5$ and $\tau_a = 10$ px/s require empirical tuning when deployed on physical robot hardware.
3. **No Learned Behavior:** The system relies on deterministic geometric heuristics.
4. **No Contact Readiness:** M5 emits behavioral observations; it does not declare the robot ready to speak or initiate contact (reserved for M6).

---

## 21. Final M5 Acceptance Checklist

- [x] **Timestamped Bounding-Box Center History:** Implemented in `history.py`.
- [x] **Five-Consecutive-Observation Initialization:** Implemented and tested.
- [x] **One-Second Bounded History:** Implemented in `history.py`.
- [x] **200 ms Maximum Observation Gap:** Enforced; resets history on discontinuity.
- [x] **Image-Space Velocity Estimation:** Timestamp-aware linear regression implemented in `motion_estimator.py`.
- [x] **1.0s Displacement Estimation:** Implemented with temporal coverage validation.
- [x] **Centroid-Directed Approach Detection:** Direction cosine and distance trend implemented in `approach_estimator.py`.
- [x] **500 ms Approach Confirmation:** Enforced using monotonic timestamps.
- [x] **Stationary Predicate:** Strict $< 20$ px/s and $< 20$ px thresholds in `stationary_estimator.py`.
- [x] **2.0s Stability Timer:** Continuous timing implemented in `stability_timer.py`.
- [x] **200 ms Cumulative Tolerance Interval Accounting (Correction 1):** Verified with 10 deterministic test cases.
- [x] **Stability Uncertainty vs Excursion Distinction (Correction 2):** Valid threshold excursions consume budget; UNKNOWN/UNCERTAIN trigger hard reset.
- [x] **Strict M3–M4 Same-Person Association (Correction 3):** Strict detection equality enforced; IoU fallback removed.
- [x] **Accurate Test Counts (Correction 4):** 78/78 dedicated tests passing across 11 files; exact sum verified.
- [x] **Detailed Latency Breakdown & Resource Metrics (Correction 5):** p50, p95, p99 percentiles and psutil CPU/RSS utilization reported.
- [x] **Predecessor Tests Passing:** 153/153 M1–M4 regression tests passing (231/231 total).
- [x] **Live Demonstration Runner:** Tested and working in `run_temporal_demo.py`.
- [x] **Configuration Validation:** Validated in `temporal_config.py` and `temporal_behaviour.yaml`.
- [x] **Documentation Created:** Implementation details recorded in `docs/m5_implementaion_details.md`.
