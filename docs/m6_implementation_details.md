# Milestone M6 — Readiness-Based Single-Person First Contact
## Implementation and Engineering Validation Report

**Document:** [`docs/m6_implementation_details.md`](file:///home/eshwar/code/detection/docs/m6_implementation_details.md)  
**Milestone:** M6 — Readiness-Based Single-Person First Contact  
**System:** Vision-Based First Contact — Reception Social Robot  
**Authoritative Specification:** [`docs/m6_readiness_single_person_first_contact.md`](file:///home/eshwar/code/detection/docs/m6_readiness_single_person_first_contact.md)  
**Implementation Status:** Complete, Hardened & Verified (328/328 Repository Tests Passing, 97/97 M6 Tests Passing)  

---

## 1. Executive Summary

Milestone **M6** integrates the complete perception pipeline developed across predecessor milestones (**M1** person detection, **M2** bounding-box smoothing, **M3** single-person Kalman tracking, **M4** frame-level readiness observations, and **M5** temporal behaviour estimation) into an authoritative, fail-closed, single-person first-contact decision layer.

The tabletop reception robot evaluates whether a single tracked visitor has satisfied all physical and behavioral criteria required for spontaneous social engagement. When all seven readiness predicates remain continuously satisfied for two unbroken seconds ($T_{\text{eligible}} = 2.0\text{ s}$), the system emits a structured, one-shot `FIRST_CONTACT_READY` event.

```text
+----------------------------------------------------------------------------------------------------+
|                                    M6 SYSTEM PIPELINE FLOWCHART                                    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   [Camera Frame: 1280x720 @ 30 FPS]                                                                |
|                   |                                                                                |
|                   v                                                                                |
|   [M1 / M2: Person Detection & BBox Smoothing (YOLOv8n + EMA)]                                     |
|                   |                                                                                |
|                   v                                                                                |
|   [M3: Single-Person Kalman Tracker (8-D State, IoU Matcher, Active Target Lock)]                    |
|                   |                                                                                |
|                   v                                                                                |
|   [M4: Frame-Level Readiness Observer (Presence P, BBox B, Zone Z, Orientation O)]                 |
|                   |                                                                                |
|                   v                                                                                |
|   [M5: Temporal Behaviour Estimator (Approach A, Stationary Movement S)]                           |
|                   |                                                                                |
|                   v                                                                                |
|   [M6: Distance Provider (Mock Provider d=1.20m / Metric Provider Fail-Closed)]                     |
|                   |                                                                                |
|                   +--------------------------------+                                               |
|                                                    |                                               |
|                                                    v                                               |
|   +--------------------------------------------------------------------------------------------+   |
|   |                           M6 READINESS ENGINE ORCHESTRATION                                |   |
|   +--------------------------------------------------------------------------------------------+   |
|   |  1. VisitManager: Target locking, 5.0s loss grace period, zone-exit termination, A-persist  |   |
|   |  2. PredicateEvaluator: E(t) = P(t) & B(t) & A(t) & Z(t) & D(t) & S(t) & O(t)              |   |
|   |  3. EligibilityTimer: Continuous 2.0s monotonic timer with max inter-frame gap check       |   |
|   |  4. ReadinessStateMachine: NOT_READY <--> PENDING <--> READY                               |   |
|   |  5. Event Generator: One-shot structured FIRST_CONTACT_READY event per visit               |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                    |                                               |
|                   +--------------------------------+                                               |
|                   |                                                                                |
|                   v                                                                                |
|   [FirstContactResult: State, Predicates, Reasons, Event, Timings]                                 |
|                   |                                                                                |
|                   v                                                                                |
|   [Live HUD Overlay / Downstream Interaction Trigger]                                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Definition of Readiness Predicates

Readiness for first contact is evaluated conjunctively across seven orthogonal predicates. The composite readiness decision function $E_i(t)$ for active track $i$ at timestamp $t$ is strictly defined as:

$$E_i(t) = P_i(t) \land B_i(t) \land A_i(t) \land Z_i(t) \land D_i(t) \land S_i(t) \land O_i(t)$$

```text
+----------------------------------------------------------------------------------------------------+
|                                  SEVEN-PREDICATE CONJUNCTION TREE                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    P: Person Presence (M1/M3) ------------+                                                        |
|    B: Bounding-Box Valid (M1/M4) ---------+                                                        |
|    A: Approach / Initial Presence --------+                                                        |
|    Z: Interaction Zone Inside (M4) -------+---> [ ALL SEVEN TRUE? ] -- YES --> E_i(t) = True       |
|    D: Distance <= 1.50m (M6) -------------+           |                                            |
|    S: Stationary Velocity < 20px/s (M5) --+           NO                                           |
|    O: Facing Robot Head Pose (M4) --------+           |                                            |
|                                                       v                                            |
|                                                E_i(t) = False                                      |
|                                            (Explainable Reason Codes)                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

| Symbol | Predicate Name | Source | Evaluation Rule | Failure Reason Code |
|---|---|---|---|---|
| **$P_i(t)$** | Person Presence | M1 + M3 | Detection exists in current frame AND `track.observed == True` AND `track.track_id == active_track_id`. Kalman coasting/predicted boxes cannot satisfy $P$. | `NO_PERSON` / `TRACKING_LOST` / `TARGET_MISMATCH` |
| **$B_i(t)$** | Bounding-Box Validity | M1 + M4 | Detection box is valid AND strictly matches M4 original detection coordinates (`IoU == 1.0`). | `INVALID_BBOX` |
| **$A_i(t)$** | Approach / Initial Presence | M5 / M6 | **Path 1:** M5 `APPROACHING + VALID` confirmed for active track ID, OR<br>**Path 2:** Reliable initial presence inside interaction zone at visit start (`INSIDE + VALID`). Persists for visit duration. | `APPROACH_NOT_CONFIRMED` |
| **$Z_i(t)$** | Interaction-Zone Membership | M4 | Person bounding-box footpoint/polygon containment evaluates to `INSIDE + VALID`. | `OUTSIDE_INTERACTION_ZONE` |
| **$D_i(t)$** | Distance Threshold | M6 Provider | Distance observation is `VALID` AND measured distance $d \le 1.50\text{ m}$ AND matches active target track ID if assigned. | `TOO_FAR` / `DISTANCE_UNAVAILABLE` / `DISTANCE_UNRELIABLE` |
| **$S_i(t)$** | Stationary Behaviour | M5 | Movement velocity $< 20.0\text{ px/s}$ AND displacement $< 20.0\text{ px}$ (`STATIONARY + VALID`) for active track ID. | `PERSON_MOVING` / `TARGET_MISMATCH` |
| **$O_i(t)$** | Head Orientation | M4 | MediaPipe face mesh yaw/pitch orientation evaluates to `FACING_ROBOT + VALID`. | `NOT_FACING_ROBOT` / `PERCEPTION_UNCERTAIN` |

If all seven predicates evaluate to `True`, $E_i(t) = \text{True}$. If any single condition is `False`, $E_i(t) = \text{False}$, and the unsatisfied predicates generate deterministic reason codes for real-time operator HUD diagnostics.

---

## 3. Continuous Eligibility Timing

Readiness requires continuous, uninterrupted satisfaction of all seven predicates:

$$T_{\text{eligible}} = 2.0\text{ seconds}$$

```text
+----------------------------------------------------------------------------------------------------+
|                                    CONTINUOUS ELIGIBILITY TIMELINE                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Timestamp (s):  1.0          1.5          2.0          2.5          3.0          3.5              |
|  All 7 Valid?    True         True         True         True         True         False (looked)   |
|  Timer State:    STARTED      RUNNING      RUNNING      RUNNING      COMPLETE     RESET            |
|  Elapsed Time:   0.0s         0.5s         1.0s         1.5s         2.0s         0.0s             |
|  Machine State:  PENDING      PENDING      PENDING      PENDING      READY        NOT_READY        |
|  Event Emitted:  None         None         None         None         EVT_READY    None             |
|                                                                                                    |
|                                                                                                    |
|  Scenario: Excursion / Interruption Reset                                                           |
|  Timestamp (s):  4.0          4.5          4.8 (loss)   5.0 (reacq)  6.0          7.0              |
|  All 7 Valid?    True         True         False        True         True         True             |
|  Timer State:    STARTED      RUNNING      RESET        STARTED      RUNNING      COMPLETE         |
|  Elapsed Time:   0.0s         0.5s         0.0s         0.0s         1.0s         2.0s             |
|  Machine State:  PENDING      PENDING      NOT_READY    PENDING      PENDING      READY (re-entry) |
|  Event Emitted:  None         None         None         None         None         None (one-shot!) |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Strict Reset Properties

1. **Monotonic Progression**: Evaluated using monotonic clock differences $\Delta t = t_k - t_{k-1}$.
2. **Immediate Fail-Closed Reset**: Any frame where $E_i(t) = \text{False}$ instantly resets accumulated elapsed time to `0.0s`.
3. **No Resumption**: Previously accumulated time is discarded; after an excursion, the candidate must satisfy all conditions continuously for a full 2.0 seconds.
4. **Inter-Frame Observation Gap Bound (`max_inter_frame_gap_seconds: 0.5s`)**: If consecutive frames are separated by $\Delta t > 0.5\text{ s}$, continuity is broken; the timer resets accumulated time to `0.0s` and restarts accumulation from the current frame.
5. **Non-Monotonic & Finite Clock Protection**: If backward ($\Delta t < 0$) or non-finite (NaN, $\pm\infty$) timestamps occur, `EligibilityTimer` resets and raises `ValueError`.

---

## 4. Single-Person Readiness State Machine

The readiness state machine operates over three discrete states:

```text
+----------------------------------------------------------------------------------------------------+
|                                   READINESS STATE TRANSITIONS                                      |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                    +-------------------------------------------------------+                       |
|                    |                                                       |                       |
|                    v                                                       |                       |
|             +--------------+     E_i(t) == True, elapsed < 2.0s     +--------------+               |
|  Init ----> |  NOT_READY   | -------------------------------------> |   PENDING    |               |
|             +--------------+                                        +--------------+               |
|                    ^   ^                                                   |   |                   |
|                    |   |         E_i(t) == False (Condition Lost)          |   |                   |
|                    |   +---------------------------------------------------+   |                   |
|                    |                                                           |                   |
|                    |              E_i(t) == True, elapsed >= 2.0s              |                   |
|                    |   +-------------------------------------------------------+                   |
|                    |   |                                                                           |
|                    |   v                                                                           |
|             +--------------+                                                                       |
|             |    READY     | -- (One-shot FIRST_CONTACT_READY event emitted on first entry)        |
|             +--------------+                                                                       |
|                    |                                                                               |
|                    +------------- E_i(t) == False (Revocation) --------------------+               |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 State Definitions

- **`NOT_READY`**: One or more readiness predicates are unsatisfied, or tracking is lost.
- **`PENDING`**: All seven predicates are simultaneously satisfied, but continuous eligibility is $< 2.0\text{ s}$.
- **`READY`**: All seven predicates have remained satisfied continuously for $\ge 2.0\text{ s}$.

### 4.2 Revocable Readiness vs. One-Shot Event

- **Revocable State**: `READY` is a live state, not a latch. If the person turns their head away, steps back, or begins moving, the state machine immediately transitions back to `NOT_READY`.
- **One-Shot Event**: The `FIRST_CONTACT_READY` event is emitted **exactly once per visit** on the initial transition from `PENDING` into `READY`. If readiness is subsequently revoked and regained within the same visit, the state returns to `READY`, but no duplicate event is emitted.

---

## 5. Visit Lifecycle Management

The `VisitManager` maintains persistent identity and target locking for a single person visit:

```text
+----------------------------------------------------------------------------------------------------+
|                                   VISIT LIFECYCLE STATE MACHINE                                    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Person Observed] -------------------> [Visit Created: visit_id locked to track_id]               |
|                                                     |                                              |
|                     +-------------------------------+-------------------------------+              |
|                     |                                                               |              |
|                     v                                                               v              |
|  [Tracking Loss: Target Unobserved]                           [Person Enters Zone & Leaves Zone]   |
|                     |                                                               |              |
|                     v                                                               v              |
|  [Grace Period Active: T_loss <= 5.0s]                               [Reliable Zone Exit]          |
|         |                        |                                                  |              |
|         v                        v                                                  v              |
|  [Reacquired < 5.0s]     [Timeout >= 5.0s]                                  [Immediate Visit]      |
|  (Retain visit_id &      (Terminate Visit,                                  [Termination]          |
|   event_emitted flag)     Next Person gets new visit)                                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 5.1 Five-Second Tracking Loss Grace Period

- When active tracking is interrupted (occlusion, detection dropout, or Kalman coasting):
  1. The readiness state machine drops immediately to `NOT_READY` (fail-closed).
  2. The `VisitManager` retains the active `visit_id` and target lock for up to $T_{\text{loss}} = 5.0\text{ s}$.
  3. If the target is reacquired within 5.0s, the same visit continues with `event_emitted` status preserved.
  4. If tracking is not reacquired within 5.0s, the visit terminates with reason `TRACKING_LOSS_TIMEOUT`.

### 5.2 Reliable Zone Exit Termination

- If a tracked person who has entered the interaction zone subsequently steps outside (`InteractionZoneValue.OUTSIDE` with `ObservationStatus.VALID`), the visit terminates **immediately** with reason `RELIABLE_ZONE_EXIT`.
- This immediate termination applies **strictly** to observations belonging to the locked active target, preventing bystanders outside the zone from accidentally terminating an active interaction.

---

## 6. Distance-Provider Integration & Production Safety

### 6.1 Distance Modes

1. **`MOCK` Mode (Default for Development / Benchmarking)**:
   - Configured via [`configs/m6/first_contact.yaml`](file:///home/eshwar/code/detection/configs/m6/first_contact.yaml) (`distance_mode: "mock"`, `mock_distance_m: 1.20`).
   - Clearly tagged with `DistanceSource.MOCK`.
   - Annotated with a prominent warning banner on the HUD overlay (`[MOCK DISTANCE: 1.20m]`).
   - Fully mutable at runtime via the demo CLI `--distance <metres>` flag.
2. **`METRIC` Mode (Production Readiness)**:
   - Queries calibrated depth sensor / stereo baseline.
   - Tagged with `DistanceSource.METRIC`.
   - **Fail-Closed Safety**: In the absence of calibrated hardware, returns `ObservationStatus.INVALID`, causing predicate $D$ to evaluate to `False` (`DISTANCE_UNAVAILABLE`) and preventing premature contact readiness.
   - Enforced by configuration validation when `require_metric_distance_in_production: true`.

---

## 7. Structured Event Schema

When first contact readiness is achieved, the engine generates an immutable, structured event:

```json
{
  "event_type": "FIRST_CONTACT_READY",
  "event_id": "evt_4a7f9b2c",
  "visit_id": "visit_8c31e0fa",
  "track_id": 1,
  "timestamp": 1795079805.421,
  "readiness_duration": 2.0,
  "distance": {
    "value_m": 1.20,
    "source": "MOCK"
  }
}
```

---

## 8. Verification & Comprehensive Test Breakdown

The complete M6 test suite comprises **97 dedicated unit, integration, and scenario tests** across 10 test modules. Combined with predecessor suites (M1–M5), the repository possesses **328 automated tests, 100% passing**.

### 8.1 Per-File Test Breakdown

| Test Suite File | Tests | Coverage Scope | Status |
|---|---|---|---|
| [`test_contact_types.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_contact_types.py) | 8 | Data models, enum mappings, event serialization, timestamp/duration validation, reason codes | Pass |
| [`test_contact_config.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_contact_config.py) | 10 | Default configs, NaN/inf validation, YAML parsing, production safeguards, inter-frame gap | Pass |
| [`test_distance_provider.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_distance_provider.py) | 4 | Mock distance mutation, metric fail-closed uncalibrated status, track_id propagation | Pass |
| [`test_eligibility_timer.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_eligibility_timer.py) | 11 | Monotonic timing, failure reset, max inter-frame gap reset, NaN/inf safety | Pass |
| [`test_predicates.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_predicates.py) | 16 | Target identity gating, stale frame checks, timestamp mismatch, reason codes | Pass |
| [`test_visit_manager.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_visit_manager.py) | 16 | Target locking, 5s grace period, other-person zone exit isolation, approach persistence | Pass |
| [`test_readiness_state_machine.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_readiness_state_machine.py) | 10 | NOT_READY $\leftrightarrow$ PENDING $\leftrightarrow$ READY, revocability, one-shot event generation | Pass |
| [`test_readiness_engine.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_readiness_engine.py) | 8 | Pipeline orchestration, predicate evaluation, latency instrumentation, reset | Pass |
| [`test_synthetic_scenarios.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_synthetic_scenarios.py) | 7 | Scenarios A–G (approach, startup presence, tracking loss, timeout, recovery, multi-person) | Pass |
| [`test_contact_demo_runner.py`](file:///home/eshwar/code/detection/tests/perception/contact/test_contact_demo_runner.py) | 7 | CLI parsing, parameter verification, camera failure cleanup, bounded runner | Pass |
| **Total M6 Dedicated Tests** | **97** | **Complete Decision & Integration Coverage** | **100% Pass** |

### 8.2 Full Repository Milestone Regression Test Summary

```text
+----------------------------------------------------------------------------------------------------+
|                               FULL REPOSITORY TEST BREAKDOWN (M1–M6)                               |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   Milestone M1: Basic Person Detection Tests ....................................  21 Passed       |
|   Milestone M2: BBox Smoothing & Calibration Tests ...............................  38 Passed       |
|   Milestone M3: Single-Person Kalman Tracking Tests ..............................  48 Passed       |
|   Milestone M4: Frame-Level Readiness & Face Landmarker Tests ...................  46 Passed       |
|   Milestone M5: Temporal Behaviour & Approach Tracking Tests ....................  78 Passed       |
|   Milestone M6: Readiness-Based Single-Person First Contact Tests ...............  97 Passed       |
|                                                                                                    |
|   TOTAL REPOSITORY AUTOMATED TESTS: ............................................ 328 PASSED (100%) |
|   REGRESSION FAILURES: .........................................................   0               |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 9. Workstation Benchmarking & Latency Breakdown

Empirical performance profiling was conducted on the development workstation using live camera capture on `/dev/video0` (1280×720 stream) running headless over 30 representative frames:

| Pipeline Stage | Component | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| **M1** | YOLOv8n Person Detection (CPU) | 375.44 | 182.59 | 400.42 | 3961.24* |
| **M3** | Kalman Tracker & Data Association | 0.76 | 0.78 | 1.09 | 1.26 |
| **M4** | Face Landmarker & Readiness Observer | 64.55 | 62.74 | 81.86 | 90.53 |
| **M5** | Temporal Estimator & Approach Filter | 0.13 | 0.13 | 0.20 | 0.22 |
| **M6** | **Readiness Engine & State Machine** | **0.09** | **0.08** | **0.14** | **0.15** |

*\*Note: M1 p99 latency reflects the initial Torch/YOLO JIT compilation and graph initialization during cold frame 0.*

---

## 10. Narrow Corrections Pass & Contract Hardening

Following architectural review of Milestone M6, targeted fixes were applied to eliminate state corruption, identity mismatches, unvalidated continuity claims, and numerical configuration flaws.

### 10.1 Active-Target Identity Gating (Priority: P0)

In multi-person scenes, M1/M2 can detect multiple people. M3 tracks a single active target with a Kalman filter. However, previously M4 frame observations or M5 temporal observations could be evaluated against the active visit even if they originated from a different person or unassociated detection.

```text
+----------------------------------------------------------------------------------------------------+
|                               ACTIVE-TARGET IDENTITY GATING ARCHITECTURE                           |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   Active Target Track (track_id = K)                                                               |
|        |                                                                                           |
|        +---> Predicate P: Requires track.track_id == active_track_id                               |
|        +---> Predicate B: Requires matched_detection bbox == m4_person.bbox_metrics                |
|        +---> Predicate A: Confirmed approach requires temporal_obs.track_id == active_track_id     |
|        +---> Predicate S: Stationary requires temporal_obs.track_id == active_track_id             |
|        +---> Predicate D: Distance observation tagged with track_id == active_track_id             |
|        +---> Visit Manager: Another person stepping OUTSIDE cannot terminate track K's visit       |
|                                                                                                    |
|   Unassociated Person (track_id = J != K)                                                          |
|        |                                                                                           |
|        +---> Emits ReasonCode.TARGET_MISMATCH -> Fails closed instantly                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

1. **Predicates Gated**: In [`perception/contact/predicates.py`](file:///home/eshwar/code/detection/perception/contact/predicates.py), predicates $P, B, S, D$ explicitly verify identity match:
   - $P$ fails with `TARGET_MISMATCH` if `track.track_id != active_track_id`.
   - $B$ fails with `INVALID_BBOX` if detection bounding box does not match M4 bbox.
   - $S$ fails with `TARGET_MISMATCH` if `temporal_observation.track_id != active_track_id`.
   - $D$ fails with `TARGET_MISMATCH` if `distance_observation.track_id != active_track_id`.
2. **Visit Isolation**: In [`perception/contact/visit_manager.py`](file:///home/eshwar/code/detection/perception/contact/visit_manager.py), zone entry, zone exit, and confirmed approach checks require `is_active_target_observed` and `is_m4_active_person`. Another person stepping `OUTSIDE` cannot terminate the active target's visit.

### 10.2 Timestamp Freshness and Consistency Checks (Priority: P0)

Readiness evaluation requires consistent temporal evidence across all pipeline stages:
1. **Frame Staleness**: If $|t_{\text{eval}} - t_{\text{frame}}| > t_{\text{freshness}}$ (default 0.5s), $P, B, Z, O$ evaluate to `False` with `ReasonCode.STALE_OBSERVATION`.
2. **M3 Track Consistency**: If $|t_{\text{frame}} - t_{\text{track}}| > t_{\text{freshness}}$, `ReasonCode.TIMESTAMP_MISMATCH` is emitted.
3. **M5 Temporal Consistency**: If $|t_{\text{eval}} - t_{\text{temporal}}| > t_{\text{freshness}}$, predicate $S$ fails with `STALE_OBSERVATION`.
4. **M6 Distance Consistency**: If $|t_{\text{eval}} - t_{\text{dist}}| > t_{\text{freshness}}$, predicate $D$ fails with `DISTANCE_UNAVAILABLE`.

### 10.3 The M3/M6 Reacquisition Contract Conflict & Resolution (Priority: P0)

#### Architectural Conflict Analysis

The M3 tracking specification (`retention_timeout_seconds: 2.0s`) and M6 visit management specification (`tracking_loss_timeout_seconds: 5.0s`) presented a fundamental identity continuity dilemma:

```text
+----------------------------------------------------------------------------------------------------+
|                               THE M3 / M6 LIFECYCLE REACQUISITION CONFLICT                         |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Time (s):         0.0           2.0           3.5                        5.0                      |
|  Active Track:     Track 1       Track 1 lost  Track 2 created            --                       |
|  M3 Tracker:       [Active]      [Expired]     [New Auto-Increment Track] [Running]                |
|                                                                                                    |
|  M6 Visit Manager: [Active Visit 1 locked to Track 1] ------------------> [Grace Period Expires]  |
|                                                ^                                                   |
|                                                | Track 2 rejected as TARGET_MISMATCH               |
|                                                +-- Visit 1 stalled until 5.0s                      |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

- In M3 (`perception/tracking/single_person_tracker.py`), when a person is lost for $> 2.0\text{ s}$, the Kalman track expires and is deleted. Any subsequent person detection generates a new track ID (e.g., Track 2).
- In M6, `VisitManager` retains `active_visit_id` locked to `active_track_id = 1` for up to 5.0 seconds.
- Without a validated appearance ReID model, M6 cannot prove that Track 2 is the same physical human as Track 1. If Track 2 is accepted, the system would falsely assert identity continuity across a 3.5s blackout.

#### Resolution Options Evaluated

1. **Option A (Increase M3 Retention to 5.0s)**:
   - *Impact*: Modifies frozen M3 tracker.
   - *Risk*: A Kalman filter coasting without measurement updates for 150 frames (5.0s) experiences massive state covariance growth, drifting wildly across the scene and risking false association with bystanders.
2. **Option B (Add Appearance Re-Identification Model)**:
   - *Impact*: Requires introducing an appearance feature extractor (e.g. OSNet / ResNet embedding) and gallery matcher.
   - *Risk*: Violates project scope and lightweight CPU execution premise; requires new unvalidated dependencies.
3. **Option C (Strictly Enforce Fail-Closed Identity Bound — Adopted)**:
   - *Impact*: Preserves frozen M3 and M6 contracts with mathematical honesty.
   - *Behavior*:
     * **Short Tracking Loss ($\le 2.0\text{ s}$)**: M3 preserves Kalman track ID (`track_id = 1`). M6 successfully reacquires the same visit within its 5.0s grace period.
     * **Extended Tracking Loss ($> 2.0\text{ s}$)**: M3 expires Track 1 and generates Track 2. M6 rejects Track 2 during the remainder of the 5.0s grace period (`TARGET_MISMATCH`). When the 5.0s grace period expires, Visit 1 terminates (`TRACKING_LOSS_TIMEOUT`), and Track 2 initiates a fresh visit.
   - *Outcome*: 100% fail-closed identity guarantee without spoofing or unvalidated assumptions.

### 10.4 Continuous Observation Gap Enforcement (Priority: P1)

A system receiving sparse or strobed observations (e.g. 1 frame per second) could theoretically accumulate two timestamps separated by 2.0 seconds while missing intermediate movements or head turns.

To guarantee true physical continuity:
- Introduced `max_inter_frame_gap_seconds` (default `0.5s`, configurable in YAML).
- In [`perception/contact/eligibility_timer.py`](file:///home/eshwar/code/detection/perception/contact/eligibility_timer.py):
  $$\Delta t_{\text{frame}} = t_k - t_{k-1} > \text{max\_inter\_frame\_gap\_seconds} \implies \text{Reset Accumulated Time to 0.0s}$$
- Continuity is broken if camera framerate drops below 2 FPS, immediately failing closed and restarting the 2.0s accumulation.

### 10.5 Initial-Presence Semantics Clarification (Priority: P1)

The specification defines two paths for predicate $A$:
- **Path 1 (Confirmed Approach)**: Observed approaching the robot outside the interaction zone, followed by stopping.
- **Path 2 (Initial Presence)**: Person is already inside the interaction zone (`INSIDE + VALID`) at visit start.

*Clarification*:
- Path 2 applies to **system startup** (initial power-on with person present) and to **subsequent independent visits** initiated directly inside the interaction zone (e.g., after an earlier visitor exited and a new person walked in during a camera pause).
- Once confirmed, $A$ persists for the entire lifecycle of that specific `visit_id`.

### 10.6 Camera Failure & Shutdown Lifecycle (Priority: P1)

In [`perception/contact/run_first_contact_demo.py`](file:///home/eshwar/code/detection/perception/contact/run_first_contact_demo.py):
- Handled camera acquisition failures (`cap.read() == False` or open failure).
- Added `try...finally` resource cleanup block ensuring `readiness_engine.reset()` is invoked upon loop exit, clearing all visit locks, eligibility timers, and distance state.
- Released OpenCV windows and underlying models cleanly.

### 10.7 Numerical Configuration Validation (Priority: P1)

In [`perception/contact/contact_config.py`](file:///home/eshwar/code/detection/perception/contact/contact_config.py):
- Added explicit checks rejecting `math.isnan()` and `math.isinf()` across all float configurations (`eligibility_duration_seconds`, `tracking_loss_timeout_seconds`, `distance_max_m`, `mock_distance_m`, `observation_freshness_seconds`, `max_inter_frame_gap_seconds`).
- Rejects negative or non-positive parameters with descriptive `ValueError` messages.

---

## 11. Operational Constraints & Limitations

1. **Single-Person Focus**: In accordance with the frozen architecture, M6 evaluates only the single active tracked person. Multi-person simultaneous readiness evaluation is explicitly out of scope.
2. **Mock Distance Mode Disclosure**: Mock distance does not represent validated metric range. Production deployment requires calibrated depth/stereo sensors.
3. **Headless Execution Compatibility**: The demo runner supports headless profiling via `--headless`, preventing crashes in displayless test and server environments.
4. **Lighting & Head Pose Range**: MediaPipe Face Landmarker requires adequate illumination and face visibility ($\pm 25^\circ$ yaw, $\pm 20^\circ$ pitch) for orientation confirmation.
