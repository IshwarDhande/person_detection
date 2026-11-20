# Project Architecture, Control & Data Flow Diagrams

> **Repository:** Vision-Based Single-Person First Contact Readiness Pipeline (Milestones M1–M6)  
> **Purpose:** Detailed architectural control-flow, data-flow, state machine, and decision logic diagrams rendered in clear ASCII.

---

## Table of Contents

1. [High-Level End-to-End System Architecture](#1-high-level-end-to-end-system-architecture)
2. [Per-Frame Pipeline Control Flow](#2-per-frame-pipeline-control-flow)
3. [Milestone M3: Single-Person Kalman Tracking Lifecycle](#3-milestone-m3-single-person-kalman-tracking-lifecycle)
4. [Milestone M4: Frame-Level Stationary Readiness Observation](#4-milestone-m4-frame-level-stationary-readiness-observation)
5. [Milestone M5: Temporal Behaviour & Approach Estimation](#5-milestone-m5-temporal-behaviour--approach-estimation)
6. [Milestone M6: Visit Lifecycle Management & Target Locking](#6-milestone-m6-visit-lifecycle-management--target-locking)
7. [Milestone M6: Seven Readiness Predicates & Decision Gate](#7-milestone-m6-seven-readiness-predicates--decision-gate)
8. [Milestone M6: Readiness State Machine & Event Isolation](#8-milestone-m6-readiness-state-machine--event-isolation)
9. [Global Monotonic Timestamp & Frame Consistency Validation](#9-global-monotonic-timestamp--frame-consistency-validation)
10. [Diagnostic HUD & Visual Feedback Layer](#10-diagnostic-hud--visual-feedback-layer)

---

## 1. High-Level End-to-End System Architecture

The following diagram illustrates how raw camera frames flow through the layered milestones (M1–M6) to generate reliable first-contact events:

```text
+--------------------------------------------------------------------------------------------------------------------+
|                                    END-TO-END PIPELINE ARCHITECTURE (M1–M6)                                        |
+--------------------------------------------------------------------------------------------------------------------+
|                                                                                                                    |
|   +-----------------------+                                                                                        |
|   |   USB / MIPI Camera   |  (/dev/video0, 1280x720 / 1920x1080 @ 30 FPS)                                         |
|   +-----------------------+                                                                                        |
|               |                                                                                                    |
|               v  cv2.VideoCapture (BGR Frame + Monotonic Timestamp)                                                |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | MILESTONE M1: Person Detection ([person_detector.py](file:///home/eshwar/code/detection/perception/detection/person_detector.py))                                                     |
|   |   - YOLOv8n inference (640x640 letterbox)                                                                  |
|   |   - Filter class_id == 0 (Person), confidence >= 0.50                                                      |
|   |   - Output: List[PersonDetection(bbox_xyxy, confidence, class_id)]                                         |
|   +------------------------------------------------------------------------------------------------------------+   |
|               |                                                                                                    |
|               +-------------------------------------------------------------+                                      |
|               | Raw Detections                                              | Raw Detections                       |
|               v                                                             v                                      |
|   +---------------------------------------+                     +----------------------------------------------+   |
|   | MILESTONE M3: Kalman Tracking         |                     | MILESTONE M4: Frame Readiness Observation    |   |
|   | ([single_person_tracker.py](file:///home/eshwar/code/detection/perception/tracking/single_person_tracker.py))            |                     | ([frame_observer.py](file:///home/eshwar/code/detection/perception/readiness/frame_observer.py))                           |   |
|   |   - 8-D Constant Velocity Filter      |                     |   - Interaction Zone Polygon PIP Test        |   |
|   |   - Spatial IoU Association (>= 0.3)  |                     |   - MediaPipe Face Landmarker & Mesh         |   |
|   |   - Single Active Target Lock         |                     |   - EPnP 3D Head Pose (Yaw / Pitch)          |   |
|   |   - 2.0s Coasting Retention           |                     |   - Bounding Box Metrics & Clipping          |   |
|   |   - Output: Track(id, bbox, observed) |                     |   - Output: FrameObservation                 |   |
|   +---------------------------------------+                     +----------------------------------------------+   |
|               |                                                                 |                                  |
|               | Track + Matched Detection                                       | FrameObservation                 |
|               +-------------------------------+---------------------------------+                                  |
|                                               |                                                                    |
|                                               v                                                                    |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | MILESTONE M5: Temporal Behaviour Estimation ([temporal_estimator.py](file:///home/eshwar/code/detection/perception/temporal/temporal_estimator.py))                                         |
|   |   - Track Center Trajectory History Window (1.5s / 30 frames)                                              |
|   |   - Linear Regression Velocity (vx, vy in px/s) & Speed Calculation                                        |
|   |   - Vector-to-Zone-Centroid Motion Angle (Approach toward vs Departure from robot)                         |
|   |   - Continuous Stability Accumulator (Stationary classification < 20 px/s)                                 |
|   |   - Output: TemporalObservation(approach, stability, movement, motion)                                     |
|   +------------------------------------------------------------------------------------------------------------+   |
|                                               |                                                                    |
|                                               | Track + FrameObservation + TemporalObservation                     |
|                                               v                                                                    |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | MILESTONE M6: Distance Provider ([distance_provider.py](file:///home/eshwar/code/detection/perception/contact/distance_provider.py))                                                        |
|   |   - Mode: MOCK (calibrated test distance) or METRIC (depth sensor / stereo)                               |
|   |   - Timestamp & Track ID Target Association Validation                                                     |
|   |   - Output: DistanceObservation(distance_m, status, timestamp, track_id)                                  |
|   +------------------------------------------------------------------------------------------------------------+   |
|                                               |                                                                    |
|                                               v                                                                    |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | MILESTONE M6: Readiness Engine & Decision Layer ([readiness_engine.py](file:///home/eshwar/code/detection/perception/contact/readiness_engine.py))                                        |
|   |   +----------------------------------------------------------------------------------------------------+   |   |
|   |   | 1. Boundary Monotonic Timestamp Validation (Rejects backward / duplicate / NaN timestamps)         |   |   |
|   |   +----------------------------------------------------------------------------------------------------+   |   |
|   |   | 2. Visit Manager ([visit_manager.py](file:///home/eshwar/code/detection/perception/contact/visit_manager.py)): Target Locking, 5.0s Grace Period, Zone Exit Lifecycle       |   |   |
|   |   +----------------------------------------------------------------------------------------------------+   |   |
|   |   | 3. Predicate Evaluator ([predicates.py](file:///home/eshwar/code/detection/perception/contact/predicates.py)): Evaluates 7 Predicates: P & B & A & Z & D & S & O         |   |   |
|   |   +----------------------------------------------------------------------------------------------------+   |   |
|   |   | 4. Criteria Selection Gate ([criteria_selection.py](file:///home/eshwar/code/detection/perception/contact/criteria_selection.py)): E_demo(t) = /\_{i in C} P_i(t)                      |   |   |
|   |   +----------------------------------------------------------------------------------------------------+   |   |
|   |   | 5. State Machine ([readiness_state_machine.py](file:///home/eshwar/code/detection/perception/contact/readiness_state_machine.py)): NOT_READY <-> PENDING (2.0s) <-> READY         |   |   |
|   |   +----------------------------------------------------------------------------------------------------+   |   |
|   |   | 6. One-Shot Event: FIRST_CONTACT_READY ({1,2,7} or all 7) vs DEMO_CRITERIA_READY (subsets)       |   |   |
|   |   +----------------------------------------------------------------------------------------------------+   |   |
|   +------------------------------------------------------------------------------------------------------------+   |
|               |                                                                 |                                  |
|               v                                                                 v                                  |
|   +---------------------------------------+                     +----------------------------------------------+   |
|   | Downstream Robot System / Action      |                     | Live HUD Display & OpenCV Visualizer         |   |
|   |   - Greeting gesture / Speech prompt  |                     |   - Bounding boxes (Green/Amber/Cyan)        |   |
|   |   - Navigation approach / Handover    |                     |   - Interaction Zone Boundary Polygon        |   |
|   |   - Event ingestion via JSON payload  |                     |   - 7-Predicate Live Checklist & Reason Codes|   |
|   +---------------------------------------+                     +----------------------------------------------+   |
|                                                                                                                    |
+--------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Per-Frame Pipeline Control Flow

This sequential flowchart illustrates the exact control decisions executed inside [`perception/contact/run_first_contact_demo.py`](file:///home/eshwar/code/detection/perception/contact/run_first_contact_demo.py) for every captured video frame:

```text
[Start: New Camera Frame Received]
                 |
                 v
   +---------------------------+
   | Capture Timestamp t_frame |
   +---------------------------+
                 |
                 v
   +---------------------------+
   | Validate Monotonicity     |
   | (t_frame > t_last_valid?) |
   +---------------------------+
         |               |
        YES              NO (Duplicate / Backward / NaN)
         |               |
         |               v
         |      +-------------------------------------------+
         |      | Fail-Closed Timestamp Boundary Handler    |
         |      | - Reset EligibilityTimer                  |
         |      | - State = NOT_READY                       |
         |      | - Reasons = [INVALID_TIMESTAMP]           |
         |      | - Emit ZERO events                        |
         |      +-------------------------------------------+
         |               |
         |               v
         |         [Skip to HUD Rendering]
         v
   +---------------------------+
   | Run YOLOv8n Detection     | ---> Filter Persons (class_id=0, conf >= 0.50)
   +---------------------------+
                 |
                 v
   +---------------------------+
   | Update SinglePersonTracker| ---> Kalman Predict -> IoU Match -> State Update
   +---------------------------+
                 |
                 v
   +---------------------------+
   | Observe Frame Readiness   | ---> Zone Polygon Test + Face Mesh + Head Pose
   +---------------------------+
                 |
                 v
   +---------------------------+
   | Estimate Temporal Motion  | ---> Rolling Window Velocity + Approach Angle
   +---------------------------+
                 |
                 v
   +---------------------------+
   | Acquire Distance Obs      | ---> Query Mock / Metric Provider
   +---------------------------+
                 |
                 v
   +-------------------------------------------------------+
   | VisitManager.update()                                 |
   | - Is target locked? (Lock on first confirmed track)   |
   | - Is active target observed in current frame?         |
   | - Tracking loss timer check (Grace period <= 5.0s)    |
   | - Reliable Zone Exit check (OUTSIDE + VALID)          |
   +-------------------------------------------------------+
                 |
                 v
   +-------------------------------------------------------+
   | PredicateEvaluator.evaluate()                         |
   | - P: Person presence valid                            |
   | - B: Bounding box matched to active track             |
   | - A: Confirmed approach or startup initial presence   |
   | - Z: Person inside interaction zone polygon           |
   | - D: Valid distance <= 1.50 metres                    |
   | - S: Stationary behaviour (< 20 px/s motion)          |
   | - O: Head orientation facing robot (|yaw| < 25 deg)   |
   +-------------------------------------------------------+
                 |
                 v
   +-------------------------------------------------------+
   | Criteria Gate Evaluation                              |
   | is_gate_valid = /\_{i in C} P_i(t)                    |
   | effective_valid = is_gate_valid AND target_observed   |
   +-------------------------------------------------------+
         |                                 |
       TRUE                              FALSE
         |                                 |
         v                                 v
   +-----------------------------+   +-----------------------------+
   | EligibilityTimer.update()   |   | EligibilityTimer.reset()    |
   | Accumulate valid time:      |   | Elapsed time reset to 0.0s  |
   | elapsed += dt               |   +-----------------------------+
   +-----------------------------+                 |
         |                                         v
         |                             +-----------------------+
         |                             | State = NOT_READY     |
         |                             | (Filter Reason Codes) |
         |                             +-----------------------+
         v                                         |
   +-----------------------------+                 |
   | Has Elapsed Reached 2.0s?   |                 |
   +-----------------------------+                 |
         |               |                         |
        YES              NO                        |
         |               |                         |
         v               v                         |
   +-----------+   +---------------+               |
   | State =   |   | State =       |               |
   | READY     |   | PENDING       |               |
   +-----------+   +---------------+               |
         |               |                         |
         v               |                         |
   +-----------------------------+                 |
   | One-Shot Event Check:       |                 |
   | Event already emitted?      |                 |
   +-----------------------------+                 |
         |               |                         |
        NO              YES                        |
         |               |                         |
         v               v                         |
   +-----------+   +---------------+               |
   | Emit      |   | Event = None  |               |
   | Event     |   | (Suppress     |               |
   | (One-Shot)|   |  Duplicates)  |               |
   +-----------+   +---------------+               |
         |               |                         |
         +-------+-------+-------------------------+
                 |
                 v
   +-------------------------------------------------------+
   | Render Diagnostic Overlays & HUD on Display Canvas    |
   +-------------------------------------------------------+
                 |
                 v
           [End of Frame]
```

---

## 3. Milestone M3: Single-Person Kalman Tracking Lifecycle

The SinglePersonTracker ([`single_person_tracker.py`](file:///home/eshwar/code/detection/perception/tracking/single_person_tracker.py)) maintains an 8-dimensional Kalman filter state vector $\mathbf{x} = [x, y, w, h, \dot{x}, \dot{y}, \dot{w}, \dot{h}]^T$ to provide smooth bounding boxes and velocity estimates across temporary occlusions:

```text
               +-------------------------------------------+
               |  Raw Detection from YOLOv8n (Person Det)  |
               +-------------------------------------------+
                                     |
                                     | New bounding box
                                     v
               +-------------------------------------------+
               |               [ TENTATIVE ]               |
               | - Track created                           |
               | - Hits = 1, Misses = 0                    |
               | - Not yet published as confirmed track    |
               +-------------------------------------------+
                      |                             |
     Consecutive Hits |                             | Miss / No detection
     Hit Count >= 3   |                             |
                      v                             v
+-----------------------------------+         +---------------------+
|           [ CONFIRMED ]           |         |     [ DELETED ]     |
| - Active tracked person           |         | - Track discarded   |
| - Kalman state updated            |         +---------------------+
| - Coordinates smoothed            |                    ^
| - Observed = True                 |                    |
+-----------------------------------+                    |
         |                  ^                            |
 Miss in |                  | Match in                   |
 frame   |                  | subsequent                 |
         v                  | frame                      |
+-----------------------------------+                    |
|         [ LOST / COASTING ]       |                    |
| - Kalman Predict only             |                    |
| - Observed = False                |                    |
| - Velocity damped (decay factor)  |                    |
| - Retention timer running (2.0s)  |                    |
+-----------------------------------+                    |
         |                                               |
         +--- Tracking Loss Duration >= 2.0s ------------+
```

---

## 4. Milestone M4: Frame-Level Stationary Readiness Observation

FrameObserver ([`frame_observer.py`](file:///home/eshwar/code/detection/perception/readiness/frame_observer.py)) performs instantaneous geometric and facial pose analysis on every frame:

```text
+----------------------------------------------------------------------------------------------------+
|                             M4 FRAME OBSERVATION PIPELINE                                          |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    Input: BGR Frame (1280x720) + Person Bounding Box [x1, y1, x2, y2]                              |
|           |                                                                                        |
|           +-------------------------------+--------------------------------+                      |
|           |                               |                                |                      |
|           v                               v                                v                      |
|   +--------------------+          +--------------------+          +--------------------+          |
|   | 1. Bbox Metrics    |          | 2. Zone Evaluator  |          | 3. Face Analyzer   |          |
|   | ([bbox_evaluator.py](file:///home/eshwar/code/detection/perception/readiness/bbox_evaluator.py))|          | ([interaction_zone.py](file:///home/eshwar/code/detection/perception/readiness/interaction_zone.py))|          | ([face_analyzer.py](file:///home/eshwar/code/detection/perception/readiness/face_analyzer.py))|          |
|   | - Width >= 50px    |          | - Polygon 4-corner |          | - Crop person box  |          |
|   | - Height >= 100px  |          |   vertices in px   |          | - MediaPipe Mesh   |          |
|   | - Aspect ratio     |          | - Bottom-center of |          |   478 3D landmarks |          |
|   |   in [0.20, 0.85]  |          |   person bbox      |          | - 6 key landmarks  |          |
|   | - Border clipping  |          | - Point-in-polygon |          |   (nose, chin, eyes|          |
|   |   margin >= 10px   |          |   test (Ray cast)  |          |   mouth corners)   |          |
|   +--------------------+          +--------------------+          +--------------------+          |
|           |                               |                                |                      |
|           | BboxMetrics                   | INSIDE / OUTSIDE               | 3D Face Landmarks    |
|           v                               v                                v                      |
|   +------------------------------------------------------------------------------------+          |
|   | Orientation Estimator ([orientation_estimator.py](file:///home/eshwar/code/detection/perception/readiness/orientation_estimator.py))                                           |          |
|   | - Solves Perspective-n-Point (EPnP / OpenCV solvePnP)                              |          |
|   | - Maps 2D landmarks to 3D anthropometric face model                                |          |
|   | - Computes Euler angles: Yaw, Pitch, Roll in degrees                               |          |
|   | - Decision Gate: |Yaw| <= 25.0° and |Pitch| <= 20.0° --> FACING_ROBOT              |          |
|   +------------------------------------------------------------------------------------+          |
|           |                                                                                        |
|           v                                                                                        |
|   +------------------------------------------------------------------------------------+          |
|   | Packaged FrameObservation:                                                         |          |
|   |   - PersonObservations: Presence=VALID, BoundingBox=VALID, Metrics                 |          |
|   |   - InteractionZone: INSIDE / OUTSIDE, Confidence, Status                          |          |
|   |   - OrientationObservations: FACING_ROBOT / NOT_FACING, Yaw, Pitch                |          |
|   +------------------------------------------------------------------------------------+          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 5. Milestone M5: Temporal Behaviour & Approach Estimation

TemporalEstimator ([`temporal_estimator.py`](file:///home/eshwar/code/detection/perception/temporal/temporal_estimator.py)) monitors movement over a rolling temporal buffer (1.5 seconds) to classify stability and approach vectors:

```text
[Track Center Coordinates (cx, cy) over 30 Frames (1.5s)]
                 |
                 +-----------------------------------+
                 |                                   |
                 v                                   v
   +---------------------------+       +-------------------------------+
   | Velocity Calculation      |       | Vector to Zone Centroid       |
   | - Ordinary Least Squares  |       | - Interaction Zone Centroid   |
   |   Linear Regression (OLS) |       |   Z_c = (cx_zone, cy_zone)    |
   | - vx = d(cx)/dt           |       | - Vector V_t = Z_c - P_t      |
   | - vy = d(cy)/dt           |       +-------------------------------+
   | - speed = sqrt(vx^2+vy^2) |                       |
   +---------------------------+                       |
                 |                                     |
                 v                                     v
   +-------------------------------------------------------+
   | Approach Angle Computation:                           |
   |   cos(theta) = (V_motion . V_t) / (|V_motion| * |V_t|)|
   +-------------------------------------------------------+
                 |
                 +-------------------+--------------------+
                 |                                        |
                 v                                        v
   +---------------------------+            +---------------------------+
   | Speed < 20 px/s?          |            | theta < 45.0 degrees?     |
   | AND Displacement < 30 px? |            | AND Distance Decreasing?  |
   +---------------------------+            +---------------------------+
         |               |                        |               |
        YES              NO                      YES              NO
         |               |                        |               |
         v               v                        v               v
   +-----------+   +-----------+            +-----------+   +-----------+
   | Movement= |   | Movement= |            | Approach= |   | Approach= |
   | STATIONARY|   | MOVING    |            |APPROACHING|   | DEPARTING |
   +-----------+   +-----------+            +-----------+   +-----------+
         |                                        |
         v                                        v
   +---------------------------+            +---------------------------+
   | Accumulate Stability Time |            | Approach Confirmed Flag   |
   | (Continuous time stable)  |            | (Persists for Visit)      |
   +---------------------------+            +---------------------------+
```

---

## 6. Milestone M6: Visit Lifecycle Management & Target Locking

The VisitManager ([`visit_manager.py`](file:///home/eshwar/code/detection/perception/contact/visit_manager.py)) manages the single-person target lock, handles tracking reacquisition within a 5.0-second grace period, and strictly terminates upon reliable zone exit:

```text
                                 [No Active Visit]
                                         |
                                         | First confirmed track appears
                                         | (with valid presence / approach)
                                         v
                         +-------------------------------+
                         |      ACTIVE VISIT CREATED     |
                         | - Generate UUID `visit_id`    |
                         | - Lock `active_track_id`      |
                         | - Event Emitted = False       |
                         | - Approach Established = False|
                         +-------------------------------+
                                         |
            +----------------------------+----------------------------+
            |                            |                            |
            v                            v                            v
   +------------------+         +------------------+         +------------------+
   | Active Target    |         | Target Lost      |         | Other Person     |
   | Track Observed   |         | (Occlusion /     |         | Appears / Moves  |
   | (ID matches lock)|         |  Detector Miss)  |         | (Mismatched ID)  |
   +------------------+         +------------------+         +------------------+
            |                            |                            |
            |                            v                            v
            |                   +------------------+         +------------------+
            |                   | 5.0s Grace Period|         | FAIL CLOSED      |
            |                   | Timer Running    |         | - Ignored for    |
            |                   | (Visit Preserved)|         |   visit lock     |
            |                   +------------------+         | - Cannot hijack  |
            |                            |                   | - Cannot reset   |
            |            +---------------+---------------+   +------------------+
            |            |                               |
            |            v Reacquired <= 5.0s            v Timeout >= 5.0s
            |   +------------------+            +------------------+
            +-->| TARGET RESTORED  |            | TERMINATE VISIT  |
                | - Resume visit   |            | Reason:          |
                +------------------+            | TRACKING_TIMEOUT |
                         |                      +------------------+
                         |                               ^
                         | Person moves OUTSIDE zone     |
                         | and Status == VALID           |
                         +-------------------------------+
                         | Reliable Zone Exit
```

---

## 7. Milestone M6: Seven Readiness Predicates & Decision Gate

The PredicateEvaluator ([`predicates.py`](file:///home/eshwar/code/detection/perception/contact/predicates.py)) computes all 7 criteria every frame, while the CriteriaSelection ([`criteria_selection.py`](file:///home/eshwar/code/detection/perception/contact/criteria_selection.py)) evaluates the dynamic conjunction gate:

```text
+--------------------------------------------------------------------------------------------------------+
|                                    7-PREDICATE EVALUATION MATRIX                                       |
+--------------------------------------------------------------------------------------------------------+
|                                                                                                        |
|  Frame Evidence                                     Predicate                      Reason Code         |
|  ------------------------------------------------   ---------------------------    ------------------- |
|  M1 YOLO Person Detected + Valid Status       --->  [P] Person Presence         -> NO_PERSON           |
|  M3 Kalman Track ID == Active Lock + Matched  --->  [B] Bounding Box Valid      -> INVALID_BBOX        |
|  M5 Approach Confirmed OR Startup Presence    --->  [A] Approach / Initial Pres -> APPROACH_UNCONF    |
|  M4 Interaction Zone == INSIDE + Valid Status  --->  [Z] Interaction Zone Inside -> OUTSIDE_ZONE       |
|  Distance Provider <= 1.50m + Valid Status    --->  [D] Distance Verified       -> TOO_FAR / NO_DIST   |
|  M5 Speed < 20 px/s + Stable Movement         --->  [S] Stationary Behaviour    -> PERSON_MOVING       |
|  M4 Head Pose |Yaw| <= 25.0° + Facing Robot   --->  [O] Facing Robot            -> NOT_FACING_ROBOT    |
|                                                                                                        |
+--------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+--------------------------------------------------------------------------------------------------------+
|                                  CONFIGURABLE CRITERIA SELECTION GATE                                  |
+--------------------------------------------------------------------------------------------------------+
|                                                                                                        |
|   CLI Selection Arguments:                                                                             |
|     * (Default)     C = {1, 2, 7}              --> P & B & O                                           |
|     * --all         C = {1, 2, 3, 4, 5, 6, 7}  --> P & B & A & Z & D & S & O                           |
|     * --stationary  C = {1, 2, 4, 7}           --> P & B & Z & O                                       |
|     * --temporal    C = {3, 6}                 --> A & S                                               |
|     * --criteria L  C = custom subset          --> e.g. {1, 2, 6} (P & B & S)                          |
|                                                                                                        |
|                                                     |                                                  |
|                                                     v                                                  |
|                   +------------------------------------------------------------------+                 |
|                   |   Decision Formula:                                              |                 |
|                   |     E_demo(t) = ( /\_{i in C} P_i(t) ) AND Target_Observed       |                 |
|                   +------------------------------------------------------------------+                 |
|                                     |                               |                                  |
|                                   TRUE                            FALSE                                |
|                                     |                               |                                  |
|                                     v                               v                                  |
|                         [Advance EligibilityTimer]        [Reset EligibilityTimer]                     |
|                                                           [Populate Selected Reasons]                  |
|                                                                                                        |
+--------------------------------------------------------------------------------------------------------+
```

---

## 8. Milestone M6: Readiness State Machine & Event Isolation

The ReadinessStateMachine ([`readiness_state_machine.py`](file:///home/eshwar/code/detection/perception/contact/readiness_state_machine.py)) regulates state progression and guarantees strict isolation between production and demo events:

```text
                          +-------------------------+
                          |      [ NOT_READY ]      | <-----------------------+
                          | Elapsed Eligibility: 0s |                         |
                          +-------------------------+                         |
                                       |                                      |
                                       | Gate Valid E_demo(t) == True         |
                                       v                                      |
                          +-------------------------+                         |
                          |       [ PENDING ]       |                         |
                          | Continuous timer running|                         |
                          | elapsed += dt           |                         |
                          +-------------------------+                         |
                             |                   |                            |
            E_demo == False  |                   | Elapsed >= 2.00s           |
            OR Gap > 0.5s    |                   | Continuous time            |
            OR Target Lost   |                   v                            |
                             |        +-------------------------+             |
                             |        |        [ READY ]        |             |
                             |        | Target fully eligible   |             |
                             |        +-------------------------+             |
                             |                     |                          |
                             |                     v                          |
                             |        +-------------------------+             |
                             |        | Check Criteria Mode:    |             |
                             |        +-------------------------+             |
                             |           |                   |                |
                             |  Default  |                   | Non-Default    |
                             |  {1,2,7}  |                   | Experimental   |
                             |  OR All 7 |                   | Subset         |
                             |           v                   v                |
                             |  +--------------------+  +--------------------+|
                             |  | Emit One-Shot Event|  | Emit One-Shot Event||
                             |  | FIRST_CONTACT_READY|  | DEMO_CRITERIA_READY||
                             |  | ([FirstContact...])|  | ([DemoCriteria...])||
                             |  +--------------------+  +--------------------+|
                             |           |                   |                |
                             |           +---------+---------+                |
                             |                     |                          |
                             |                     v                          |
                             |        +-------------------------+             |
                             |        | Mark Visit Event Emitted|             |
                             |        | (Suppress Duplicates)   |             |
                             |        +-------------------------+             |
                             |                     |                          |
                             |                     | Any Predicate Broken     |
                             |                     | OR Target Leaves Zone    |
                             +---------------------+--------------------------+
```

---

## 9. Global Monotonic Timestamp & Frame Consistency Validation

The boundary timestamp safety layer in [`readiness_engine.py`](file:///home/eshwar/code/detection/perception/contact/readiness_engine.py) provides fail-closed protection against non-monotonic clock skew and frame mismatches:

```text
[Incoming Captured Frame Timestamp: t_frame]
                      |
                      v
        +----------------------------+
        | Is t_frame None,           |
        | NaN, Infinite, or < 0.0?   |
        +----------------------------+
               |               |
              YES              NO
               |               |
               |               v
               | +----------------------------+
               | | t_last_accepted is set     |
               | | AND t_frame <= t_last?     |
               | +----------------------------+
               |        |               |
               |       YES              NO (Valid Progression)
               |        |               |
               v        v               v
    +------------------------+  +-------------------------------------+
    | FAIL CLOSED BOUNDARY   |  | t_last_accepted = t_frame           |
    | - Reset State Machine  |  +-------------------------------------+
    | - State = NOT_READY    |                  |
    | - Reasons =            |                  v
    |   [INVALID_TIMESTAMP]  |  +-------------------------------------+
    | - Zero Events Emitted  |  | Distance Observation Freshness:     |
    | - Retain t_last_valid  |  | |t_dist - t_frame| <= 0.10s?        |
    +------------------------+  +-------------------------------------+
               |                         |                 |
               v                        YES                NO
        [Return Early]                   |                 |
                                         v                 v
                                    [Accept Dist]    [Reject Dist: TIMESTAMP_MISMATCH]
```

---

## 10. Diagnostic HUD & Visual Feedback Layer

The visual renderer in [`run_first_contact_demo.py`](file:///home/eshwar/code/detection/perception/contact/run_first_contact_demo.py) transforms all pipeline metrics into an intuitive live heads-up display:

```text
+----------------------------------------------------------------------------------------------------+
| CAMERA WINDOW: "Vision-Based First Contact — M6 Single-Person Readiness Pipeline"                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  HUD PANEL (Top-Left):                                                                             |
|  =======================================================                                           |
|  === M6 FIRST-CONTACT READINESS PIPELINE ===            (Cyan Header)                              |
|  FPS: 30.0 | Latency: 12.1ms Y, 0.4ms T, 8.2ms M4...    (White Telemetry)                          |
|  MODE: DEMO CRITERIA GATE (Selected: {1,2,7} [P, B, O]) (Amber Mode Banner)                        |
|  STATE: [ PENDING ] Elapsed: 1.40s/2.00s | Visit: cd47  (Amber State Progress)                     |
|  -- Criteria: [SEL]=Active Gate, [---]=Passive --       (Checklist Header)                         |
|   [+] [SEL] 1. P - Person presence (detected)           (Bright Green - Satisfied Selected)        |
|   [+] [SEL] 2. B - BBox valid (matched)                 (Bright Green - Satisfied Selected)        |
|   [-] [---] 3. A - Approach/init (unconfirmed) (passive)(Muted Gray - Passive Unselected)          |
|   [+] [---] 4. Z - Interaction zone (inside) (passive)  (Muted Gray - Passive Unselected)          |
|   [+] [---] 5. D - Distance <=1.5m (1.2m) (passive)     (Muted Gray - Passive Unselected)          |
|   [-] [---] 6. S - Stationary (moving) (passive)        (Muted Gray - Passive Unselected)          |
|   [+] [SEL] 7. O - Facing robot (facing)                (Bright Green - Satisfied Selected)        |
|  =======================================================                                           |
|                                                                                                    |
|                                    +-----------------------+                                       |
|                                    | [!] MOCK DIST (1.2m)  | (Amber Badge, Top-Right)              |
|                                    +-----------------------+                                       |
|                                                                                                    |
|                 +-------------------------------------------------+                                |
|                 | Interaction Zone Polygon (Cyan Boundary Lines)  |                                |
|                 |                                                 |                                |
|                 |                  +---------+                    |                                |
|                 |                  | ID: 1   | (Bounding Box:     |                                |
|                 |                  | PENDING |  Amber if Pending, |                                |
|                 |                  |         |  Green if Ready,   |                                |
|                 |                  |         |  Magenta if Coast) |                                |
|                 |                  +---------+                    |                                |
|                 |                       x (Centroid Marker)       |                                |
|                 |                                                 |                                |
|                 +-------------------------------------------------+                                |
|                                                                                                    |
|  EVENT NOTIFICATION BANNER (Bottom-Center):                                                        |
|  *** FIRST CONTACT READY: FIRST_CONTACT_READY (id=evt_1b38558c, criteria={1,2,7}) ***             |
+----------------------------------------------------------------------------------------------------+
```

---

## 11. Cross-Reference of Key Codebase Files

| Layer | Implementation File | Key Classes & Functions |
|---|---|---|
| **M1 Detector** | [`perception/detection/person_detector.py`](file:///home/eshwar/code/detection/perception/detection/person_detector.py) | `PersonDetector`, `PersonDetection` |
| **M3 Tracker** | [`perception/tracking/single_person_tracker.py`](file:///home/eshwar/code/detection/perception/tracking/single_person_tracker.py) | `SinglePersonTracker`, `Track`, `TrackState` |
| **M4 Observer** | [`perception/readiness/frame_observer.py`](file:///home/eshwar/code/detection/perception/readiness/frame_observer.py) | `FrameObserver`, `InteractionZoneEvaluator`, `OrientationEstimator` |
| **M5 Temporal** | [`perception/temporal/temporal_estimator.py`](file:///home/eshwar/code/detection/perception/temporal/temporal_estimator.py) | `TemporalEstimator`, `TemporalObservation` |
| **M6 Distance** | [`perception/contact/distance_provider.py`](file:///home/eshwar/code/detection/perception/contact/distance_provider.py) | `DistanceProvider`, `MockDistanceProvider`, `create_distance_provider` |
| **M6 Visit** | [`perception/contact/visit_manager.py`](file:///home/eshwar/code/detection/perception/contact/visit_manager.py) | `VisitManager`, target locking, 5.0s grace period |
| **M6 Predicates** | [`perception/contact/predicates.py`](file:///home/eshwar/code/detection/perception/contact/predicates.py) | `PredicateEvaluator`, 7 readiness predicates |
| **M6 Selection** | [`perception/contact/criteria_selection.py`](file:///home/eshwar/code/detection/perception/contact/criteria_selection.py) | `CriteriaSelection`, stable IDs, CLI parsing, reason filtering |
| **M6 State Machine**| [`perception/contact/readiness_state_machine.py`](file:///home/eshwar/code/detection/perception/contact/readiness_state_machine.py) | `ReadinessStateMachine`, `EligibilityTimer`, event isolation |
| **M6 Engine** | [`perception/contact/readiness_engine.py`](file:///home/eshwar/code/detection/perception/contact/readiness_engine.py) | `ReadinessEngine`, boundary timestamp validator |
| **M6 Runner** | [`perception/contact/run_first_contact_demo.py`](file:///home/eshwar/code/detection/perception/contact/run_first_contact_demo.py) | Full demonstration runner & interactive HUD overlays |

