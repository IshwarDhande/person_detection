# Simple Architecture & Flow Diagrams

> **Project:** Vision-Based Single-Person First Contact Readiness Pipeline (M1–M6)  
> **Summary:** A simplified, high-level overview of the system architecture and decision flow.

---

## 1. Simple System Architecture Diagram

```text
===================================================================================
                       SIMPLE END-TO-END PIPELINE ARCHITECTURE
===================================================================================

    [ Camera Video Stream ] (/dev/video0)
               |
               v
  +-------------------------+
  | M1: Person Detector     |  Detects human bounding boxes via YOLOv8n
  +-------------------------+
               |
               v
  +-------------------------+
  | M3: Single-Person Track |  Tracks the target with an 8-D Kalman Filter & IoU
  +-------------------------+
               |
               +-----------------------------------+
               |                                   |
               v                                   v
  +-------------------------+         +-------------------------+
  | M4: Frame Readiness     |         | M5: Temporal Behaviour  |
  | - Interaction Zone PIP  |         | - Track Center Velocity |
  | - 3D Head Pose (Facing) |         | - Approach toward Robot |
  +-------------------------+         | - Stationary Stability  |
               |                      +-------------------------+
               |                                   |
               +-----------------+-----------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  | M6: Distance Provider                                       |
  | - Estimated / Mocked distance to target (<= 1.5 metres)     |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  | M6: Readiness Engine & Decision Gate                        |
  | - Locks onto single person (Visit Manager)                  |
  | - Checks 7 Criteria: Person, BBox, Approach, Zone,          |
  |                      Distance, Stationary, Orientation      |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  | M6: Eligibility Timer (2.0 Seconds Continuous)              |
  | - Person must satisfy conditions continuously for 2.0s      |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  | OUTPUT: First Contact Event & Interactive Display           |
  | - Production: FIRST_CONTACT_READY ({1, 2, 7} or All 7)      |
  | - Demo Gate : DEMO_CRITERIA_READY (Experimental Subsets)    |
  +-------------------------------------------------------------+
===================================================================================
```

---

## 2. Simple Decision Flow Diagram

```text
===================================================================================
                         SIMPLE FRAME DECISION FLOW
===================================================================================

                  [ New Video Frame Arrives ]
                               |
                               v
                  /-------------------------\
                 <  Is a Person Detected?    > --- NO ---> [ State = NOT_READY ]
                  \-------------------------/
                               |
                              YES
                               v
                  /-------------------------\
                 <   Is Active Target Locked > --- NO ---> [ Lock Track ID & Start Visit ]
                  \-------------------------/
                               |
                              YES
                               v
                  /-------------------------\
                 <  Inside Interaction Zone? > --- NO ---> [ Reset Timer -> NOT_READY ]
                  \-------------------------/
                               |
                              YES
                               v
                  /-------------------------\
                 <  Facing Toward Robot?     > --- NO ---> [ Reset Timer -> NOT_READY ]
                  \-------------------------/
                               |
                              YES
                               v
                  /-------------------------\
                 <  Stationary & Close Dist? > --- NO ---> [ Reset Timer -> NOT_READY ]
                  \-------------------------/
                               |
                              YES (All Criteria Satisfied)
                               v
                  +-------------------------+
                  | State = PENDING         |
                  | Accumulate Valid Time   |
                  +-------------------------+
                               |
                               v
                  /-------------------------\
                 <   Has 2.0 Seconds Elapsed > --- NO ---> [ Keep PENDING, Wait Next Frame ]
                  \-------------------------/
                               |
                              YES
                               v
                  +-------------------------+
                  | State = READY           |
                  | Emit One-Shot Event!    |
                  +-------------------------+
===================================================================================
```

---

## 3. Simple State Machine Transition Diagram

```text
===================================================================================
                          SIMPLE READINESS STATE MACHINE
===================================================================================

                     +-------------------+
                     |     NOT_READY     | <-------------------------+
                     +-------------------+                           |
                               |                                     |
                               | Conditions satisfied                |
                               v                                     | Condition breaks
                     +-------------------+                           | (e.g. person turns,
                     |      PENDING      | --------------------------+  steps outside,
                     | (Timer 0.0s->2.0s)|                              moves quickly)
                     +-------------------+                           |
                               |                                     |
                               | 2.0s continuous                     |
                               v                                     |
                     +-------------------+                           |
                     |       READY       | --------------------------+
                     |  (Event Emitted)  |
                     +-------------------+
===================================================================================
```

---

## 4. Pipeline Stages & Key Modules

| Milestone | Purpose | Key File Link |
|---|---|---|
| **M1** | Detects human bodies in the video frame | [`person_detector.py`](file:///home/eshwar/code/detection/perception/detection/person_detector.py) |
| **M3** | Tracks a single target over time with Kalman filtering | [`single_person_tracker.py`](file:///home/eshwar/code/detection/perception/tracking/single_person_tracker.py) |
| **M4** | Checks if the person is inside the zone and facing the robot | [`frame_observer.py`](file:///home/eshwar/code/detection/perception/readiness/frame_observer.py) |
| **M5** | Evaluates if the person approached and is standing still | [`temporal_estimator.py`](file:///home/eshwar/code/detection/perception/temporal/temporal_estimator.py) |
| **M6** | Orchestrates visit lifecycle, 7 criteria gate, and 2.0s timer | [`readiness_engine.py`](file:///home/eshwar/code/detection/perception/contact/readiness_engine.py) |
| **Demo** | Live interactive camera application with visual HUD | [`run_first_contact_demo.py`](file:///home/eshwar/code/detection/perception/contact/run_first_contact_demo.py) |

