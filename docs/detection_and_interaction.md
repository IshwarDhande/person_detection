# Detection and Interaction Module Integration Architecture

> **Purpose:** Detailed architectural guide and flow diagrams explaining how the **Perception/Detection Pipeline** (M1–M6) and the **Interaction Module** (`interaction/`) interface, exchange data, and coordinate robotic behavior during human-robot encounters.

---

## Table of Contents

1. [Architectural Philosophy: Perception vs. Interaction](#1-architectural-philosophy-perception-vs-interaction)
2. [End-to-End System Integration Diagram](#2-end-to-end-system-integration-diagram)
3. [Lifecycle Sequence Diagram (Encounter to Farewell)](#3-lifecycle-sequence-diagram-encounter-to-farewell)
4. [Data Contracts & Boundary Payloads](#4-data-contracts--boundary-payloads)
5. [State Mapping: Readiness vs. Interaction](#5-state-mapping-readiness-vs-interaction)
6. [Supported Integration Patterns (API & CLI)](#6-supported-integration-patterns-api--cli)
7. [Key Safety Invariants Across the Boundary](#7-key-safety-invariants-across-the-boundary)
8. [Codebase Cross-References](#8-codebase-cross-references)

---

## 1. Architectural Philosophy: Perception vs. Interaction

In autonomous service robotics, coupling raw computer vision directly to physical actuators (speech, screens, motors) leads to jittery behavior, false triggers, and duplicate speech. This project enforces a clean architectural separation between:

```text
+---------------------------------------------------+      +---------------------------------------------------+
|               PERCEPTION SUBSYSTEM                |      |               INTERACTION SUBSYSTEM               |
|      ([perception/](file:///home/eshwar/code/detection/perception/))                     |      |      ([interaction/](file:///home/eshwar/code/detection/interaction/))                   |
+---------------------------------------------------+      +---------------------------------------------------+
| - Passive, continuous sensor processing           |      | - Active, event-driven decision making            |
| - High-frequency frame loop (30 FPS)              |      | - State-driven robot dialogue & behavior          |
| - Probabilistic detections, Kalman filters        | ---> | - Deterministic actions: SPEAK, DISPLAY, GESTURE  |
| - 7-predicate readiness validation                |      | - User engagement lifecycle: IDLE to FAREWELL     |
| - Enforces 2.0s continuous eligibility timer      |      | - Prevents repeated or out-of-order greetings     |
| - Emits structured one-shot events & frame data   |      | - Dispatches to hardware actuators / screen UI    |
+---------------------------------------------------+      +---------------------------------------------------+
```

- **Perception asks:** *"Is there a specific single human in front of the robot who has intentionally approached, entered the interaction zone, stopped, and is looking at the robot for 2 continuous seconds?"*
- **Interaction asks:** *"Given that first-contact readiness is confirmed, how should the robot welcome this person, what should it say, which UI card should it show, and how should it say goodbye when they leave?"*

---

## 2. End-to-End System Integration Diagram

The following diagram illustrates the complete data flow from the camera lens to downstream robotic actuation:

```text
+---------------------------------------------------------------------------------------------------------------------+
|                             PERCEPTION-TO-INTERACTION INTEGRATION PIPELINE                                          |
+---------------------------------------------------------------------------------------------------------------------+
|                                                                                                                     |
|   CAMERA STREAM                                                                                                     |
|   [/dev/video0 @ 30 FPS]                                                                                            |
|           |                                                                                                         |
|           v                                                                                                         |
|   +-------------------------------------------------------------------------------------------------------------+   |
|   | PERCEPTION SUBSYSTEM (M1–M6)                                                                                |   |
|   |                                                                                                             |   |
|   |  Stage 1: YOLOv8n Person Detector (M1) --------------------------> Detects 2D human bounding boxes         |   |
|   |  Stage 2: Single-Person Kalman Tracker (M3) ---------------------> Tracks target ID & spatial consistency  |   |
|   |  Stage 3: Frame Readiness Observer (M4) -------------------------> Face orientation (EPnP) & zone test     |   |
|   |  Stage 4: Temporal Motion Estimator (M5) ------------------------> Speed < 20 px/s & confirmed approach     |   |
|   |  Stage 5: Distance Provider (M6) --------------------------------> Measures proximity (<= 1.5 metres)       |   |
|   |  Stage 6: Visit Manager (M6) ------------------------------------> Locks target ID & 5.0s grace period      |   |
|   |  Stage 7: Readiness State Machine & Timer (M6) ------------------> Enforces continuous 2.00s eligibility    |   |
|   +-------------------------------------------------------------------------------------------------------------+   |
|           |                                                                                                         |
|           | BOUNDARY INTERFACE: Emits FirstContactResult / FirstContactReadyEvent / DemoCriteriaReadyEvent          |
|           v                                                                                                         |
|   +-------------------------------------------------------------------------------------------------------------+   |
|   | INTERACTION SUBSYSTEM ([interaction/mock_interaction_receiver.py](file:///home/eshwar/code/detection/interaction/mock_interaction_receiver.py))                                              |   |
|   |                                                                                                             |   |
|   |   +-----------------------------------------------------------------------------------------------------+   |   |
|   |   | 1. Ingestion Adapter (Direct Method Call / JSON Line Parser / Pipe Consumer)                        |   |   |
|   |   +-----------------------------------------------------------------------------------------------------+   |   |
|   |                                                      |                                                          |
|   |                                                      v                                                          |
|   |   +-----------------------------------------------------------------------------------------------------+   |   |
|   |   | 2. Interaction State Machine                                                                        |   |   |
|   |   |    [ IDLE ]  --->  [ ATTENDING ]  --->  [ ENGAGED ]  --->  [ INTERACTING ]  --->  [ FAREWELL ]      |   |   |
|   |   +-----------------------------------------------------------------------------------------------------+   |   |
|   |                                                      |                                                          |
|   |                                                      v                                                          |
|   |   +-----------------------------------------------------------------------------------------------------+   |   |
|   |   | 3. Action Dispatcher (Generates Structured InteractionAction Objects)                               |   |   |
|   |   +-----------------------------------------------------------------------------------------------------+   |   |
|   +-------------------------------------------------------------------------------------------------------------+   |
|           |                               |                               |                               |         |
|           v                               v                               v                               v         |
|   +---------------+               +---------------+               +---------------+               +---------------+ |
|   | SPEAK ACTION  |               | DISPLAY ACTION|               | GESTURE ACTION|               | LED ACTION    | |
|   | TTS Greeting: |               | Screen UI:    |               | Physical:     |               | Hardware Ring:| |
|   | "Hello! I am  |               | Welcome Card, |               | Wave and Nod, |               | Green (Ready),| |
|   |  ready to     |               | Target ID,    |               | Attentive Nod,|               | Amber (Pend), | |
|   |  assist you." |               | Proximity     |               | Orient Head   |               | Off (Standby) | |
|   +---------------+               +---------------+               +---------------+               +---------------+ |
|                                                                                                                     |
+---------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Lifecycle Sequence Diagram (Encounter to Farewell)

This sequence diagram illustrates the chronological step-by-step handshake between a human user, the perception engine, and the mock interaction receiver:

```text
Human User              Perception Engine (M1-M6)             Interaction Receiver (Mock)
    |                               |                                     |
    |------- Approaches robot ----->|                                     |
    |                               | [M1/M3] Target detected & locked    |
    |                               | [M6] Visit initialized (visit_123)  |
    |                               |                                     |
    |------- Enters Zone & Faces -->|                                     |
    |                               | [M4/M5] Zone=INSIDE, Facing=TRUE    |
    |                               | [M6] State: NOT_READY -> PENDING   |
    |                               |                                     |
    |                               |-- FirstContactResult(PENDING) ----->|
    |                               |                                     | [State: ATTENDING]
    |                               |                                     | Actions:
    |                               |                                     |   - LED: AMBER (pulse)
    |                               |                                     |   - GESTURE: HEAD_ORIENT
    |                               |                                     |
    |<------ Robot looks up & pulses amber LED ---------------------------|
    |                               |                                     |
    |------- Stands still 2.0s ---->|                                     |
    |                               | [M6] EligibilityTimer = 2.00s       |
    |                               | [M6] State: PENDING -> READY        |
    |                               | [M6] Emits FIRST_CONTACT_READY      |
    |                               |                                     |
    |                               |-- FirstContactReadyEvent ---------->|
    |                               |   (visit_123, track_1, 1.2m)        | [State: ENGAGED]
    |                               |                                     | Actions:
    |                               |                                     |   - LED: GREEN (glow ring)
    |                               |                                     |   - GESTURE: WAVE_AND_NOD
    |                               |                                     |   - SPEAK: "Hello! Welcome..."
    |                               |                                     |   - DISPLAY: Welcome Card
    |                               |                                     |
    |<------ Robot waves, glows green & speaks greeting ------------------|
    |                               |                                     |
    |------- User talks / interacts |                                     |
    |        (Active Session)       | [M6] Predicates satisfied           | [State: INTERACTING]
    |                               | [M6] Duplicate events SUPPRESSED    | (No duplicate speech)
    |                               |                                     |
    |------- User steps away ------>|                                     |
    |                               | [M4] Interaction Zone = OUTSIDE     |
    |                               | [M6] VisitManager terminates visit  |
    |                               |                                     |
    |                               |-- handle_visit_terminated() ------->|
    |                               |   (visit_123, reason=ZONE_EXIT)     | [State: FAREWELL -> IDLE]
    |                               |                                     | Actions:
    |                               |                                     |   - SPEAK: "Goodbye!"
    |                               |                                     |   - DISPLAY: Standby Screen
    |                               |                                     |   - LED: OFF
    |                               |                                     |
    |<------ Robot bids farewell & enters standby ------------------------|
    |                               |                                     |
```

---

## 4. Data Contracts & Boundary Payloads

The boundary between perception and interaction passes strongly-typed data structures defined in [`perception/contact/contact_types.py`](file:///home/eshwar/code/detection/perception/contact/contact_types.py):

### A. High-Frequency Frame Result (`FirstContactResult`)
Passed on every video frame to update continuous telemetry:
```python
FirstContactResult(
    state=ReadinessState.PENDING,          # NOT_READY | PENDING | READY
    event=None,                            # Populated ONLY on transition to READY
    predicates=ReadinessPredicates(...),   # P, B, A, Z, D, S, O Boolean flags
    reasons=[],                            # Explainable reason codes if unsatisfied
    visit_id="visit_cd472eab",             # Unique visit UUID
    track_id=17,                           # Active Kalman track identifier
    elapsed_eligibility=1.40,              # Continuous valid duration (seconds)
    distance=DistanceObservation(...),     # Proximity measurement
    selected_criteria=(1, 2, 7),           # Active criteria gate numbers
)
```

### B. Discrete Readiness Events
Emitted **exactly once** per visit when readiness criteria hold continuously for 2.00s:

1. **Production Event (`FIRST_CONTACT_READY`)** — Triggered when default production criteria `{1, 2, 7}` (`P, B, O`) or all 7 criteria `{1, 2, 3, 4, 5, 6, 7}` are selected:
```json
{
  "event_type": "FIRST_CONTACT_READY",
  "event_id": "evt_4a71b29d",
  "visit_id": "visit_cd472eab",
  "track_id": 17,
  "timestamp": 6469.757,
  "readiness_duration": 2.0,
  "distance": {
    "value_m": 1.20,
    "source": "MOCK"
  }
}
```

2. **Experimental Demo Event (`DEMO_CRITERIA_READY`)** — Triggered when a non-default experimental subset (e.g. `--temporal` `{3, 6}`, `--stationary` `{1, 2, 4, 7}`, or `--criteria 1,6`) is selected:
```json
{
  "event_type": "DEMO_CRITERIA_READY",
  "event_id": "demo_evt_1b38558c",
  "visit_id": "visit_cd472eab",
  "track_id": 17,
  "timestamp": 6469.757,
  "readiness_duration": 2.0,
  "distance": {
    "value_m": 1.20,
    "source": "MOCK"
  },
  "selected_criteria": [3, 6]
}
```

### C. Downstream Interaction Action (`InteractionAction`)
Dispatched by [`MockInteractionReceiver`](file:///home/eshwar/code/detection/interaction/mock_interaction_receiver.py) for robot actuation:
```python
InteractionAction(
    action_type=RobotActionType.SPEAK,
    payload={
        "text": "Hello! I am Assistant Robot. I see you are standing in front of me at 1.2 metres. How can I assist you today?",
        "voice": "friendly_assistant",
        "volume": 0.85
    },
    timestamp=1795154000.12,
    visit_id="visit_cd472eab",
    track_id=17,
    source_event="FIRST_CONTACT_READY"
)
```

---

## 5. State Mapping: Readiness vs. Interaction

The two subsystems maintain distinct but synchronized finite state machines:

| Perception Readiness State | Interaction Lifecycle State | Physical Robot Behavioral Response |
|:---|:---|:---|
| **`NOT_READY`** | **`IDLE`** | Standby mode. Screen displays idle attract loop; LEDs are turned off; robot body is relaxed. |
| **`PENDING`** | **`ATTENDING`** | Target is approaching or orienting. Robot orients head slightly toward target track; LED ring pulses **Amber** at 60% intensity. |
| **`READY`** (Initial Frame) | **`ENGAGED`** | First contact event received. Robot waves, nods, lights LED **Green** (or **Cyan** in demo mode), plays welcoming speech prompt, and renders Welcome Card on screen. |
| **`READY`** (Subsequent Frames) | **`INTERACTING`** | Active dialogue in progress. Duplicate greetings suppressed; robot listens for user input / speech recognition. |
| **`NOT_READY`** (Reverted during Pending) | **`IDLE`** | Person turned away before reaching 2.0s. Robot LED returns to OFF; head resets to neutral pose. |
| **`TERMINATED`** (Zone Exit / 5s Loss) | **`FAREWELL` $\rightarrow$ `IDLE`** | Person walked away. Robot says *"Goodbye! Have a wonderful day"*, clears user screen card, and enters standby. |

---

## 6. Supported Integration Patterns (API & CLI)

The interaction receiver supports two operational modes:

### Pattern A: Direct Python API Integration (In-Process)
Ideal for combined robotics nodes (e.g. ROS 2 node, integrated Python service):
```python
from interaction.mock_interaction_receiver import MockInteractionReceiver
from perception.contact.readiness_engine import ReadinessEngine

# 1. Initialize both components
engine = ReadinessEngine()
receiver = MockInteractionReceiver(robot_name="RoboHost")

# 2. Main processing loop
for frame, timestamp in camera_stream:
    # Evaluate frame readiness
    result = engine.evaluate(track, frame_obs, temp_obs, matched_det, timestamp=timestamp)

    # Ingest result into interaction layer
    actions = receiver.handle_result(result)

    # Dispatch triggered actions to hardware
    for action in actions:
        execute_robot_action(action)
```

### Pattern B: Piped JSON Streaming (Process-Isolated)
Ideal for decoupled microservices, where perception and interaction run in separate processes:
```bash
# Terminal 1: Pipe live demo runner JSON output directly to interaction receiver
PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 perception/contact/run_first_contact_demo.py --headless \
  | PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m interaction.mock_interaction_receiver --listen
```

### Pattern C: Standalone Simulation Walkthrough
Used for offline verification and regression testing:
```bash
PYTHONPATH=. /home/eshwar/code/interrupt/venv/bin/python3 -m interaction.mock_interaction_receiver --demo
```

---

## 7. Key Safety Invariants Across the Boundary

1. **Target Locking Immunity**:
   - The perception layer locks onto one active person (`active_track_id`).
   - If a bystander walks behind or past the active user, the bystander's detections are ignored.
   - The interaction module is guaranteed that actions are always targeted to the specific locked `visit_id`.

2. **One-Shot Greeting Guarantee**:
   - `VisitManager.mark_event_emitted()` ensures that only **one** `FIRST_CONTACT_READY` or `DEMO_CRITERIA_READY` event is emitted per visit.
   - The interaction module will never repeatedly greet a user who remains standing in front of the robot.

3. **5.0-Second Occlusion Tolerance**:
   - If the camera briefly drops tracking (e.g. someone walks between the user and robot for 1–2 seconds), the visit is preserved under the 5.0s grace period.
   - The interaction receiver does not immediately terminate or say goodbye; it waits until tracking is definitively lost (> 5.0s) or the user reliably exits the interaction zone.

4. **Fail-Closed Boundary Protection**:
   - If timestamp jitter, backward clock skew, or pipeline exceptions occur, the perception boundary resets eligibility to 0.0s and returns `INVALID_TIMESTAMP`.
   - The interaction module reverts to `IDLE` / standby rather than acting on corrupt perception data.

---

## 8. Codebase Cross-References

| Component | Source File Path | Description |
|---|---|---|
| **Readiness Engine** | [`perception/contact/readiness_engine.py`](file:///home/eshwar/code/detection/perception/contact/readiness_engine.py) | Milestone M6 core decision pipeline and boundary timestamp validator |
| **Visit Manager** | [`perception/contact/visit_manager.py`](file:///home/eshwar/code/detection/perception/contact/visit_manager.py) | Target locking, 5.0s grace period, and zone exit lifecycle manager |
| **Readiness State Machine** | [`perception/contact/readiness_state_machine.py`](file:///home/eshwar/code/detection/perception/contact/readiness_state_machine.py) | 2.0s eligibility accumulator and one-shot event emitter |
| **Event Definitions** | [`perception/contact/contact_types.py`](file:///home/eshwar/code/detection/perception/contact/contact_types.py) | Data models for `FirstContactResult`, `FirstContactReadyEvent`, `DemoCriteriaReadyEvent` |
| **Demo Runner** | [`perception/contact/run_first_contact_demo.py`](file:///home/eshwar/code/detection/perception/contact/run_first_contact_demo.py) | Interactive camera runner with visual HUD and criteria selection CLI |
| **Mock Interaction Receiver** | [`interaction/mock_interaction_receiver.py`](file:///home/eshwar/code/detection/interaction/mock_interaction_receiver.py) | Downstream mock consumer simulating robot speech, gestures, UI, and LEDs |
| **Interaction Tests** | [`tests/interaction/test_mock_interaction_receiver.py`](file:///home/eshwar/code/detection/tests/interaction/test_mock_interaction_receiver.py) | Dedicated unit tests verifying event ingestion, state transitions, and JSON handling |

