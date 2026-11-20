# Downstream Integration Guide: How to Use Detection Module Outputs

> **Target Audience:** Any roboticist, software engineer, or downstream process (Dialogue Systems, Speech Synthesis, Robot Gaze Controllers, Motion/Navigation Bases, Tablet Screen UIs, Hardware LED Controllers, or ROS 2 nodes) needing to consume outputs from the Vision-Based Detection and Readiness Subsystem.

---

## Table of Contents

1. [Architectural Overview & Data Flow](#1-architectural-overview--data-flow)
2. [The Two Core Outputs: Output 1 vs. Output 2](#2-the-two-core-outputs-output-1-vs-output-2)
3. [Key Python Source Files & Data Contracts](#3-key-python-source-files--data-contracts)
4. [Output 1: Discrete Encounter Events (Event-Driven)](#4-output-1-discrete-encounter-events-event-driven)
5. [Output 2: Continuous Frame Telemetry (30 Hz Stream)](#5-output-2-continuous-frame-telemetry-30-hz-stream)
6. [Downstream Integration Patterns](#6-downstream-integration-patterns)
   - [Pattern A: Direct Python In-Memory API (Same Process)](#pattern-a-direct-python-in-memory-api-same-process)
   - [Pattern B: Subprocess / Stdin-Stdout JSON Pipe (Separate Processes)](#pattern-b-subprocess--stdin-stdout-json-pipe-separate-processes)
   - [Pattern C: ROS 2 Topic Bridge (Robotics Architecture)](#pattern-c-ros-2-topic-bridge-robotics-architecture)
   - [Pattern D: WebSocket Bridge (Tablet Screen UI / Web Dashboard)](#pattern-d-websocket-bridge-tablet-screen-ui--web-dashboard)
7. [Downstream Module Responsibility Matrix](#7-downstream-module-responsibility-matrix)
8. [Safety Invariants & Best Practices for Consumers](#8-safety-invariants--best-practices-for-consumers)

---

## 1. Architectural Overview & Data Flow

The perception pipeline continuously analyzes camera frames (at 30 FPS). It handles person detection, single-person Kalman tracking, 3D head pose estimation, and temporal readiness filtering. 

Downstream modules do **not** need to process raw video frames. Instead, they consume structured objects:

```text
+-----------------------------------------------------------------------------------------------------------------+
|                                 DETECTION TO DOWNSTREAM INTERFACE ARCHITECTURE                                  |
+-----------------------------------------------------------------------------------------------------------------+
|                                                                                                                 |
|  CAMERA STREAM [/dev/video0 @ 30 FPS]                                                                           |
|         |                                                                                                       |
|         v                                                                                                       |
|  +-----------------------------------------------------------------------------------------------------------+  |
|  | PERCEPTION PIPELINE (M1–M6)                                                                               |  |
|  | - YOLOv8n Person Detector (M1)                                                                            |  |
|  | - Single-Person Kalman Tracker (M3)                                                                       |  |
|  | - Face Orientation & Interaction Zone Observer (M4)                                                       |  |
|  | - Temporal Motion Estimator (M5)                                                                          |  |
|  | - Readiness Engine & 2.0s Continuous Eligibility Gate (M6)                                                |  |
|  +-----------------------------------------------------------------------------------------------------------+  |
|         |                                                                     |                                 |
|         | (Every Frame @ 30 Hz)                                               | (Triggered Once per Visit)      |
|         v                                                                     v                                 |
|  +----------------------------------------------+           +------------------------------------------------+  |
|  | OUTPUT 2: CONTINUOUS FRAME TELEMETRY         |           | OUTPUT 1: DISCRETE ENCOUNTER EVENTS            |  |
|  | Class: FirstContactResult                    |           | Class: FirstContactReadyEvent                  |  |
|  | File : perception/contact/contact_types.py   |           | File : perception/contact/contact_types.py     |  |
|  +----------------------------------------------+           +------------------------------------------------+  |
|         |                                                                     |                                 |
|         +--------------------------------------+------------------------------+                                 |
|                                                |                                                                |
|                                                v                                                                |
|  +-----------------------------------------------------------------------------------------------------------+  |
|  | DOWNSTREAM CONSUMER ADAPTERS                                                                              |  |
|  | (Direct Python API / JSON Pipes / ROS 2 Publisher / WebSockets / IPC Sockets)                              |  |
|  +-----------------------------------------------------------------------------------------------------------+  |
|         |                             |                               |                              |          |
|         v                             v                               v                              v          |
|  +--------------------+      +--------------------+          +--------------------+         +-----------------+ |
|  | SPEECH & DIALOGUE  |      | GAZE & HEAD PAN    |          | NAVIGATION BASE    |         | TABLET UI & LED | |
|  | One-shot greeting  |      | Head centers on    |          | Stop / anchor base |         | Progress ring,  | |
|  | & personalized TTS |      | target centroid    |          | during encounter   |         | Welcome card    | |
|  +--------------------+      +--------------------+          +--------------------+         +-----------------+ |
|                                                                                                                 |
+-----------------------------------------------------------------------------------------------------------------+
```

---

## 2. The Two Core Outputs: Output 1 vs. Output 2

Any downstream system only needs to understand two fundamental output contracts:

| Output Channel | Frequency | Data Type / Class | Primary Purpose | Example Action Triggered |
|---|---|---|---|---|
| **Output 1: Discrete Events** | Event-Driven (Triggered **once** per visit) | [`FirstContactReadyEvent`](file:///home/eshwar/code/detection/perception/contact/contact_types.py#L115-L145) | State machine transitions, discrete dialogue initiation | Say *"Hello! How can I help you?"*, display welcome card. |
| **Output 2: Continuous Telemetry** | High-Frequency (Every frame, **30 Hz**) | [`FirstContactResult`](file:///home/eshwar/code/detection/perception/contact/contact_types.py#L165-L210) | Live tracking, pan-tilt head gaze, countdown displays | Pulse amber LED, draw countdown timer (0.0s $\rightarrow$ 2.0s), turn robot eyes toward human. |

---

## 3. Key Python Source Files & Data Contracts

These are the exact repository files involved in producing, defining, and consuming these outputs:

```text
+-----------------------------------------------------------------------------------------------------------------+
| REPOSITORY SOURCE FILE                             | ROLE              | KEY CLASSES / FUNCTIONS                |
+----------------------------------------------------+-------------------+----------------------------------------+
| perception/contact/contact_types.py                | Data Contracts    | FirstContactResult,                    |
|                                                    |                   | FirstContactReadyEvent,                |
|                                                    |                   | ReadinessState, ReasonCode             |
+----------------------------------------------------+-------------------+----------------------------------------+
| perception/contact/readiness_engine.py             | Producer Engine   | ReadinessEngine.evaluate()             |
|                                                    |                   | ReadinessEngine.reset()                |
+----------------------------------------------------+-------------------+----------------------------------------+
| perception/contact/run_first_contact_demo.py       | Executable Runner | run_first_contact_pipeline()           |
|                                                    |                   | (Main video capture & inference loop)  |
+----------------------------------------------------+-------------------+----------------------------------------+
| interaction/mock_interaction_receiver.py           | Consumer Example  | MockInteractionReceiver                |
|                                                    |                   | .handle_result(), .handle_event(),     |
|                                                    |                   | .handle_visit_terminated()             |
+----------------------------------------------------+-------------------+----------------------------------------+
```

---

## 4. Output 1: Discrete Encounter Events (Event-Driven)

### When Is It Emitted?
It is emitted **exactly once** when the person maintains the required production criteria (Criteria `{1, 2, 7}`: Person presence, Bounding box valid, Facing robot) continuously for **2.00 seconds**.

### Python Dataclass Definition
Defined in [`perception/contact/contact_types.py`](file:///home/eshwar/code/detection/perception/contact/contact_types.py#L115-L145):

```python
@dataclass(frozen=True)
class FirstContactReadyEvent:
    event_id: str                      # Unique UUID, e.g. "evt_4a71b29d"
    visit_id: str                      # Visit session UUID, e.g. "visit_cd472eab"
    track_id: int                      # Integer Kalman track ID, e.g. 1
    timestamp: float                   # Monotonic frame timestamp in seconds
    distance: DistanceInfo             # Proximity measurement object
    event_type: str = "FIRST_CONTACT_READY"
    readiness_duration: float = 2.0    # Required continuous eligibility (2.0s)
    utc_timestamp: Optional[str] = None
```

### Serialized JSON Payload
```json
{
  "event_type": "FIRST_CONTACT_READY",
  "event_id": "evt_4a71b29d",
  "visit_id": "visit_cd472eab",
  "track_id": 1,
  "timestamp": 6469.757,
  "readiness_duration": 2.0,
  "distance": {
    "value_m": 1.20,
    "source": "MOCK"
  }
}
```

### How Downstream Uses It:
1. **Dialogue / TTS:** Start speech greeting. Because the event fires **only once** per visit, your speech synthesizer will never repeatedly trigger or interrupt itself.
2. **Hardware LEDs:** Transition status ring from blinking/amber to solid glowing green.
3. **Screen UI:** Dismiss the standby screen and display the interactive greeting card.

---

## 5. Output 2: Continuous Frame Telemetry (30 Hz Stream)

### When Is It Emitted?
On **every camera frame** (30 times per second) as the return value of `ReadinessEngine.evaluate()`.

### Python Dataclass Definition
Defined in [`perception/contact/contact_types.py`](file:///home/eshwar/code/detection/perception/contact/contact_types.py#L165-L210):

```python
@dataclass(frozen=True)
class FirstContactResult:
    state: ReadinessState               # NOT_READY | PENDING | READY
    event: Optional[FirstContactReadyEvent] # Populated ONLY on transition frame to READY
    predicates: ReadinessPredicates     # Individual boolean status for P, B, A, Z, D, S, O
    reasons: list[ReasonCode]           # Empty when passing; failure codes when unsatisfied
    visit_id: Optional[str]             # Active visit identifier
    track_id: Optional[int]             # Active Kalman track identifier
    elapsed_eligibility: float          # Continuous valid duration (0.00s to 2.00s)
    distance: Optional[DistanceObservation]
    timings: ContactTimings             # Pipeline profiling latencies in milliseconds
    selected_criteria: tuple[int, ...]  # Active gate criteria numbers (default: (1, 2, 7))
```

### Key Fields for Downstream Processes:

1. **`result.state` (`ReadinessState`)**:
   - `NOT_READY`: Target is absent, turned away, or invalid. Robot should remain in idle/standby.
   - `PENDING`: Target is facing robot and valid. Eligibility timer is accumulating ($0.0s < t < 2.0s$). Robot should look attentive (amber light, orient head).
   - `READY`: Eligibility reached 2.0s. First contact confirmed.

2. **`result.elapsed_eligibility` (`float`)**:
   - A float between `0.0` and `2.0`.
   - Ideal for tablet UI circular progress bars (e.g. `progress = result.elapsed_eligibility / 2.0`).

3. **`result.event` (`Optional[FirstContactReadyEvent]`)**:
   - Contains **Output 1** if and only if the current frame is the moment of readiness completion!

4. **`result.reasons` (`list[ReasonCode]`)**:
   - Diagnostic reasons explaining why the person is not ready:
     `NOT_FACING_ROBOT`, `OUTSIDE_INTERACTION_ZONE`, `PERSON_MOVING`, `DISTANCE_TOO_FAR`.

---

## 6. Downstream Integration Patterns

### Pattern A: Direct Python In-Memory API (Same Process)

If your robot software is a unified Python application, instantiate [`ReadinessEngine`](file:///home/eshwar/code/detection/perception/contact/readiness_engine.py) directly:

```python
import time
from perception.contact.readiness_engine import ReadinessEngine
from perception.contact.contact_types import ReadinessState
from interaction.mock_interaction_receiver import MockInteractionReceiver

def run_robot_perception_loop():
    # 1. Initialize the Readiness Engine (automatically uses production criteria {1, 2, 7})
    engine = ReadinessEngine()
    
    # 2. Initialize your downstream consumer
    receiver = MockInteractionReceiver(robot_name="Robot")

    # In your camera processing loop (e.g., 30 FPS):
    while True:
        # (Acquire frame, run detector, tracker, observer...)
        t_now = time.monotonic()
        
        # Evaluate current frame
        result = engine.evaluate(
            track=current_track,
            frame_observation=current_frame_obs,
            timestamp=t_now
        )

        # -----------------------------------------------------------------
        # OUTPUT 2 CONSUMPTION (Continuous 30 Hz):
        # -----------------------------------------------------------------
        if result.state == ReadinessState.PENDING:
            # Update screen UI progress bar (0.0 to 1.0)
            ui_progress = result.elapsed_eligibility / 2.0
            print(f"Human is preparing... Progress: {ui_progress * 100:.0f}%")
            # Robot pan-tilt gaze or amber LED
            set_robot_led(color="AMBER", pulse=True)

        elif result.state == ReadinessState.NOT_READY:
            set_robot_led(color="OFF")

        # -----------------------------------------------------------------
        # OUTPUT 1 CONSUMPTION (Discrete Event Trigger):
        # -----------------------------------------------------------------
        if result.event is not None:
            # Reached 2.0s! result.event is FirstContactReadyEvent
            print(f"*** READINESS CONFIRMED: {result.event.event_id} ***")
            
            # Dispatch to interaction subsystem (TTS, screen card, gesture)
            actions = receiver.handle_event(result.event)
            for action in actions:
                execute_robot_action(action)
```

---

### Pattern B: Subprocess / Stdin-Stdout JSON Pipe (Separate Processes)

If your downstream system is in C++, Node.js, Python, or another service, run the perception runner in `--headless` mode and read JSON lines from standard output.

```text
+---------------------------------------------+               +---------------------------------------------+
| PROCESS 1: DETECTION PIPELINE               |               | PROCESS 2: INTERACTION SERVICE              |
| python3 run_first_contact_demo.py --headless| ---- PIPE --> | python3 mock_interaction_receiver.py --listen|
| (Emits JSON lines to stdout)                |               | (Reads JSON lines from stdin)               |
+---------------------------------------------+               +---------------------------------------------+
```

#### Running via Terminal Pipe:
```bash
PYTHONPATH=. python3 perception/contact/run_first_contact_demo.py --headless | \
PYTHONPATH=. python3 interaction/mock_interaction_receiver.py --listen
```

#### Consumer Implementation in Python (`consumer.py`):
```python
import sys
import json

def listen_to_detection_stream():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Check if incoming message is Output 1 (Discrete Event)
        if data.get("event_type") == "FIRST_CONTACT_READY":
            print(f"[EVENT] Greet person on visit {data['visit_id']} at distance {data['distance']['value_m']}m")
            trigger_tts_greeting("Hello! How can I help you today?")

        # Check if incoming message is Output 2 (Continuous Telemetry)
        elif "state" in data:
            state = data["state"]
            elapsed = data.get("elapsed_eligibility", 0.0)
            update_screen_ui(state=state, progress=elapsed / 2.0)

if __name__ == "__main__":
    listen_to_detection_stream()
```

---

### Pattern C: ROS 2 Topic Bridge (Robotics Architecture)

If running in a ROS 2 ecosystem (e.g. Humble/Iron/Jazzy), publish the two outputs on separate ROS 2 topics:

```text
+-------------------------------------------------------------------------------------------------------------+
| ROS 2 NODE: /perception_bridge_node                                                                         |
|                                                                                                             |
|   1. Topic: /perception/telemetry (30 Hz)                                                                   |
|      Message: perception_interfaces/msg/FirstContactTelemetry                                               |
|      Fields : string state, float32 elapsed_seconds, int32 track_id, string visit_id                        |
|                                                                                                             |
|   2. Topic: /perception/first_contact_ready (Triggered Once)                                                |
|      Message: perception_interfaces/msg/FirstContactReadyEvent                                              |
|      Fields : string event_id, string visit_id, int32 track_id, float32 distance_m                         |
+-------------------------------------------------------------------------------------------------------------+
```

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class PerceptionBridgeNode(Node):
    def __init__(self):
        super().__init__('perception_bridge')
        # Publisher for Output 2 (Continuous Telemetry @ 30 Hz)
        self.telemetry_pub = self.create_publisher(String, '/perception/telemetry', 10)
        # Publisher for Output 1 (Discrete One-Shot Event)
        self.event_pub = self.create_publisher(String, '/perception/first_contact_ready', 10)

    def publish_frame_result(self, result):
        # Publish Output 2
        telem_msg = String()
        telem_msg.data = json.dumps({
            "state": result.state.value,
            "elapsed": result.elapsed_eligibility,
            "track_id": result.track_id,
            "visit_id": result.visit_id
        })
        self.telemetry_pub.publish(telem_msg)

        # Publish Output 1 if event triggered
        if result.event is not None:
            evt_msg = String()
            evt_msg.data = json.dumps(result.event.to_dict())
            self.event_pub.publish(evt_msg)
            self.get_logger().info(f"Published FIRST_CONTACT_READY: {result.event.event_id}")
```

---

### Pattern D: WebSocket Bridge (Tablet Screen UI / Web Dashboard)

For modern service robots featuring an interactive chest or head tablet (React/Vue/HTML5), broadcast Output 2 over a local WebSocket server:

```text
+---------------------------------------+             WebSocket JSON             +----------------------------+
| Python WebSocket Server (port 8765)   | -------------------------------------> | Robot Chest Tablet UI      |
| Broadcasts Output 2 (30 Hz)           |        {state: "PENDING", ...}         | (React / HTML5 Canvas)     |
| Broadcasts Output 1 (Event)           |                                        | Displays Radial Timer & UI |
+---------------------------------------+                                        +----------------------------+
```

---

## 7. Downstream Module Responsibility Matrix

| Subsystem | Input Used | Purpose / Resulting Behavior |
|---|---|---|
| **Dialogue & Speech (TTS)** | **Output 1** (`FirstContactReadyEvent`) | Triggers welcome greeting once; sets voice volume based on `distance.value_m`. |
| **Gaze & Pan-Tilt Head** | **Output 2** (`track.bbox_xyxy` & `orient_val`) | Continuously steers robot head / pan-tilt servos to maintain eye contact with human. |
| **Mobility & Navigation** | **Output 2** (`state` & `predicates.Z`) | Pauses autonomous base navigation when a human enters `PENDING` to prevent collisions. |
| **Tablet Screen UI** | **Output 2** (`elapsed_eligibility`) + **Output 1** | Renders 0.0s–2.0s radial countdown circle; flips to interactive menu on `READY`. |
| **Hardware LED Ring** | **Output 2** (`state`) | **Standby**: OFF/Dim; **Pending**: Amber pulsing; **Ready**: Solid Green. |
| **Session Manager** | **Visit ID** (`visit_id`) | Groups all human dialogue history, user preferences, and orders under this session ID. |

---

## 8. Safety Invariants & Best Practices for Consumers

When writing any downstream module, follow these principles:

1. **Rely on the One-Shot Contract:**
   The perception engine guarantees that `result.event` will be populated on **exactly one frame** per visit. Downstream code does **not** need to implement custom debouncing or rate-limiting to prevent repeat greetings.

2. **Always Store `visit_id`:**
   Every human interaction belongs to a `visit_id`. If the person walks away and returns 10 seconds later, a **new** `visit_id` is generated, and a new greeting will legitimately trigger.

3. **Handle Visit Terminations:**
   When a human leaves the interaction zone or is lost for $> 5.0$ seconds, the visit terminates. Downstream systems should listen for visit resets to issue farewells (e.g. *"Goodbye!"*) and return the robot to idle standby.

