# Vision-Based First Contact Subsystem

A vision-only, single-person first-contact perception subsystem for autonomous service and social robotics. It continuously detects, tracks, and evaluates human encounter behaviour, confirming readiness and emitting one-shot structured events to downstream interaction and speech modules.

```text
+---------------------------------------------------------------------------------------------------------+
|                                    VISION-BASED FIRST CONTACT SYSTEM                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   CAMERA STREAM [/dev/video0 @ 30 FPS]                                                                  |
|         |                                                                                               |
|         v                                                                                               |
|   +--------------------------------------------------------------------------------------------------+  |
|   | PERCEPTION SUBSYSTEM (M1–M6)                                                                     |  |
|   |  1. YOLOv8n Person Detector (M1)          --> Locates 2D human bounding boxes                    |  |
|   |  2. Single-Person Kalman Tracker (M3)      --> Locks target identity & trajectory                 |  |
|   |  3. Frame Readiness Observer (M4)         --> Head orientation (EPnP yaw/pitch) & zone test      |  |
|   |  4. Temporal Motion Estimator (M5)         --> Velocity (<20 px/s) & approach confirmation        |  |
|   |  5. Readiness Engine & Gate (M6)           --> Enforces 2.0s continuous alignment {1, 2, 7}       |  |
|   +--------------------------------------------------------------------------------------------------+  |
|         |                                                                |                              |
|         | Output 2 (Continuous 30 Hz Telemetry)                          | Output 1 (One-Shot Trigger)  |
|         v                                                                v                              |
|   +--------------------------------------------------------------------------------------------------+  |
|   | DOWNSTREAM INTERACTION SUBSYSTEM (interaction/)                                                  |  |
|   |  - State Machine: IDLE -> ATTENDING -> ENGAGED -> INTERACTING -> FAREWELL                        |  |
|   |  - Actions: Text-to-Speech Greeting, Welcome Card UI, Head Pan Gaze, Status LED (Green/Amber)    |  |
|   +--------------------------------------------------------------------------------------------------+  |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 1. System Prerequisites

Before installing, ensure the host machine has:
* **Operating System:** Linux (Ubuntu 20.04/22.04/24.04 recommended)
* **Python Version:** Python 3.10, 3.11, or 3.12
* **Hardware:** Webcam (e.g. `/dev/video0`) or video capture device
* **Core Tools:** `python3`, `python3-venv`, `python3-pip`

---

## 2. Installation on a New Machine

Follow these step-by-step commands to set up the environment and install dependencies:

```text
+---------------------------------------------------------------------------------------------------------+
|                                     ENVIRONMENT SETUP WORKFLOW                                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. Clone / Copy Repository   ---->   2. Create Virtualenv   ---->   3. Activate & Install             |
|      cd detection/                         python3 -m venv .venv          pip install -r requirements.txt
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Step 1: Navigate to Project Directory
```bash
cd detection
```

### Step 2: Create a Dedicated Virtual Environment
```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```bash
source .venv/bin/activate
```
*(Or use the portable helper script: `source activate_env.sh`)*

### Step 4: Upgrade Pip & Install Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verify & Download Model Weights

Because binary weights files (`*.pt`, `*.task`) are excluded from Git repositories via `.gitignore`, model files are handled depending on how you transferred the code:

* **If you copied the folder directly (`scp`, `rsync`, or USB drive):** The `models/` directory is already included.
* **If you cloned via Git (`git clone`):** Run the automated downloader script to fetch the two models in ~5 seconds:
  ```bash
  ./download_models.sh
  ```

The pipeline requires two pre-trained model files stored in [`models/`](file:///home/eshwar/code/detection/models/):

```text
models/
├── yolov8n.pt             # YOLOv8 nano person detection weights (6.3 MB)
└── face_landmarker.task   # MediaPipe face mesh & head pose model (3.6 MB)
```

Verify that both model files exist:
```bash
ls -lh models/
```

Test loading the detector model:
```bash
python3 -c "from ultralytics import YOLO; model = YOLO('models/yolov8n.pt'); print('YOLO model loaded:', model.names[0])"
```

---

## 4. Running the System

### Option A: Production First-Contact Runner (Interactive GUI)
Runs the complete perception pipeline with real-time video preview, Kalman track bounds, interaction zone, and live readiness HUD:
```bash
PYTHONPATH=. python3 perception/contact/run_first_contact_demo.py
```
> *Press `q` or `Esc` in the camera window to exit.*

### Option B: Headless Production Runner (Robot Base / Server Mode)
Runs headless without opening OpenCV graphical windows, logging frame metrics and events to the terminal:
```bash
PYTHONPATH=. python3 perception/contact/run_first_contact_demo.py --headless
```

### Option C: Simulated Interaction Receiver Walkthrough
Simulates an encounter from person approach to greeting dispatch and farewell:
```bash
PYTHONPATH=. python3 -m interaction.mock_interaction_receiver --demo
```

### Option D: Streaming Pipeline via Stdin-Stdout JSON Pipe
Runs detection in headless mode and pipes JSON events directly into the interaction receiver:
```bash
PYTHONPATH=. python3 perception/contact/run_first_contact_demo.py --headless | \
PYTHONPATH=. python3 -m interaction.mock_interaction_receiver --listen
```

---

## 5. Downstream Integration: Using Detection Outputs

For roboticists and software engineers integrating this perception system with downstream modules (Dialogue Systems, Head Gaze Servos, Navigation Bases, Tablet UIs, and ROS 2 stacks), read the full guide in:

👉 **[`docs/how_to_use_detection_outputs.md`](file:///home/eshwar/code/detection/docs/how_to_use_detection_outputs.md)**

```text
+---------------------------------------------------------------------------------------------------------+
|                                    DETECTION OUTPUTS AT A GLANCE                                        |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ PERCEPTION SUBSYSTEM (M1–M6) ]                                                                       |
|         |                                                                                               |
|         +--> OUTPUT 1: DISCRETE ENCOUNTER EVENTS (One-Shot Trigger per Visit)                           |
|         |    * Class: FirstContactReadyEvent in perception/contact/contact_types.py                     |
|         |    * Emitted ONCE when criteria {1, 2, 7} hold continuously for 2.0s                          |
|         |    * Consumers: Dialogue / TTS Greeting, Welcome Card UI, Engagement State                    |
|         |                                                                                               |
|         `--> OUTPUT 2: CONTINUOUS FRAME TELEMETRY (High-Frequency 30 Hz Stream)                         |
|              * Class: FirstContactResult in perception/contact/contact_types.py                         |
|              * Emitted EVERY frame: state, elapsed_eligibility (0.0s->2.0s), track_id, visit_id         |
|              * Consumers: Pan-Tilt Gaze (eye contact), UI Radial Progress Bar, Status LEDs              |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 6. Running the Automated Test Suite

Run all 382 unit tests across all milestones (M1–M6) and the interaction module:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -t . -v
```

To run individual milestone test suites:
```bash
# Milestone M6 & First Contact Suite (142 tests)
PYTHONPATH=. python3 -m unittest discover -s tests/perception/contact -t . -v

# Interaction Subsystem Suite (9 tests)
PYTHONPATH=. python3 -m unittest discover -s tests/interaction -t . -v

# Temporal Motion Suite (61 tests)
PYTHONPATH=. python3 -m unittest discover -s tests/perception/temporal -t . -v

# Frame Readiness & Observer Suite (95 tests)
PYTHONPATH=. python3 -m unittest discover -s tests/perception/readiness -t . -v

# Single-Person Tracking Suite (26 tests)
PYTHONPATH=. python3 -m unittest discover -s tests/perception/tracking -t . -v

# Person Detection Suite (49 tests)
PYTHONPATH=. python3 -m unittest discover -s tests/perception/detection -t . -v
```

---

## 7. Documentation Directory

Detailed design, integration guides, and architectural diagrams are located in [`docs/`](file:///home/eshwar/code/detection/docs/):

* **[How to Use Detection Outputs](file:///home/eshwar/code/detection/docs/how_to_use_detection_outputs.md):** Complete integration guide with code examples for downstream dialogue, navigation, ROS 2, and tablet UI systems.
* **[Detection and Interaction Integration](file:///home/eshwar/code/detection/docs/detection_and_interaction.md):** Architectural breakdown and state machine synchronization between perception and interaction.
* **[Demo & Execution Commands](file:///home/eshwar/code/detection/docs/demo_commands.md):** Exhaustive CLI reference for every milestone and configuration option.
* **[Control Flow & Architecture Diagrams](file:///home/eshwar/code/detection/docs/diagrams.md):** Comprehensive ASCII flowcharts and pipeline schematics.
* **[Simple Architecture](file:///home/eshwar/code/detection/docs/simple_architecture.md):** High-level summary of the 7 stages and decision logic.

