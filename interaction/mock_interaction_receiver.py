"""Mock Interaction Receiver and Consumer for First-Contact Readiness Outputs.

This module acts as the downstream receiver/consumer for the Vision-Based
First-Contact Readiness Perception Pipeline. It receives structured readiness events
(FIRST_CONTACT_READY and DEMO_CRITERIA_READY) and frame results from M6,
and simulates appropriate downstream robotic interaction responses (speech, UI display,
gestures, and LED indicators).

Architecture & Integration Flow:
+---------------------------------------------------------------------------------------+
|                       INTERACTION LAYER INTEGRATION FLOW                              |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   [ Vision-Based First Contact Pipeline (M1–M6) ]                                     |
|         |                                                                             |
|         +--> Emits: FirstContactResult / FirstContactReadyEvent / DemoCriteriaReady  |
|                                                                                       |
|                                   | (Direct Python API / JSON Stream / Pipe)          |
|                                   v                                                   |
|   +-------------------------------------------------------------------------------+   |
|   | MockInteractionReceiver ([mock_interaction_receiver.py](file:///home/eshwar/code/detection/interaction/mock_interaction_receiver.py))                          |   |
|   |   - Parses incoming FirstContactReadyEvent or DemoCriteriaReadyEvent          |   |
|   |   - Maintains Interaction Lifecycle State: IDLE -> ATTENDING -> ENGAGED       |   |
|   |   - Simulates Downstream Robot Actuation Actions:                             |   |
|   |       * SPEAK: Text-to-Speech greeting dialogue                               |   |
|   |       * DISPLAY: Screen UI welcome card update                                |   |
|   |       * GESTURE: Robot head nod / hand wave simulation                        |   |
|   |       * LED_INDICATOR: Hardware status ring light (Green/Amber/Cyan)          |   |
|   +-------------------------------------------------------------------------------+   |
|                                   |                                                   |
|                                   v                                                   |
|   [ Simulated Robot Action Dispatch / Console Logger / Downstream Dialogue System ]   |
|                                                                                       |
+---------------------------------------------------------------------------------------+
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import sys
import time
from typing import Any, Optional, Sequence, Union

from perception.contact.contact_types import (
    DemoCriteriaReadyEvent,
    DistanceInfo,
    FirstContactReadyEvent,
    FirstContactResult,
    ReadinessState,
    ReasonCode,
)


class InteractionState(str, Enum):
    """Lifecycle state of the interaction system."""

    IDLE = "IDLE"                        # No person present or target not ready
    ATTENDING = "ATTENDING"              # Target is in PENDING state (approaching/orienting)
    ENGAGED = "ENGAGED"                  # First contact ready! Greeting initiated
    INTERACTING = "INTERACTING"          # Ongoing active dialogue/task with user
    FAREWELL = "FAREWELL"                # Target left interaction zone; wrap-up


class RobotActionType(str, Enum):
    """Types of mock actions triggered on the robot."""

    SPEAK = "SPEAK"                      # Text-to-speech prompt
    DISPLAY = "DISPLAY"                  # Screen UI card update
    GESTURE = "GESTURE"                  # Physical robotic gesture (wave, nod, orient)
    LED_INDICATOR = "LED_INDICATOR"      # Light ring indicator (Green, Amber, Cyan, Off)


@dataclass(frozen=True)
class InteractionAction:
    """Action dispatched by the interaction layer in response to readiness outputs."""

    action_type: RobotActionType
    payload: dict[str, Any]
    timestamp: float
    visit_id: Optional[str] = None
    track_id: Optional[int] = None
    source_event: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize action to a JSON-compatible dictionary."""
        return {
            "action_type": self.action_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "visit_id": self.visit_id,
            "track_id": self.track_id,
            "source_event": self.source_event,
        }


class MockInteractionReceiver:
    """Consumes first-contact readiness outputs and simulates downstream robot responses."""

    def __init__(self, robot_name: str = "Assistant Robot") -> None:
        """Initialize mock interaction receiver.

        Args:
            robot_name: Name of the robot for simulated greetings.
        """
        self.robot_name = robot_name
        self.state: InteractionState = InteractionState.IDLE
        self.active_visit_id: Optional[str] = None
        self.active_track_id: Optional[int] = None

        self._action_history: list[InteractionAction] = []
        self._received_events_count: int = 0
        self._total_results_processed: int = 0

    @property
    def action_history(self) -> list[InteractionAction]:
        """List of all generated interaction actions."""
        return list(self._action_history)

    @property
    def received_events_count(self) -> int:
        """Total number of readiness events received."""
        return self._received_events_count

    def reset(self) -> None:
        """Reset the interaction receiver state and history."""
        self.state = InteractionState.IDLE
        self.active_visit_id = None
        self.active_track_id = None
        self._action_history.clear()
        self._received_events_count = 0
        self._total_results_processed = 0

    def handle_result(self, result: FirstContactResult) -> list[InteractionAction]:
        """Ingest per-frame FirstContactResult from ReadinessEngine.

        Args:
            result: FirstContactResult instance for the current frame.

        Returns:
            List of new InteractionAction objects triggered by this frame.
        """
        self._total_results_processed += 1
        actions: list[InteractionAction] = []
        now = time.time()

        # Update tracked visit and target ID
        self.active_visit_id = result.visit_id
        self.active_track_id = result.track_id

        # 1. Check for one-shot FirstContactReadyEvent or DemoCriteriaReadyEvent
        if result.event is not None:
            return self.handle_event(result.event)

        # 2. Check for PENDING state (target is preparing/orienting)
        if result.state == ReadinessState.PENDING:
            if self.state == InteractionState.IDLE:
                self.state = InteractionState.ATTENDING
                actions.append(
                    InteractionAction(
                        action_type=RobotActionType.LED_INDICATOR,
                        payload={"color": "AMBER", "pattern": "PULSE", "intensity": 0.6},
                        timestamp=now,
                        visit_id=result.visit_id,
                        track_id=result.track_id,
                        source_event="STATE_PENDING",
                    )
                )
                actions.append(
                    InteractionAction(
                        action_type=RobotActionType.GESTURE,
                        payload={"gesture": "HEAD_ORIENT_TOWARDS_TARGET", "track_id": result.track_id},
                        timestamp=now,
                        visit_id=result.visit_id,
                        track_id=result.track_id,
                        source_event="STATE_PENDING",
                    )
                )
                for a in actions:
                    self._record_action(a)
            return actions

        # 3. Check for NOT_READY revert
        if result.state == ReadinessState.NOT_READY:
            if self.state == InteractionState.ATTENDING:
                self.state = InteractionState.IDLE
                idle_action = InteractionAction(
                    action_type=RobotActionType.LED_INDICATOR,
                    payload={"color": "OFF", "pattern": "STEADY", "intensity": 0.0},
                    timestamp=now,
                    visit_id=result.visit_id,
                    track_id=result.track_id,
                    source_event="STATE_NOT_READY_REVERT",
                )
                self._record_action(idle_action)
                actions.append(idle_action)

        return actions

    def handle_event(
        self,
        event: Union[FirstContactReadyEvent, DemoCriteriaReadyEvent, dict[str, Any]],
    ) -> list[InteractionAction]:
        """Ingest a structured readiness event (direct object or deserialized dict).

        Args:
            event: FirstContactReadyEvent, DemoCriteriaReadyEvent, or dict.

        Returns:
            List of generated InteractionAction objects.
        """
        self._received_events_count += 1
        now = time.time()
        actions: list[InteractionAction] = []

        # Extract normalized fields
        if isinstance(event, dict):
            event_type = event.get("event_type", "UNKNOWN")
            event_id = event.get("event_id", "unknown_evt")
            visit_id = event.get("visit_id")
            track_id = event.get("track_id")
            dist_data = event.get("distance", {})
            dist_val = dist_data.get("value_m", 1.2) if isinstance(dist_data, dict) else 1.2
            selected_criteria = tuple(event.get("selected_criteria", (1, 2, 3, 4, 5, 6, 7)))
        elif isinstance(event, DemoCriteriaReadyEvent):
            event_type = event.event_type
            event_id = event.event_id
            visit_id = event.visit_id
            track_id = event.track_id
            dist_val = event.distance.value_m
            selected_criteria = event.selected_criteria
        elif isinstance(event, FirstContactReadyEvent):
            event_type = event.event_type
            event_id = event.event_id
            visit_id = event.visit_id
            track_id = event.track_id
            dist_val = event.distance.value_m
            selected_criteria = (1, 2, 3, 4, 5, 6, 7)
        else:
            raise TypeError(f"Unsupported event type: {type(event)}")

        self.state = InteractionState.ENGAGED
        self.active_visit_id = visit_id
        self.active_track_id = track_id

        # Action 1: Hardware LED Pulse
        led_color = "GREEN" if event_type == "FIRST_CONTACT_READY" else "CYAN"
        actions.append(
            InteractionAction(
                action_type=RobotActionType.LED_INDICATOR,
                payload={"color": led_color, "pattern": "GLOW_RING", "intensity": 1.0},
                timestamp=now,
                visit_id=visit_id,
                track_id=track_id,
                source_event=event_type,
            )
        )

        # Action 2: Robotic Physical Gesture
        gesture_name = "WAVE_AND_NOD" if event_type == "FIRST_CONTACT_READY" else "ATTENTIVE_NOD"
        actions.append(
            InteractionAction(
                action_type=RobotActionType.GESTURE,
                payload={"gesture": gesture_name, "speed": 1.0},
                timestamp=now,
                visit_id=visit_id,
                track_id=track_id,
                source_event=event_type,
            )
        )

        # Action 3: Text-to-Speech Greeting
        if event_type == "FIRST_CONTACT_READY":
            greeting_text = (
                f"Hello! I am {self.robot_name}. I see you are standing in front of me "
                f"at {dist_val:.1f} metres. How can I assist you today?"
            )
        else:
            sel_str = ",".join(str(n) for n in selected_criteria)
            greeting_text = (
                f"Hello! Demo readiness criteria {{{sel_str}}} satisfied at {dist_val:.1f} metres. "
                "Welcome to the demonstration!"
            )

        actions.append(
            InteractionAction(
                action_type=RobotActionType.SPEAK,
                payload={"text": greeting_text, "voice": "friendly_assistant", "volume": 0.85},
                timestamp=now,
                visit_id=visit_id,
                track_id=track_id,
                source_event=event_type,
            )
        )

        # Action 4: Screen UI Display Prompt
        actions.append(
            InteractionAction(
                action_type=RobotActionType.DISPLAY,
                payload={
                    "screen_state": "WELCOME_CARD",
                    "title": "Ready to Assist",
                    "subtitle": f"Target ID: {track_id} | Visit: {visit_id}",
                    "distance_m": dist_val,
                    "event_id": event_id,
                },
                timestamp=now,
                visit_id=visit_id,
                track_id=track_id,
                source_event=event_type,
            )
        )

        for a in actions:
            self._record_action(a)

        return actions

    def handle_visit_terminated(self, visit_id: str, reason: str = "ZONE_EXIT") -> list[InteractionAction]:
        """Ingest visit termination notification from VisitManager.

        Args:
            visit_id: Unique identifier of the visit that ended.
            reason: Termination reason (e.g. ZONE_EXIT, TRACKING_TIMEOUT).

        Returns:
            List of generated InteractionAction objects.
        """
        now = time.time()
        actions: list[InteractionAction] = []

        if self.state in (InteractionState.ENGAGED, InteractionState.INTERACTING, InteractionState.ATTENDING):
            self.state = InteractionState.FAREWELL

            farewell_text = "Goodbye! Have a wonderful day."
            actions.append(
                InteractionAction(
                    action_type=RobotActionType.SPEAK,
                    payload={"text": farewell_text, "volume": 0.75},
                    timestamp=now,
                    visit_id=visit_id,
                    track_id=self.active_track_id,
                    source_event=f"VISIT_TERMINATED:{reason}",
                )
            )
            actions.append(
                InteractionAction(
                    action_type=RobotActionType.DISPLAY,
                    payload={"screen_state": "STANDBY", "message": "Ready for next person"},
                    timestamp=now,
                    visit_id=visit_id,
                    track_id=self.active_track_id,
                    source_event=f"VISIT_TERMINATED:{reason}",
                )
            )
            actions.append(
                InteractionAction(
                    action_type=RobotActionType.LED_INDICATOR,
                    payload={"color": "OFF", "pattern": "STEADY", "intensity": 0.0},
                    timestamp=now,
                    visit_id=visit_id,
                    track_id=self.active_track_id,
                    source_event=f"VISIT_TERMINATED:{reason}",
                )
            )

            for a in actions:
                self._record_action(a)

        self.state = InteractionState.IDLE
        self.active_visit_id = None
        self.active_track_id = None
        return actions

    def receive_json(self, json_str: str) -> list[InteractionAction]:
        """Ingest a JSON string line representing a FirstContactReadyEvent or DemoCriteriaReadyEvent.

        Args:
            json_str: Serialized JSON string.

        Returns:
            List of generated InteractionAction objects.
        """
        stripped = json_str.strip()
        if not stripped:
            return []
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return self.handle_event(data)
        except json.JSONDecodeError as exc:
            print(f"[MockInteractionReceiver ERROR] Failed to parse JSON: {exc}", file=sys.stderr)
        return []

    def _record_action(self, action: InteractionAction) -> None:
        """Internal helper to record and log actions."""
        self._action_history.append(action)

    def get_summary(self) -> dict[str, Any]:
        """Return execution summary statistics of the mock interaction layer."""
        return {
            "current_state": self.state.value,
            "active_visit_id": self.active_visit_id,
            "active_track_id": self.active_track_id,
            "total_results_processed": self._total_results_processed,
            "received_events_count": self._received_events_count,
            "total_actions_dispatched": len(self._action_history),
        }


# ---------------------------------------------------------------------------
# Standalone CLI Demonstration / Pipe Consumer
# ---------------------------------------------------------------------------

def run_simulated_demo() -> None:
    """Run simulated interaction walkthrough demonstrating pipeline outputs ingestion."""
    print("==============================================================================")
    print(" Mock Interaction Receiver — Downstream Pipeline Consumer Demo")
    print("==============================================================================")
    receiver = MockInteractionReceiver(robot_name="RoboHost")

    print("\n--- Phase 1: Target Approaching (Simulated PENDING state) ---")
    simulated_pending_event = {
        "event_type": "DEMO_CRITERIA_READY",
        "event_id": "demo_evt_test_01",
        "visit_id": "visit_sim_abc123",
        "track_id": 1,
        "timestamp": time.time(),
        "readiness_duration": 2.0,
        "distance": {"value_m": 1.25, "source": "MOCK"},
        "selected_criteria": [1, 2, 7],
    }

    print("Simulating ingestion of DEMO_CRITERIA_READY output from M6...")
    actions = receiver.handle_event(simulated_pending_event)

    print(f"\n[MockInteractionReceiver] Ingested Event: {simulated_pending_event['event_type']}")
    print(f"Current State: {receiver.state.value}")
    print(f"Dispatched {len(actions)} Robotic Actions:")
    for i, act in enumerate(actions, 1):
        print(f"  {i}. [{act.action_type.value}] -> {act.payload}")

    print("\n--- Phase 2: Ingesting Full Production FIRST_CONTACT_READY Event ---")
    simulated_full_event = {
        "event_type": "FIRST_CONTACT_READY",
        "event_id": "prod_evt_test_02",
        "visit_id": "visit_sim_xyz789",
        "track_id": 2,
        "timestamp": time.time(),
        "readiness_duration": 2.0,
        "distance": {"value_m": 1.10, "source": "METRIC"},
    }
    prod_actions = receiver.handle_event(simulated_full_event)
    print(f"\n[MockInteractionReceiver] Ingested Event: {simulated_full_event['event_type']}")
    print(f"Current State: {receiver.state.value}")
    print(f"Dispatched {len(prod_actions)} Robotic Actions:")
    for i, act in enumerate(prod_actions, 1):
        print(f"  {i}. [{act.action_type.value}] -> {act.payload}")

    print("\n--- Phase 3: Person Leaves Interaction Zone (Visit Termination) ---")
    term_actions = receiver.handle_visit_terminated("visit_sim_xyz789", reason="OUTSIDE_ZONE")
    print(f"[MockInteractionReceiver] Visit Terminated. Dispatched {len(term_actions)} Actions:")
    for i, act in enumerate(term_actions, 1):
        print(f"  {i}. [{act.action_type.value}] -> {act.payload}")

    print("\n==============================================================================")
    print(" Mock Interaction Receiver Execution Summary:")
    print(json.dumps(receiver.get_summary(), indent=2))
    print("==============================================================================")


def main() -> None:
    """CLI entry point for mock interaction receiver."""
    parser = argparse.ArgumentParser(
        description="Mock Interaction Receiver for First-Contact Readiness Outputs"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run simulated interaction walkthrough with sample M6 outputs",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Listen for JSON events line-by-line on standard input (pipe mode)",
    )
    args = parser.parse_args()

    if args.listen:
        receiver = MockInteractionReceiver()
        print("[MockInteractionReceiver] Listening for JSON events on stdin...", file=sys.stderr)
        try:
            for line in sys.stdin:
                actions = receiver.receive_json(line)
                for a in actions:
                    print(json.dumps(a.to_dict()))
        except KeyboardInterrupt:
            pass
        return

    # Default to running the simulated demo
    run_simulated_demo()


if __name__ == "__main__":
    main()

