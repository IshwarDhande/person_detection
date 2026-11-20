"""Unit tests for MockInteractionReceiver in the interaction package.

Tests:
1. Default initialization and state transitions.
2. Ingestion of FirstContactReadyEvent (production event handling).
3. Ingestion of DemoCriteriaReadyEvent (demo subset event handling).
4. Ingestion of FirstContactResult (PENDING state transition & NOT_READY revert).
5. Ingestion of visit termination (farewell actions and state cleanup).
6. Parsing of serialized JSON strings and error resilience.
7. Action history logging and receiver reset.
"""

from __future__ import annotations

import json
import time
import unittest

from interaction.mock_interaction_receiver import (
    InteractionAction,
    InteractionState,
    MockInteractionReceiver,
    RobotActionType,
)
from perception.contact.contact_types import (
    ContactTimings,
    DemoCriteriaReadyEvent,
    DistanceInfo,
    DistanceObservation,
    DistanceSource,
    FirstContactReadyEvent,
    FirstContactResult,
    ReadinessPredicates,
    ReadinessState,
    ReasonCode,
)
from perception.readiness.observation_types import ObservationStatus


def _make_dummy_result(
    state: ReadinessState = ReadinessState.NOT_READY,
    event: FirstContactReadyEvent | DemoCriteriaReadyEvent | None = None,
    visit_id: str | None = "visit_123",
    track_id: int | None = 1,
) -> FirstContactResult:
    return FirstContactResult(
        state=state,
        event=event,
        predicates=ReadinessPredicates(
            p_presence=True,
            b_bbox=True,
            a_approach=True,
            z_zone=True,
            d_distance=True,
            s_stationary=True,
            o_orientation=True,
        ),
        reasons=[],
        visit_id=visit_id,
        track_id=track_id,
        elapsed_eligibility=0.0 if state != ReadinessState.READY else 2.0,
        distance=DistanceObservation(
            distance_m=1.2,
            source=DistanceSource.MOCK,
            status=ObservationStatus.VALID,
            timestamp=1.0,
            track_id=track_id,
        ),
        timings=ContactTimings(),
    )


class TestMockInteractionReceiver(unittest.TestCase):
    """Test suite for MockInteractionReceiver."""

    def setUp(self) -> None:
        self.receiver = MockInteractionReceiver(robot_name="TestBot")

    def test_initial_state(self) -> None:
        """Verify default state is IDLE and history is empty."""
        self.assertEqual(self.receiver.state, InteractionState.IDLE)
        self.assertIsNone(self.receiver.active_visit_id)
        self.assertIsNone(self.receiver.active_track_id)
        self.assertEqual(self.receiver.received_events_count, 0)
        self.assertEqual(len(self.receiver.action_history), 0)

    def test_handle_first_contact_ready_event(self) -> None:
        """Verify FIRST_CONTACT_READY event triggers production greeting and green LED."""
        event = FirstContactReadyEvent(
            event_id="evt_001",
            visit_id="visit_abc",
            track_id=5,
            timestamp=100.0,
            readiness_duration=2.0,
            distance=DistanceInfo(value_m=1.1, source="MOCK"),
        )

        actions = self.receiver.handle_event(event)

        self.assertEqual(self.receiver.state, InteractionState.ENGAGED)
        self.assertEqual(self.receiver.active_visit_id, "visit_abc")
        self.assertEqual(self.receiver.active_track_id, 5)
        self.assertEqual(self.receiver.received_events_count, 1)

        # Check action types generated
        action_types = [a.action_type for a in actions]
        self.assertIn(RobotActionType.LED_INDICATOR, action_types)
        self.assertIn(RobotActionType.GESTURE, action_types)
        self.assertIn(RobotActionType.SPEAK, action_types)
        self.assertIn(RobotActionType.DISPLAY, action_types)

        # Verify speech text includes robot name
        speak_action = next(a for a in actions if a.action_type == RobotActionType.SPEAK)
        self.assertIn("TestBot", speak_action.payload["text"])
        self.assertIn("1.1", speak_action.payload["text"])

        # Verify LED color is GREEN for full event
        led_action = next(a for a in actions if a.action_type == RobotActionType.LED_INDICATOR)
        self.assertEqual(led_action.payload["color"], "GREEN")

    def test_handle_demo_criteria_ready_event(self) -> None:
        """Verify DEMO_CRITERIA_READY event triggers demo greeting and cyan LED."""
        event = DemoCriteriaReadyEvent(
            event_id="demo_002",
            visit_id="visit_xyz",
            track_id=3,
            timestamp=200.0,
            readiness_duration=2.0,
            distance=DistanceInfo(value_m=1.3, source="MOCK"),
            selected_criteria=(1, 2, 7),
        )

        actions = self.receiver.handle_event(event)

        self.assertEqual(self.receiver.state, InteractionState.ENGAGED)
        self.assertEqual(self.receiver.active_visit_id, "visit_xyz")
        self.assertEqual(self.receiver.active_track_id, 3)

        # Verify speech mentions demo criteria {1,2,7}
        speak_action = next(a for a in actions if a.action_type == RobotActionType.SPEAK)
        self.assertIn("{1,2,7}", speak_action.payload["text"])

        # Verify LED color is CYAN for demo event
        led_action = next(a for a in actions if a.action_type == RobotActionType.LED_INDICATOR)
        self.assertEqual(led_action.payload["color"], "CYAN")

    def test_handle_result_pending_transitions_to_attending(self) -> None:
        """Verify receiving PENDING result transitions state to ATTENDING with amber LED."""
        res_pending = _make_dummy_result(state=ReadinessState.PENDING, visit_id="v1", track_id=2)
        actions = self.receiver.handle_result(res_pending)

        self.assertEqual(self.receiver.state, InteractionState.ATTENDING)
        self.assertEqual(len(actions), 2)

        led_action = next(a for a in actions if a.action_type == RobotActionType.LED_INDICATOR)
        self.assertEqual(led_action.payload["color"], "AMBER")

        gesture_action = next(a for a in actions if a.action_type == RobotActionType.GESTURE)
        self.assertEqual(gesture_action.payload["gesture"], "HEAD_ORIENT_TOWARDS_TARGET")

    def test_handle_result_not_ready_reverts_attending(self) -> None:
        """Verify NOT_READY result reverts ATTENDING back to IDLE and turns off LED."""
        # Enter ATTENDING first
        res_pending = _make_dummy_result(state=ReadinessState.PENDING, visit_id="v1", track_id=2)
        self.receiver.handle_result(res_pending)
        self.assertEqual(self.receiver.state, InteractionState.ATTENDING)

        # Target turns away -> NOT_READY
        res_not_ready = _make_dummy_result(state=ReadinessState.NOT_READY, visit_id="v1", track_id=2)
        actions = self.receiver.handle_result(res_not_ready)

        self.assertEqual(self.receiver.state, InteractionState.IDLE)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, RobotActionType.LED_INDICATOR)
        self.assertEqual(actions[0].payload["color"], "OFF")

    def test_handle_visit_terminated(self) -> None:
        """Verify visit termination triggers farewell dialogue and resets state to IDLE."""
        # Engage target first
        event = FirstContactReadyEvent(
            event_id="evt_003",
            visit_id="visit_farewell",
            track_id=1,
            timestamp=50.0,
            readiness_duration=2.0,
            distance=DistanceInfo(value_m=1.0, source="MOCK"),
        )
        self.receiver.handle_event(event)
        self.assertEqual(self.receiver.state, InteractionState.ENGAGED)

        # Person steps outside zone -> visit terminated
        actions = self.receiver.handle_visit_terminated("visit_farewell", reason="ZONE_EXIT")

        self.assertEqual(self.receiver.state, InteractionState.IDLE)
        self.assertIsNone(self.receiver.active_visit_id)

        action_types = [a.action_type for a in actions]
        self.assertIn(RobotActionType.SPEAK, action_types)
        self.assertIn(RobotActionType.DISPLAY, action_types)
        self.assertIn(RobotActionType.LED_INDICATOR, action_types)

        speak_act = next(a for a in actions if a.action_type == RobotActionType.SPEAK)
        self.assertIn("Goodbye", speak_act.payload["text"])

    def test_receive_json_string(self) -> None:
        """Verify receiver can parse and handle serialized JSON string lines."""
        json_line = json.dumps({
            "event_type": "FIRST_CONTACT_READY",
            "event_id": "json_evt_99",
            "visit_id": "visit_json",
            "track_id": 7,
            "timestamp": 123.45,
            "readiness_duration": 2.0,
            "distance": {"value_m": 1.2, "source": "MOCK"},
        })

        actions = self.receiver.receive_json(json_line)
        self.assertEqual(self.receiver.state, InteractionState.ENGAGED)
        self.assertEqual(len(actions), 4)
        self.assertEqual(self.receiver.received_events_count, 1)

    def test_receive_malformed_json_fails_gracefully(self) -> None:
        """Verify invalid JSON string is rejected gracefully without crashing."""
        actions = self.receiver.receive_json("{invalid json string...")
        self.assertEqual(len(actions), 0)
        self.assertEqual(self.receiver.state, InteractionState.IDLE)

    def test_receiver_reset(self) -> None:
        """Verify reset clears state, counts, and history."""
        event = FirstContactReadyEvent(
            event_id="evt_004",
            visit_id="v_reset",
            track_id=1,
            timestamp=10.0,
            readiness_duration=2.0,
            distance=DistanceInfo(value_m=1.0, source="MOCK"),
        )
        self.receiver.handle_event(event)
        self.assertGreater(len(self.receiver.action_history), 0)

        self.receiver.reset()
        self.assertEqual(self.receiver.state, InteractionState.IDLE)
        self.assertEqual(len(self.receiver.action_history), 0)
        self.assertEqual(self.receiver.received_events_count, 0)


if __name__ == "__main__":
    unittest.main()

