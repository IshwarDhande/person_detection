"""Unit tests for ReadinessStateMachine (Milestone M6).

Tests state transitions (NOT_READY -> PENDING -> READY), revocability,
one-shot FIRST_CONTACT_READY event generation, and multi-visit lifecycle.
"""

from __future__ import annotations

import unittest

from perception.contact.contact_types import (
    DistanceObservation,
    DistanceSource,
    FirstContactReadyEvent,
    ReadinessState,
)
from perception.contact.readiness_state_machine import ReadinessStateMachine
from perception.contact.visit_manager import VisitManager
from perception.readiness.observation_types import (
    BboxMetrics,
    BboxValidityValue,
    FaceObservations,
    FaceVisibilityValue,
    FrameMetadata,
    FrameObservation,
    HeadOrientationValue,
    InteractionZoneValue,
    Observation,
    ObservationStatus,
    OrientationObservations,
    PersonObservations,
    PersonPresenceValue,
)
from perception.tracking.single_person_tracker import Track, TrackState


def _make_track(track_id: int = 1, observed: bool = True, timestamp: float = 1.0) -> Track:
    bbox = (100.0, 100.0, 300.0, 500.0)
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox,
        center=(200.0, 300.0),
        velocity=(0.0, 0.0, 0.0, 0.0),
        timestamp=timestamp,
        creation_timestamp=timestamp,
        last_seen_timestamp=timestamp,
        hit_count=3,
        miss_count=0,
        state=TrackState.CONFIRMED,
        observed=observed,
    )


def _make_dummy_frame_obs(timestamp: float = 1.0) -> FrameObservation:
    bbox = (100.0, 100.0, 300.0, 500.0)
    metrics = BboxMetrics(
        original_bbox_xyxy=bbox,
        clipped_bbox_xyxy=bbox,
        width=200.0,
        height=400.0,
        area=80000.0,
        area_ratio=80000.0 / (1280 * 720),
        image_width=1280,
        image_height=720,
    )
    return FrameObservation(
        person=PersonObservations(
            presence=Observation(value=PersonPresenceValue.DETECTED, confidence=0.9, status=ObservationStatus.VALID),
            bounding_box=Observation(value=BboxValidityValue.VALID, confidence=None, status=ObservationStatus.VALID),
            bbox_metrics=metrics,
            interaction_zone=Observation(value=InteractionZoneValue.INSIDE, confidence=None, status=ObservationStatus.VALID),
        ),
        face=FaceObservations(
            visibility=Observation(value=FaceVisibilityValue.SUFFICIENT, confidence=0.9, status=ObservationStatus.VALID),
            metrics=None,
        ),
        orientation=OrientationObservations(
            head_orientation=Observation(value=HeadOrientationValue.FACING_ROBOT, confidence=None, status=ObservationStatus.VALID),
            metrics=None,
        ),
        frame_metadata=FrameMetadata(
            timestamp=timestamp,
            image_width=1280,
            image_height=720,
        ),
    )


class TestReadinessStateMachine(unittest.TestCase):
    """Test suite for ReadinessStateMachine."""

    def setUp(self) -> None:
        self.sm = ReadinessStateMachine(eligibility_duration_seconds=2.0)
        self.vm = VisitManager()
        # Seed an active visit
        self.track = _make_track(track_id=1, timestamp=1.0)
        self.vm.update(self.track, _make_dummy_frame_obs(1.0), None, timestamp=1.0)
        self.dist_obs = DistanceObservation(
            distance_m=1.2,
            status=ObservationStatus.VALID,
            source=DistanceSource.MOCK,
            timestamp=1.0,
        )

    def test_initial_state(self) -> None:
        """Verify initial state machine properties."""
        sm = ReadinessStateMachine(eligibility_duration_seconds=2.0)
        self.assertEqual(sm.state, ReadinessState.NOT_READY)
        self.assertIsNone(sm.eligible_since)
        self.assertIsNone(sm.ready_since)
        self.assertEqual(sm.elapsed_eligibility, 0.0)

    def test_no_visit_stays_not_ready(self) -> None:
        """Verify when no visit is active, state machine returns NOT_READY."""
        vm_empty = VisitManager()
        state, event, elapsed = self.sm.update(
            is_all_valid=True,
            visit_manager=vm_empty,
            track_id=1,
            distance_obs=self.dist_obs,
            timestamp=1.0,
        )
        self.assertEqual(state, ReadinessState.NOT_READY)
        self.assertIsNone(event)
        self.assertEqual(elapsed, 0.0)

    def test_transition_not_ready_to_pending(self) -> None:
        """Verify first eligible frame transitions to PENDING."""
        state, event, elapsed = self.sm.update(
            is_all_valid=True,
            visit_manager=self.vm,
            track_id=1,
            distance_obs=self.dist_obs,
            timestamp=1.0,
        )
        self.assertEqual(state, ReadinessState.PENDING)
        self.assertIsNone(event)
        self.assertEqual(elapsed, 0.0)
        self.assertEqual(self.sm.eligible_since, 1.0)
        self.assertIsNone(self.sm.ready_since)

    def test_pending_state_accumulation(self) -> None:
        """Verify continuing eligibility stays PENDING while < 2.0s."""
        t0 = 1.0
        dt = 0.1
        for i in range(11):  # 1.0s elapsed: t = 1.0 to 2.0
            state, event, elapsed = self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=t0 + i * dt)

        self.assertEqual(state, ReadinessState.PENDING)
        self.assertIsNone(event)
        self.assertAlmostEqual(elapsed, 1.0)

    def test_transition_pending_to_ready_and_event_emitted(self) -> None:
        """Verify reaching 2.0s continuous validity transitions to READY and emits event."""
        t0 = 1.0
        dt = 0.1
        for i in range(21):  # 2.0s elapsed: t = 1.0 to 3.0
            state, event, elapsed = self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=t0 + i * dt)

        self.assertEqual(state, ReadinessState.READY)
        self.assertAlmostEqual(elapsed, 2.0)
        self.assertEqual(self.sm.ready_since, 3.0)
        self.assertIsNotNone(event)
        self.assertIsInstance(event, FirstContactReadyEvent)
        self.assertEqual(event.visit_id, self.vm.active_visit_id)
        self.assertEqual(event.track_id, 1)
        self.assertEqual(event.timestamp, 3.0)
        self.assertEqual(event.readiness_duration, 2.0)
        self.assertEqual(event.distance.value_m, 1.2)
        self.assertEqual(event.distance.source, "MOCK")
        self.assertTrue(self.vm.event_emitted)

    def test_revocability_ready_to_not_ready(self) -> None:
        """Verify failure of conditions while in READY revokes readiness immediately."""
        t0 = 1.0
        dt = 0.1
        # Reach READY (2.0s elapsed: t = 1.0 to 3.0)
        for i in range(21):
            self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=t0 + i * dt)
        self.assertEqual(self.sm.state, ReadinessState.READY)

        # Condition fails at t = 3.1
        state, event, elapsed = self.sm.update(
            is_all_valid=False,
            visit_manager=self.vm,
            track_id=1,
            distance_obs=self.dist_obs,
            timestamp=3.1,
        )
        self.assertEqual(state, ReadinessState.NOT_READY)
        self.assertIsNone(event)
        self.assertEqual(elapsed, 0.0)
        self.assertIsNone(self.sm.eligible_since)
        self.assertIsNone(self.sm.ready_since)

    def test_revocability_pending_to_not_ready(self) -> None:
        """Verify failure while in PENDING resets to NOT_READY."""
        self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=1.0)
        self.assertEqual(self.sm.state, ReadinessState.PENDING)

        state, event, elapsed = self.sm.update(
            is_all_valid=False,
            visit_manager=self.vm,
            track_id=1,
            distance_obs=self.dist_obs,
            timestamp=1.1,
        )
        self.assertEqual(state, ReadinessState.NOT_READY)
        self.assertIsNone(event)
        self.assertEqual(elapsed, 0.0)

    def test_no_duplicate_event_on_reentering_ready(self) -> None:
        """Verify re-entering READY within the same visit does not emit a duplicate event."""
        t0 = 1.0
        dt = 0.1
        # First arrival at READY (2.0s: t = 1.0 to 3.0)
        for i in range(20):
            self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=t0 + i * dt)
        _, event1, _ = self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=3.0)
        self.assertIsNotNone(event1)
        self.assertTrue(self.vm.event_emitted)

        # Drop out of readiness (person looks away at t = 3.1)
        self.sm.update(False, self.vm, 1, self.dist_obs, timestamp=3.1)
        self.assertEqual(self.sm.state, ReadinessState.NOT_READY)

        # Re-accumulate 2.0s (t = 4.0 to 6.0)
        for i in range(20):
            self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=4.0 + i * dt)
        state, event2, elapsed = self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=6.0)

        # Re-enters READY, but NO event emitted
        self.assertEqual(state, ReadinessState.READY)
        self.assertIsNone(event2)
        self.assertAlmostEqual(elapsed, 2.0)

    def test_new_visit_allows_new_event(self) -> None:
        """Verify a subsequent new visit can emit a new event."""
        t0 = 1.0
        dt = 0.1
        # First visit reaches READY
        for i in range(20):
            self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=t0 + i * dt)
        _, event1, _ = self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=3.0)
        self.assertIsNotNone(event1)
        old_visit_id = self.vm.active_visit_id

        # End first visit
        self.vm.terminate_visit("RELIABLE_ZONE_EXIT")
        self.assertFalse(self.vm.has_active_visit)

        # Start second visit with new track
        track2 = _make_track(track_id=2, timestamp=10.0)
        self.vm.update(track2, _make_dummy_frame_obs(10.0), None, timestamp=10.0)
        new_visit_id = self.vm.active_visit_id
        self.assertNotEqual(old_visit_id, new_visit_id)
        self.assertFalse(self.vm.event_emitted)

        # Second visit achieves READY (t = 10.0 to 12.0)
        for i in range(20):
            self.sm.update(True, self.vm, 2, self.dist_obs, timestamp=10.0 + i * dt)
        state, event2, _ = self.sm.update(True, self.vm, 2, self.dist_obs, timestamp=12.0)

        self.assertEqual(state, ReadinessState.READY)
        self.assertIsNotNone(event2)
        self.assertEqual(event2.visit_id, new_visit_id)
        self.assertEqual(event2.track_id, 2)
        self.assertNotEqual(event1.event_id, event2.event_id)

    def test_reset(self) -> None:
        """Verify reset clears state machine completely."""
        self.sm.update(True, self.vm, 1, self.dist_obs, timestamp=1.0)
        self.sm.reset()

        self.assertEqual(self.sm.state, ReadinessState.NOT_READY)
        self.assertIsNone(self.sm.eligible_since)
        self.assertIsNone(self.sm.ready_since)
        self.assertEqual(self.sm.elapsed_eligibility, 0.0)


if __name__ == "__main__":
    unittest.main()

