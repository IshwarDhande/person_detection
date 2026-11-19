"""Unit tests for ReadinessEngine (Milestone M6).

Tests full orchestration of predicates, visit manager, state machine,
timings instrumentation, and distance provider integration.
"""

from __future__ import annotations

import unittest

from perception.contact.contact_config import ContactConfig
from perception.contact.contact_types import (
    DistanceObservation,
    DistanceSource,
    FirstContactReadyEvent,
    ReadinessState,
    ReasonCode,
)
from perception.contact.distance_provider import MockDistanceProvider
from perception.contact.readiness_engine import ReadinessEngine
from perception.detection.person_detector import PersonDetection
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
from perception.temporal.observation_types import (
    ApproachValue,
    InitializationState,
    MotionMetrics,
    MovementValue,
    StabilityObservation,
    StabilityValue,
    TemporalObservation,
)
from perception.tracking.single_person_tracker import Track, TrackState


def _make_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (300.0, 200.0, 500.0, 600.0),
    observed: bool = True,
    timestamp: float = 1.0,
) -> Track:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox,
        center=(cx, cy),
        velocity=(0.0, 0.0, 0.0, 0.0),
        timestamp=timestamp,
        creation_timestamp=timestamp,
        last_seen_timestamp=timestamp,
        hit_count=3,
        miss_count=0,
        state=TrackState.CONFIRMED,
        observed=observed,
    )


def _make_frame_obs(
    bbox: tuple[float, float, float, float] = (300.0, 200.0, 500.0, 600.0),
    presence_val: PersonPresenceValue = PersonPresenceValue.DETECTED,
    presence_stat: ObservationStatus = ObservationStatus.VALID,
    bbox_val: BboxValidityValue = BboxValidityValue.VALID,
    bbox_stat: ObservationStatus = ObservationStatus.VALID,
    zone_val: InteractionZoneValue = InteractionZoneValue.INSIDE,
    zone_stat: ObservationStatus = ObservationStatus.VALID,
    orient_val: HeadOrientationValue = HeadOrientationValue.FACING_ROBOT,
    orient_stat: ObservationStatus = ObservationStatus.VALID,
    timestamp: float = 1.0,
) -> FrameObservation:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    metrics = BboxMetrics(
        original_bbox_xyxy=bbox,
        clipped_bbox_xyxy=bbox,
        width=w,
        height=h,
        area=w * h,
        area_ratio=(w * h) / (1280 * 720),
        image_width=1280,
        image_height=720,
    )
    return FrameObservation(
        person=PersonObservations(
            presence=Observation(value=presence_val, confidence=0.9, status=presence_stat),
            bounding_box=Observation(value=bbox_val, confidence=None, status=bbox_stat),
            bbox_metrics=metrics,
            interaction_zone=Observation(value=zone_val, confidence=None, status=zone_stat),
        ),
        face=FaceObservations(
            visibility=Observation(value=FaceVisibilityValue.SUFFICIENT, confidence=0.9, status=ObservationStatus.VALID),
            metrics=None,
        ),
        orientation=OrientationObservations(
            head_orientation=Observation(value=orient_val, confidence=None, status=orient_stat),
            metrics=None,
        ),
        frame_metadata=FrameMetadata(
            timestamp=timestamp,
            image_width=1280,
            image_height=720,
        ),
    )


def _make_temporal_obs(
    movement_val: MovementValue = MovementValue.STATIONARY,
    movement_stat: ObservationStatus = ObservationStatus.VALID,
    approach_val: ApproachValue = ApproachValue.APPROACHING,
    approach_stat: ObservationStatus = ObservationStatus.VALID,
    timestamp: float = 1.0,
) -> TemporalObservation:
    return TemporalObservation(
        timestamp=timestamp,
        track_id=1,
        observed=True,
        approach=Observation(value=approach_val, confidence=None, status=approach_stat),
        movement=Observation(value=movement_val, confidence=None, status=movement_stat),
        stability=StabilityObservation(
            value=StabilityValue.ACCUMULATING,
            elapsed_seconds=1.0,
            violation_seconds=0.0,
            status=ObservationStatus.VALID,
        ),
        motion=MotionMetrics(),
        initialization_state=InitializationState.READY,
        initialization_count=5,
    )


class TestReadinessEngine(unittest.TestCase):
    """Test suite for ReadinessEngine."""

    def setUp(self) -> None:
        self.config = ContactConfig(
            distance_mode="mock",
            mock_distance_m=1.2,
            distance_max_m=1.5,
            eligibility_duration_seconds=2.0,
            tracking_loss_timeout_seconds=5.0,
        )
        self.engine = ReadinessEngine(config=self.config)
        self.box = (300.0, 200.0, 500.0, 600.0)
        self.det = PersonDetection(bbox_xyxy=self.box, confidence=0.92, class_id=0)
        self.track = _make_track(track_id=1, bbox=self.box, timestamp=1.0)

    def test_default_initialization(self) -> None:
        """Verify default engine instantiation."""
        engine = ReadinessEngine()
        self.assertIsNotNone(engine.config)
        self.assertIsNotNone(engine.distance_provider)
        self.assertIsNotNone(engine.visit_manager)
        self.assertIsNotNone(engine.predicate_evaluator)
        self.assertIsNotNone(engine.state_machine)
        self.assertIsNone(engine.last_result)

    def test_manual_reset(self) -> None:
        """Verify manual engine reset clears visit and state machine."""
        frame_obs = _make_frame_obs(timestamp=1.0)
        temp_obs = _make_temporal_obs(timestamp=1.0)
        self.engine.evaluate(self.track, frame_obs, temp_obs, self.det, timestamp=1.0)

        self.assertIsNotNone(self.engine.last_result)
        self.assertTrue(self.engine.visit_manager.has_active_visit)

        self.engine.reset()
        self.assertIsNone(self.engine.last_result)
        self.assertFalse(self.engine.visit_manager.has_active_visit)
        self.assertEqual(self.engine.state_machine.state, ReadinessState.NOT_READY)

    def test_first_frame_starts_visit_and_transitions_to_pending(self) -> None:
        """Verify initial frame with all conditions satisfied enters PENDING."""
        frame_obs = _make_frame_obs(timestamp=1.0)
        temp_obs = _make_temporal_obs(timestamp=1.0)

        result = self.engine.evaluate(
            track=self.track,
            frame_observation=frame_obs,
            temporal_observation=temp_obs,
            matched_detection=self.det,
            timestamp=1.0,
        )

        self.assertEqual(result.state, ReadinessState.PENDING)
        self.assertIsNone(result.event)
        self.assertTrue(result.predicates.is_all_valid)
        self.assertEqual(len(result.reasons), 0)
        self.assertEqual(result.elapsed_eligibility, 0.0)
        self.assertIsNotNone(result.visit_id)
        self.assertEqual(result.track_id, 1)

    def test_readiness_event_emitted_after_two_seconds(self) -> None:
        """Verify reaching 2.0 continuous seconds produces FIRST_CONTACT_READY event."""
        t0 = 1.0
        dt = 0.1  # 10 FPS continuous frames

        # Feed 10 frames: t = 1.0 to 1.9s -> PENDING
        for i in range(10):
            t = t0 + i * dt
            trk = _make_track(track_id=1, bbox=self.box, timestamp=t)
            res_mid = self.engine.evaluate(
                trk,
                _make_frame_obs(timestamp=t),
                _make_temporal_obs(timestamp=t),
                self.det,
                timestamp=t,
            )
            self.assertEqual(res_mid.state, ReadinessState.PENDING)
            self.assertIsNone(res_mid.event)

        # Feed remaining frames up to t = 3.0s (2.0s elapsed from 1.0s) -> transitions to READY
        for i in range(10, 21):
            t = t0 + i * dt
            trk = _make_track(track_id=1, bbox=self.box, timestamp=t)
            res_final = self.engine.evaluate(
                trk,
                _make_frame_obs(timestamp=t),
                _make_temporal_obs(timestamp=t),
                self.det,
                timestamp=t,
            )

        self.assertEqual(res_final.state, ReadinessState.READY)
        self.assertIsNotNone(res_final.event)
        self.assertIsInstance(res_final.event, FirstContactReadyEvent)
        self.assertEqual(res_final.event.visit_id, res_final.visit_id)
        self.assertEqual(res_final.event.track_id, 1)
        self.assertEqual(res_final.event.timestamp, 3.0)
        self.assertEqual(res_final.event.distance.value_m, 1.2)
        self.assertEqual(res_final.event.distance.source, "MOCK")

    def test_broken_predicate_reverts_state_to_not_ready(self) -> None:
        """Verify unsatisfied predicate transitions state to NOT_READY with reasons."""
        # Frame 1 at t = 1.0 -> PENDING
        self.engine.evaluate(
            self.track,
            _make_frame_obs(timestamp=1.0),
            _make_temporal_obs(timestamp=1.0),
            self.det,
            timestamp=1.0,
        )

        # Frame 2 at t = 1.1: person turns away (NOT_FACING_ROBOT)
        trk_broken = _make_track(track_id=1, bbox=self.box, timestamp=1.1)
        res_broken = self.engine.evaluate(
            trk_broken,
            _make_frame_obs(orient_val=HeadOrientationValue.NOT_FACING_ROBOT, timestamp=1.1),
            _make_temporal_obs(timestamp=1.1),
            self.det,
            timestamp=1.1,
        )
        self.assertEqual(res_broken.state, ReadinessState.NOT_READY)
        self.assertIsNone(res_broken.event)
        self.assertFalse(res_broken.predicates.O)
        self.assertIn(ReasonCode.NOT_FACING_ROBOT, res_broken.reasons)
        self.assertEqual(res_broken.elapsed_eligibility, 0.0)

    def test_timings_instrumented(self) -> None:
        """Verify execution timings are populated with non-negative values."""
        frame_obs = _make_frame_obs(timestamp=1.0)
        temp_obs = _make_temporal_obs(timestamp=1.0)

        result = self.engine.evaluate(
            self.track, frame_obs, temp_obs, self.det, timestamp=1.0
        )
        timings = result.timings
        self.assertGreaterEqual(timings.predicate_eval_ms, 0.0)
        self.assertGreaterEqual(timings.visit_management_ms, 0.0)
        self.assertGreaterEqual(timings.state_machine_ms, 0.0)
        self.assertGreaterEqual(timings.total_ms, 0.0)

    def test_metric_distance_mode_fails_closed(self) -> None:
        """Verify metric distance mode fails closed when uncalibrated/sensor missing."""
        cfg_metric = ContactConfig(
            distance_mode="metric",
            distance_max_m=1.5,
        )
        engine_metric = ReadinessEngine(config=cfg_metric)
        frame_obs = _make_frame_obs(timestamp=1.0)
        temp_obs = _make_temporal_obs(timestamp=1.0)

        result = engine_metric.evaluate(
            self.track, frame_obs, temp_obs, self.det, timestamp=1.0
        )
        self.assertFalse(result.predicates.D)
        self.assertIn(ReasonCode.DISTANCE_UNAVAILABLE, result.reasons)
        self.assertEqual(result.state, ReadinessState.NOT_READY)

    def test_last_result_property(self) -> None:
        """Verify last_result reflects the most recent evaluate output."""
        frame_obs = _make_frame_obs(timestamp=1.0)
        temp_obs = _make_temporal_obs(timestamp=1.0)

        result = self.engine.evaluate(
            self.track, frame_obs, temp_obs, self.det, timestamp=1.0
        )
        self.assertIs(self.engine.last_result, result)


if __name__ == "__main__":
    unittest.main()

