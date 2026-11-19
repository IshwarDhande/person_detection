"""Unit tests for individual readiness predicates and explainable reason codes."""

import unittest

from perception.contact.contact_config import ContactConfig
from perception.contact.contact_types import (
    DistanceObservation,
    DistanceSource,
    ReasonCode,
)
from perception.contact.predicates import PredicateEvaluator
from perception.contact.visit_manager import VisitManager
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
    timestamp: float = 10.0,
) -> FrameObservation:
    """Construct synthetic FrameObservation with explicit condition values."""
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
    timestamp: float = 10.0,
) -> TemporalObservation:
    """Construct synthetic TemporalObservation."""
    return TemporalObservation(
        timestamp=timestamp,
        track_id=1,
        observed=True,
        approach=Observation(value=approach_val, confidence=None, status=approach_stat),
        movement=Observation(value=movement_val, confidence=None, status=movement_stat),
        stability=StabilityObservation(value=StabilityValue.ACCUMULATING, elapsed_seconds=1.0, violation_seconds=0.0, status=ObservationStatus.VALID),
        motion=MotionMetrics(),
        initialization_state=InitializationState.READY,
        initialization_count=5,
    )


class TestPredicates(unittest.TestCase):
    """Test every readiness predicate independently and verify reason codes."""

    def setUp(self) -> None:
        self.config = ContactConfig(distance_max_m=1.5, observation_freshness_seconds=0.5)
        self.evaluator = PredicateEvaluator(self.config)
        self.box = (300.0, 200.0, 500.0, 600.0)
        self.det = PersonDetection(bbox_xyxy=self.box, confidence=0.92, class_id=0)
        self.track = Track(
            track_id=1,
            bbox_xyxy=self.box,
            center=(400.0, 400.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=10.0,
            last_seen_timestamp=10.0,
            hit_count=3,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )
        self.visit_manager = VisitManager()
        # Seed visit with approach confirmed
        self.visit_manager.update(
            track=self.track,
            frame_observation=_make_frame_obs(bbox=self.box, timestamp=10.0),
            temporal_observation=_make_temporal_obs(timestamp=10.0),
            timestamp=10.0,
        )

    def test_all_seven_predicates_valid(self) -> None:
        """Verify when all conditions are satisfied, all predicates evaluate True."""
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist_obs = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist_obs,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )

        self.assertTrue(preds.p_presence)
        self.assertTrue(preds.b_bbox)
        self.assertTrue(preds.a_approach)
        self.assertTrue(preds.z_zone)
        self.assertTrue(preds.d_distance)
        self.assertTrue(preds.s_stationary)
        self.assertTrue(preds.o_orientation)
        self.assertTrue(preds.is_all_valid)
        self.assertEqual(len(reasons), 0)

    def test_predicate_p_missing_person(self) -> None:
        """Verify missing person presence makes P=False and emits NO_PERSON."""
        f_obs = _make_frame_obs(presence_val=PersonPresenceValue.NOT_DETECTED, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.p_presence)
        self.assertIn(ReasonCode.NO_PERSON, reasons)

    def test_predicate_p_unobserved_coasting_track(self) -> None:
        """Verify unobserved Kalman coasting track cannot satisfy P."""
        coasting_track = Track(
            track_id=1,
            bbox_xyxy=self.box,
            center=(400.0, 400.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=9.0,
            last_seen_timestamp=9.5,
            hit_count=0,
            miss_count=1,
            state=TrackState.LOST,
            observed=False,  # Unobserved!
        )
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=coasting_track,
            matched_detection=None,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.p_presence)
        self.assertIn(ReasonCode.TRACKING_LOST, reasons)

    def test_predicate_b_mismatched_detection(self) -> None:
        """Verify detection coordinate mismatch fails B with INVALID_BBOX."""
        diff_det = PersonDetection(bbox_xyxy=(10.0, 10.0, 50.0, 100.0), confidence=0.8, class_id=0)
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=diff_det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.b_bbox)
        self.assertIn(ReasonCode.INVALID_BBOX, reasons)

    def test_predicate_z_outside_zone(self) -> None:
        """Verify person outside interaction zone fails Z with OUTSIDE_INTERACTION_ZONE."""
        f_obs = _make_frame_obs(zone_val=InteractionZoneValue.OUTSIDE, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.z_zone)
        self.assertIn(ReasonCode.OUTSIDE_INTERACTION_ZONE, reasons)

    def test_predicate_d_too_far(self) -> None:
        """Verify distance exceeding distance_max_m fails D with TOO_FAR."""
        f_obs = _make_frame_obs(timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        # 2.2m exceeds 1.5m threshold
        dist = DistanceObservation(distance_m=2.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.d_distance)
        self.assertIn(ReasonCode.TOO_FAR, reasons)

    def test_predicate_d_unavailable_or_unreliable(self) -> None:
        """Verify invalid/uncertain distance fails D with appropriate reason."""
        f_obs = _make_frame_obs(timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)

        # Distance None -> DISTANCE_UNAVAILABLE
        preds1, reasons1 = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=None,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds1.d_distance)
        self.assertIn(ReasonCode.DISTANCE_UNAVAILABLE, reasons1)

        # Distance UNCERTAIN -> DISTANCE_UNRELIABLE
        dist_unc = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.UNCERTAIN, timestamp=10.0)
        preds2, reasons2 = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist_unc,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds2.d_distance)
        self.assertIn(ReasonCode.DISTANCE_UNRELIABLE, reasons2)

    def test_predicate_s_person_moving(self) -> None:
        """Verify MOVING state fails S with PERSON_MOVING."""
        f_obs = _make_frame_obs(timestamp=10.0)
        t_obs = _make_temporal_obs(movement_val=MovementValue.MOVING, timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.s_stationary)
        self.assertIn(ReasonCode.PERSON_MOVING, reasons)

    def test_predicate_o_not_facing_robot(self) -> None:
        """Verify NOT_FACING_ROBOT fails O with NOT_FACING_ROBOT."""
        f_obs = _make_frame_obs(orient_val=HeadOrientationValue.NOT_FACING_ROBOT, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.o_orientation)
        self.assertIn(ReasonCode.NOT_FACING_ROBOT, reasons)

    def test_stale_observation_flagged(self) -> None:
        """Verify observation older than freshness threshold adds STALE_OBSERVATION and invalidates predicates."""
        # Frame timestamp is 10.0, evaluation timestamp is 11.0 (difference 1.0s > 0.5s)
        f_obs = _make_frame_obs(timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=11.0,  # Stale!
        )
        self.assertIn(ReasonCode.STALE_OBSERVATION, reasons)
        self.assertFalse(preds.p_presence)
        self.assertFalse(preds.is_all_valid)

    def test_active_target_missing_while_another_person_observed(self) -> None:
        """Verify when active target (track 1) is missing, another track (track 2) cannot satisfy readiness."""
        # Track 2 is observed
        track2 = Track(
            track_id=2,
            bbox_xyxy=(50.0, 50.0, 150.0, 250.0),
            center=(100.0, 150.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=10.0,
            last_seen_timestamp=10.0,
            hit_count=3,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )
        det2 = PersonDetection(bbox_xyxy=(50.0, 50.0, 150.0, 250.0), confidence=0.9, class_id=0)
        f_obs = _make_frame_obs(bbox=(50.0, 50.0, 150.0, 250.0), timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=track2,
            matched_detection=det2,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,  # Locked to track 1!
            timestamp=10.0,
        )
        self.assertFalse(preds.p_presence)
        self.assertFalse(preds.is_all_valid)
        self.assertIn(ReasonCode.TARGET_MISMATCH, reasons)

    def test_m3_track_id_and_temporal_obs_id_mismatch(self) -> None:
        """Verify temporal observation with mismatched track ID fails S with TARGET_MISMATCH."""
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        # Temporal observation for track 99
        t_obs_mismatched = TemporalObservation(
            timestamp=10.0,
            track_id=99,
            observed=True,
            approach=Observation(value=ApproachValue.APPROACHING, confidence=None, status=ObservationStatus.VALID),
            movement=Observation(value=MovementValue.STATIONARY, confidence=None, status=ObservationStatus.VALID),
            stability=StabilityObservation(value=StabilityValue.ACCUMULATING, elapsed_seconds=1.0, violation_seconds=0.0, status=ObservationStatus.VALID),
            motion=MotionMetrics(),
            initialization_state=InitializationState.READY,
            initialization_count=5,
        )
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs_mismatched,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.s_stationary)
        self.assertFalse(preds.is_all_valid)
        self.assertIn(ReasonCode.TARGET_MISMATCH, reasons)

    def test_stale_m5_temporal_observation(self) -> None:
        """Verify stale temporal observation fails S with STALE_OBSERVATION / TIMESTAMP_MISMATCH."""
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        # Temporal observation timestamp is 8.0s (2.0s stale)
        t_obs_stale = _make_temporal_obs(timestamp=8.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs_stale,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.s_stationary)
        self.assertIn(ReasonCode.STALE_OBSERVATION, reasons)

    def test_stale_distance_observation(self) -> None:
        """Verify stale distance observation fails D with STALE_OBSERVATION."""
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        # Distance observation timestamp is 7.0s (3.0s stale)
        dist_stale = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=7.0)

        preds, reasons = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist_stale,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.d_distance)
        self.assertIn(ReasonCode.STALE_OBSERVATION, reasons)

    def test_mismatched_timestamps_across_observations(self) -> None:
        """Verify mismatched timestamps across track, frame, and temporal data fail validation."""
        # Track timestamp is 8.0s while frame is 10.0s
        track_mismatched = Track(
            track_id=1,
            bbox_xyxy=self.box,
            center=(400.0, 400.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=8.0,
            creation_timestamp=5.0,
            last_seen_timestamp=8.0,
            hit_count=3,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        preds, reasons = self.evaluator.evaluate(
            track=track_mismatched,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=10.0,
        )
        self.assertFalse(preds.p_presence)
        self.assertIn(ReasonCode.TIMESTAMP_MISMATCH, reasons)

    def test_nan_and_inf_timestamps_rejected(self) -> None:
        """Verify non-finite timestamps (NaN, inf) fail closed with INVALID_TIMESTAMP."""
        f_obs = _make_frame_obs(bbox=self.box, timestamp=10.0)
        t_obs = _make_temporal_obs(timestamp=10.0)
        dist = DistanceObservation(distance_m=1.2, source=DistanceSource.MOCK, status=ObservationStatus.VALID, timestamp=10.0)

        # NaN timestamp
        preds_nan, reasons_nan = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=float("nan"),
        )
        self.assertFalse(preds_nan.is_all_valid)
        self.assertIn(ReasonCode.INVALID_TIMESTAMP, reasons_nan)

        # Inf timestamp
        preds_inf, reasons_inf = self.evaluator.evaluate(
            track=self.track,
            matched_detection=self.det,
            frame_observation=f_obs,
            temporal_observation=t_obs,
            distance_observation=dist,
            visit_manager=self.visit_manager,
            timestamp=float("inf"),
        )
        self.assertFalse(preds_inf.is_all_valid)
        self.assertIn(ReasonCode.INVALID_TIMESTAMP, reasons_inf)


if __name__ == "__main__":
    unittest.main()

