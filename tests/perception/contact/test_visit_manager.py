"""Unit tests for VisitManager (Milestone M6).

Tests target locking, 5.0s tracking loss grace period, reacquisition,
reliable zone exit termination, and approach/initial presence persistence.
"""

from __future__ import annotations

import unittest

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
from perception.temporal.observation_types import (
    ApproachValue,
    InitializationState,
    MotionMetrics,
    MovementValue,
    StabilityObservation,
    StabilityValue,
    TemporalObservation,
)
from perception.detection.person_detector import PersonDetection
from perception.tracking.single_person_tracker import Track, TrackState


def _make_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 300.0, 500.0),
    observed: bool = True,
    timestamp: float = 1.0,
) -> Track:
    """Helper to construct Track with default values."""
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


def _make_dummy_frame_obs(
    presence_val: PersonPresenceValue = PersonPresenceValue.DETECTED,
    presence_status: ObservationStatus = ObservationStatus.VALID,
    zone_val: InteractionZoneValue = InteractionZoneValue.INSIDE,
    zone_status: ObservationStatus = ObservationStatus.VALID,
    timestamp: float = 1.0,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 300.0, 500.0),
) -> FrameObservation:
    """Helper to construct dummy FrameObservation."""
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
            presence=Observation(value=presence_val, confidence=0.9, status=presence_status),
            bounding_box=Observation(value=BboxValidityValue.VALID, confidence=None, status=ObservationStatus.VALID),
            bbox_metrics=metrics,
            interaction_zone=Observation(value=zone_val, confidence=None, status=zone_status),
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


def _make_dummy_temporal_obs(
    approach_val: ApproachValue = ApproachValue.APPROACHING,
    approach_status: ObservationStatus = ObservationStatus.VALID,
    timestamp: float = 1.0,
    track_id: int = 1,
) -> TemporalObservation:
    """Helper to construct dummy TemporalObservation."""
    return TemporalObservation(
        timestamp=timestamp,
        track_id=track_id,
        observed=True,
        approach=Observation(value=approach_val, confidence=None, status=approach_status),
        movement=Observation(value=MovementValue.STATIONARY, confidence=None, status=ObservationStatus.VALID),
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


class TestVisitManager(unittest.TestCase):
    """Test suite for VisitManager."""

    def test_initial_state(self) -> None:
        """Verify newly created visit manager has no active visit."""
        vm = VisitManager(tracking_loss_timeout_seconds=5.0)
        self.assertFalse(vm.has_active_visit)
        self.assertIsNone(vm.active_visit_id)
        self.assertIsNone(vm.active_track_id)
        self.assertFalse(vm.approach_or_presence_established)
        self.assertFalse(vm.event_emitted)
        self.assertFalse(vm.is_in_loss_grace_period)
        self.assertEqual(vm.tracking_loss_duration, 0.0)
        self.assertIsNone(vm.last_termination_reason)

    def test_invalid_timeout_raises(self) -> None:
        """Verify non-positive tracking_loss_timeout_seconds raises ValueError."""
        with self.assertRaises(ValueError):
            VisitManager(tracking_loss_timeout_seconds=0.0)
        with self.assertRaises(ValueError):
            VisitManager(tracking_loss_timeout_seconds=-1.0)

    def test_new_visit_initiated_on_observed_track(self) -> None:
        """Verify new visit is created and target locked when observed track appears."""
        vm = VisitManager(tracking_loss_timeout_seconds=5.0)
        track = _make_track(track_id=42, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs = _make_dummy_frame_obs(zone_val=InteractionZoneValue.OUTSIDE)

        visit_id, is_active = vm.update(track, frame_obs, None, timestamp=1.0)

        self.assertTrue(vm.has_active_visit)
        self.assertIsNotNone(visit_id)
        self.assertEqual(vm.active_visit_id, visit_id)
        self.assertEqual(vm.active_track_id, 42)
        self.assertTrue(is_active)
        self.assertFalse(vm.approach_or_presence_established)

    def test_no_visit_when_track_not_observed(self) -> None:
        """Verify no visit is initiated if track is unobserved or None."""
        vm = VisitManager()
        frame_obs = _make_dummy_frame_obs()

        # Track is None
        v1, a1 = vm.update(None, frame_obs, None, timestamp=1.0)
        self.assertIsNone(v1)
        self.assertFalse(a1)
        self.assertFalse(vm.has_active_visit)

        # Track is unobserved (coasting)
        track = _make_track(track_id=1, bbox=(10.0, 10.0, 50.0, 50.0), observed=False)
        v2, a2 = vm.update(track, frame_obs, None, timestamp=2.0)
        self.assertIsNone(v2)
        self.assertFalse(a2)
        self.assertFalse(vm.has_active_visit)

    def test_initial_presence_path(self) -> None:
        """Verify startup / arrival directly in interaction zone sets approach_or_presence."""
        vm = VisitManager()
        box = (100.0, 100.0, 300.0, 500.0)
        track = _make_track(track_id=10, bbox=box, observed=True)
        det = PersonDetection(bbox_xyxy=box, confidence=0.92, class_id=0)
        frame_obs = _make_dummy_frame_obs(
            bbox=box,
            presence_val=PersonPresenceValue.DETECTED,
            presence_status=ObservationStatus.VALID,
            zone_val=InteractionZoneValue.INSIDE,
            zone_status=ObservationStatus.VALID,
        )

        vm.update(track, frame_obs, None, timestamp=1.0, matched_detection=det)
        self.assertTrue(vm.approach_or_presence_established)

    def test_confirmed_approach_path(self) -> None:
        """Verify M5 confirmed approach sets approach_or_presence_established."""
        vm = VisitManager()
        box = (100.0, 100.0, 300.0, 500.0)
        track = _make_track(track_id=10, bbox=box, observed=True)
        det = PersonDetection(bbox_xyxy=box, confidence=0.92, class_id=0)
        frame_obs = _make_dummy_frame_obs(bbox=box, zone_val=InteractionZoneValue.OUTSIDE)
        temp_obs = _make_dummy_temporal_obs(
            approach_val=ApproachValue.APPROACHING,
            approach_status=ObservationStatus.VALID,
            track_id=10,
        )

        vm.update(track, frame_obs, temp_obs, timestamp=1.0, matched_detection=det)
        self.assertTrue(vm.approach_or_presence_established)

    def test_approach_persistence_across_state_changes(self) -> None:
        """Verify once approach/presence is established, it persists for the visit."""
        vm = VisitManager()
        box = (100.0, 100.0, 300.0, 500.0)
        track = _make_track(track_id=10, bbox=box, observed=True)
        det = PersonDetection(bbox_xyxy=box, confidence=0.92, class_id=0)
        frame_obs = _make_dummy_frame_obs(bbox=box, zone_val=InteractionZoneValue.INSIDE)

        vm.update(track, frame_obs, None, timestamp=1.0, matched_detection=det)
        self.assertTrue(vm.approach_or_presence_established)

        # Later frame: temporal approach says UNKNOWN / NOT_APPROACHING
        temp_obs_stationary = _make_dummy_temporal_obs(
            approach_val=ApproachValue.UNKNOWN,
            approach_status=ObservationStatus.UNCERTAIN,
            track_id=10,
        )
        vm.update(track, frame_obs, temp_obs_stationary, timestamp=1.5, matched_detection=det)
        self.assertTrue(vm.approach_or_presence_established)

    def test_target_locking(self) -> None:
        """Verify locked target cannot be hijacked by another track ID."""
        vm = VisitManager()
        track1 = _make_track(track_id=10, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs = _make_dummy_frame_obs()

        v1, _ = vm.update(track1, frame_obs, None, timestamp=1.0)
        self.assertEqual(vm.active_track_id, 10)

        # Second track appears instead of track 10
        track2 = _make_track(track_id=99, bbox=(50.0, 50.0, 150.0, 200.0), observed=True)
        v2, is_active2 = vm.update(track2, frame_obs, None, timestamp=1.1)

        # Visit is still active with locked track 10, but target 10 is not currently active
        self.assertEqual(v2, v1)
        self.assertFalse(is_active2)
        self.assertTrue(vm.is_in_loss_grace_period)

    def test_tracking_loss_grace_period_and_reacquisition(self) -> None:
        """Verify 5.0s grace period preserves visit and reacquisition restores tracking."""
        vm = VisitManager(tracking_loss_timeout_seconds=5.0)
        track = _make_track(track_id=7, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs = _make_dummy_frame_obs()

        v_init, _ = vm.update(track, frame_obs, None, timestamp=10.0)
        self.assertFalse(vm.is_in_loss_grace_period)

        # Tracking loss at t = 11.0 (elapsed = 0.0s)
        v_loss1, is_act1 = vm.update(None, frame_obs, None, timestamp=11.0)
        self.assertEqual(v_loss1, v_init)
        self.assertFalse(is_act1)
        self.assertTrue(vm.is_in_loss_grace_period)
        self.assertAlmostEqual(vm.tracking_loss_duration, 0.0)

        # Tracking loss continues at t = 13.5 (elapsed = 2.5s < 5.0s)
        v_loss2, is_act2 = vm.update(None, frame_obs, None, timestamp=13.5)
        self.assertEqual(v_loss2, v_init)
        self.assertFalse(is_act2)
        self.assertTrue(vm.is_in_loss_grace_period)
        self.assertAlmostEqual(vm.tracking_loss_duration, 2.5)

        # Reacquisition at t = 14.0 (before 5.0s timeout)
        v_reacq, is_act3 = vm.update(track, frame_obs, None, timestamp=14.0)
        self.assertEqual(v_reacq, v_init)
        self.assertTrue(is_act3)
        self.assertFalse(vm.is_in_loss_grace_period)
        self.assertEqual(vm.tracking_loss_duration, 0.0)

    def test_tracking_loss_timeout_terminates_visit(self) -> None:
        """Verify tracking loss >= 5.0s terminates visit."""
        vm = VisitManager(tracking_loss_timeout_seconds=5.0)
        track = _make_track(track_id=7, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs = _make_dummy_frame_obs()

        vm.update(track, frame_obs, None, timestamp=10.0)

        # Loss starts at t = 11.0
        vm.update(None, frame_obs, None, timestamp=11.0)
        self.assertTrue(vm.has_active_visit)

        # Loss at t = 16.0 (elapsed = 5.0s >= 5.0s)
        v_term, is_act = vm.update(None, frame_obs, None, timestamp=16.0)
        self.assertIsNone(v_term)
        self.assertFalse(is_act)
        self.assertFalse(vm.has_active_visit)
        self.assertEqual(vm.last_termination_reason, "TRACKING_LOSS_TIMEOUT")

    def test_reliable_zone_exit_terminates_immediately(self) -> None:
        """Verify OUTSIDE + VALID interaction zone terminates visit immediately."""
        vm = VisitManager()
        box = (100.0, 100.0, 300.0, 500.0)
        track = _make_track(track_id=7, bbox=box, observed=True)
        det = PersonDetection(bbox_xyxy=box, confidence=0.92, class_id=0)
        frame_obs_inside = _make_dummy_frame_obs(bbox=box, zone_val=InteractionZoneValue.INSIDE)

        vm.update(track, frame_obs_inside, None, timestamp=1.0, matched_detection=det)
        self.assertTrue(vm.has_active_visit)

        # Person steps OUTSIDE (VALID)
        frame_obs_outside = _make_dummy_frame_obs(
            bbox=box,
            zone_val=InteractionZoneValue.OUTSIDE,
            zone_status=ObservationStatus.VALID,
        )
        v_out, is_act = vm.update(track, frame_obs_outside, None, timestamp=1.1, matched_detection=det)

        self.assertIsNone(v_out)
        self.assertFalse(is_act)
        self.assertFalse(vm.has_active_visit)
        self.assertEqual(vm.last_termination_reason, "RELIABLE_ZONE_EXIT")

    def test_zone_exit_without_valid_status_does_not_terminate(self) -> None:
        """Verify OUTSIDE with UNCERTAIN/INVALID status does not terminate visit."""
        vm = VisitManager()
        box = (100.0, 100.0, 300.0, 500.0)
        track = _make_track(track_id=7, bbox=box, observed=True)
        det = PersonDetection(bbox_xyxy=box, confidence=0.92, class_id=0)
        frame_obs_inside = _make_dummy_frame_obs(bbox=box, zone_val=InteractionZoneValue.INSIDE)

        vm.update(track, frame_obs_inside, None, timestamp=1.0, matched_detection=det)

        # UNCERTAIN outside
        frame_obs_uncertain = _make_dummy_frame_obs(
            bbox=box,
            zone_val=InteractionZoneValue.OUTSIDE,
            zone_status=ObservationStatus.UNCERTAIN,
        )
        v_unc, is_act = vm.update(track, frame_obs_uncertain, None, timestamp=1.1, matched_detection=det)

        self.assertIsNotNone(v_unc)
        self.assertTrue(is_act)
        self.assertTrue(vm.has_active_visit)

    def test_missing_matched_detection_does_not_establish_initial_presence(self) -> None:
        """Verify missing matched_detection fails closed: initial presence is NOT established."""
        vm = VisitManager()
        track = _make_track(track_id=10, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs = _make_dummy_frame_obs(
            presence_val=PersonPresenceValue.DETECTED,
            presence_status=ObservationStatus.VALID,
            zone_val=InteractionZoneValue.INSIDE,
            zone_status=ObservationStatus.VALID,
        )
        vm.update(track, frame_obs, None, timestamp=1.0, matched_detection=None)
        self.assertFalse(vm.approach_or_presence_established)

    def test_missing_matched_detection_does_not_terminate_on_outside(self) -> None:
        """Verify missing matched_detection fails closed: cannot terminate active visit."""
        vm = VisitManager()
        box = (100.0, 100.0, 300.0, 500.0)
        track = _make_track(track_id=7, bbox=box, observed=True)
        det = PersonDetection(bbox_xyxy=box, confidence=0.92, class_id=0)
        frame_obs_inside = _make_dummy_frame_obs(bbox=box, zone_val=InteractionZoneValue.INSIDE)
        vm.update(track, frame_obs_inside, None, timestamp=1.0, matched_detection=det)
        self.assertTrue(vm.has_active_visit)

        frame_obs_outside = _make_dummy_frame_obs(
            bbox=box,
            zone_val=InteractionZoneValue.OUTSIDE,
            zone_status=ObservationStatus.VALID,
        )
        v_out, _ = vm.update(track, frame_obs_outside, None, timestamp=1.1, matched_detection=None)
        # Should NOT terminate because identity could not be established without matched_detection!
        self.assertIsNotNone(v_out)
        self.assertTrue(vm.has_active_visit)

    def test_missing_temporal_track_id_does_not_confirm_approach(self) -> None:
        """Verify temporal observation with track_id=None does not confirm approach."""
        vm = VisitManager()
        box = (100.0, 100.0, 300.0, 500.0)
        track = _make_track(track_id=10, bbox=box, observed=True)
        det = PersonDetection(bbox_xyxy=box, confidence=0.92, class_id=0)
        frame_obs = _make_dummy_frame_obs(bbox=box, zone_val=InteractionZoneValue.OUTSIDE)
        temp_obs_no_id = _make_dummy_temporal_obs(
            approach_val=ApproachValue.APPROACHING,
            approach_status=ObservationStatus.VALID,
            track_id=None,
        )
        vm.update(track, frame_obs, temp_obs_no_id, timestamp=1.0, matched_detection=det)
        self.assertFalse(vm.approach_or_presence_established)

    def test_event_emitted_tracking(self) -> None:
        """Verify event_emitted flag can be marked and is preserved."""
        vm = VisitManager()
        track = _make_track(track_id=1, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs = _make_dummy_frame_obs()

        vm.update(track, frame_obs, None, timestamp=1.0)
        self.assertFalse(vm.event_emitted)

        vm.mark_event_emitted()
        self.assertTrue(vm.event_emitted)

        # Remains marked during tracking loss grace period
        vm.update(None, frame_obs, None, timestamp=2.0)
        self.assertTrue(vm.event_emitted)

        # Cleared on termination
        vm.terminate_visit("MANUAL")
        self.assertFalse(vm.event_emitted)


    def test_other_person_zone_exit_does_not_terminate_active_visit(self) -> None:
        """Verify another person stepping OUTSIDE does not terminate active person's visit."""
        vm = VisitManager()
        track_active = _make_track(track_id=1, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs_inside = _make_dummy_frame_obs(zone_val=InteractionZoneValue.INSIDE)

        v_init, _ = vm.update(track_active, frame_obs_inside, None, timestamp=1.0)
        self.assertTrue(vm.has_active_visit)

        # In the next frame, active track 1 is missing, but a second person (track 2) is OUTSIDE
        track_other = _make_track(track_id=2, bbox=(500.0, 500.0, 600.0, 700.0), observed=True)
        frame_obs_other_outside = _make_dummy_frame_obs(
            zone_val=InteractionZoneValue.OUTSIDE,
            zone_status=ObservationStatus.VALID,
        )
        # Bbox of m4 frame observation does not match track 1
        v_curr, is_act = vm.update(track_other, frame_obs_other_outside, None, timestamp=1.1)

        # Active visit MUST NOT be terminated by person 2's exit!
        self.assertEqual(v_curr, v_init)
        self.assertFalse(is_act)
        self.assertTrue(vm.has_active_visit)
        self.assertTrue(vm.is_in_loss_grace_period)

    def test_other_person_m4_observation_does_not_establish_presence_for_active_visit(self) -> None:
        """Verify another person's inside observation does not establish approach/presence for active target."""
        vm = VisitManager()
        track_active = _make_track(track_id=1, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs_outside = _make_dummy_frame_obs(zone_val=InteractionZoneValue.OUTSIDE)

        vm.update(track_active, frame_obs_outside, None, timestamp=1.0)
        self.assertFalse(vm.approach_or_presence_established)

        # Frame 2: active target is unobserved, but another person (track 2) is observed inside
        track_other = _make_track(track_id=2, bbox=(200.0, 200.0, 400.0, 600.0), observed=True)
        frame_obs_other_inside = _make_dummy_frame_obs(zone_val=InteractionZoneValue.INSIDE)

        vm.update(track_other, frame_obs_other_inside, None, timestamp=1.1)
        self.assertFalse(vm.approach_or_presence_established)

    def test_m3_track_id_and_m5_temporal_identity_mismatch_rejected(self) -> None:
        """Verify temporal observation with mismatched track ID cannot satisfy approach."""
        vm = VisitManager()
        track_active = _make_track(track_id=1, bbox=(100.0, 100.0, 300.0, 500.0), observed=True)
        frame_obs_outside = _make_dummy_frame_obs(zone_val=InteractionZoneValue.OUTSIDE)

        # Temporal observation belongs to track 999, NOT track 1
        temp_obs_mismatched = _make_dummy_temporal_obs(
            approach_val=ApproachValue.APPROACHING,
            approach_status=ObservationStatus.VALID,
            track_id=999,
        )

        vm.update(track_active, frame_obs_outside, temp_obs_mismatched, timestamp=1.0)
        self.assertFalse(vm.approach_or_presence_established)


if __name__ == "__main__":
    unittest.main()
