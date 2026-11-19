"""Multi-frame end-to-end synthetic scenarios for Milestone M6.

Tests comprehensive scenarios:
- Scenario A: Standard approach, stop, face robot, wait 2s -> event -> look away -> walk away (exit).
- Scenario B: Initial presence at startup (person already inside zone).
- Scenario C: Tracking loss during pending eligibility with reacquisition within 5s grace period.
- Scenario D: Tracking loss timeout (>= 5.0s) terminating visit and creating new visit on return.
- Scenario E: Revocation and recovery within same visit (no duplicate events).
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
    track_id: int = 1,
) -> TemporalObservation:
    return TemporalObservation(
        timestamp=timestamp,
        track_id=track_id,
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


class TestSyntheticScenarios(unittest.TestCase):
    """End-to-end multi-frame scenario validation."""

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
        self.det = PersonDetection(bbox_xyxy=self.box, confidence=0.95, class_id=0)

    def test_scenario_a_standard_approach_contact_and_exit(self) -> None:
        """Scenario A: Approach -> stop inside zone -> face robot -> wait 2s -> ready -> exit."""
        t = 0.0
        fps = 20.0
        dt = 1.0 / fps

        # Phase 1: Person walking towards robot outside interaction zone (0.0s to 1.0s, 20 frames)
        for _ in range(20):
            track = _make_track(track_id=1, bbox=self.box, timestamp=t)
            frame_obs = _make_frame_obs(
                bbox=self.box,
                zone_val=InteractionZoneValue.OUTSIDE,
                timestamp=t,
            )
            temp_obs = _make_temporal_obs(
                movement_val=MovementValue.MOVING,
                approach_val=ApproachValue.APPROACHING,
                timestamp=t,
            )
            result = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
            self.assertEqual(result.state, ReadinessState.NOT_READY)
            self.assertIn(ReasonCode.OUTSIDE_INTERACTION_ZONE, result.reasons)
            self.assertIn(ReasonCode.PERSON_MOVING, result.reasons)
            t += dt

        initial_visit_id = self.engine.visit_manager.active_visit_id
        self.assertIsNotNone(initial_visit_id)
        # Approach condition confirmed
        self.assertTrue(self.engine.visit_manager.approach_or_presence_established)

        # Phase 2: Person enters interaction zone, stops, faces robot (t = 1.0s to 2.95s, 40 frames)
        # Continuous eligibility accumulates from 0.0s up to 1.95s (< 2.0s)
        for _ in range(40):
            track = _make_track(track_id=1, bbox=self.box, timestamp=t)
            frame_obs = _make_frame_obs(
                bbox=self.box,
                zone_val=InteractionZoneValue.INSIDE,
                orient_val=HeadOrientationValue.FACING_ROBOT,
                timestamp=t,
            )
            temp_obs = _make_temporal_obs(
                movement_val=MovementValue.STATIONARY,
                approach_val=ApproachValue.NOT_APPROACHING,
                timestamp=t,
            )
            result = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
            self.assertEqual(result.state, ReadinessState.PENDING)
            self.assertIsNone(result.event)
            self.assertTrue(result.predicates.is_all_valid)
            t += dt

        # Phase 3: At t = 3.0s (elapsed >= 2.0s), transitions to READY and emits event exactly once
        track = _make_track(track_id=1, bbox=self.box, timestamp=t)
        frame_obs = _make_frame_obs(
            bbox=self.box,
            zone_val=InteractionZoneValue.INSIDE,
            orient_val=HeadOrientationValue.FACING_ROBOT,
            timestamp=t,
        )
        temp_obs = _make_temporal_obs(
            movement_val=MovementValue.STATIONARY,
            approach_val=ApproachValue.NOT_APPROACHING,
            timestamp=t,
        )
        result_ready = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
        self.assertEqual(result_ready.state, ReadinessState.READY)
        self.assertIsNotNone(result_ready.event)
        self.assertEqual(result_ready.event.visit_id, initial_visit_id)
        self.assertEqual(result_ready.event.track_id, 1)
        t += dt

        # Phase 4: Person stays in READY for 10 frames, no duplicate events
        for _ in range(10):
            track = _make_track(track_id=1, bbox=self.box, timestamp=t)
            frame_obs = _make_frame_obs(
                bbox=self.box,
                zone_val=InteractionZoneValue.INSIDE,
                orient_val=HeadOrientationValue.FACING_ROBOT,
                timestamp=t,
            )
            temp_obs = _make_temporal_obs(
                movement_val=MovementValue.STATIONARY,
                approach_val=ApproachValue.NOT_APPROACHING,
                timestamp=t,
            )
            res_stay = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
            self.assertEqual(res_stay.state, ReadinessState.READY)
            self.assertIsNone(res_stay.event)
            t += dt

        # Phase 5: Person looks away (t = 4.0s) -> state drops to NOT_READY
        frame_obs_away = _make_frame_obs(
            bbox=self.box,
            zone_val=InteractionZoneValue.INSIDE,
            orient_val=HeadOrientationValue.NOT_FACING_ROBOT,
            timestamp=t,
        )
        res_away = self.engine.evaluate(track, frame_obs_away, temp_obs, self.det, timestamp=t)
        self.assertEqual(res_away.state, ReadinessState.NOT_READY)
        self.assertIn(ReasonCode.NOT_FACING_ROBOT, res_away.reasons)
        t += dt

        # Phase 6: Person steps OUTSIDE (VALID) -> visit terminates immediately
        frame_obs_exit = _make_frame_obs(
            bbox=self.box,
            zone_val=InteractionZoneValue.OUTSIDE,
            zone_stat=ObservationStatus.VALID,
            timestamp=t,
        )
        res_exit = self.engine.evaluate(track, frame_obs_exit, temp_obs, self.det, timestamp=t)
        self.assertEqual(res_exit.state, ReadinessState.NOT_READY)
        self.assertFalse(self.engine.visit_manager.has_active_visit)
        self.assertEqual(self.engine.visit_manager.last_termination_reason, "RELIABLE_ZONE_EXIT")

    def test_scenario_b_initial_presence_at_startup(self) -> None:
        """Scenario B: Person is already inside zone at startup without prior approach."""
        t = 0.0
        dt = 0.1

        # Frame 0: Startup frame with person already inside zone and stationary
        track = _make_track(track_id=5, bbox=self.box, timestamp=t)
        frame_obs = _make_frame_obs(
            bbox=self.box,
            zone_val=InteractionZoneValue.INSIDE,
            orient_val=HeadOrientationValue.FACING_ROBOT,
            timestamp=t,
        )
        temp_obs = _make_temporal_obs(
            movement_val=MovementValue.STATIONARY,
            approach_val=ApproachValue.NOT_APPROACHING,
            timestamp=t,
            track_id=5,
        )
        res0 = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
        self.assertTrue(self.engine.visit_manager.approach_or_presence_established)
        self.assertTrue(res0.predicates.A)
        self.assertEqual(res0.state, ReadinessState.PENDING)

        # Run for 2.0 continuous seconds
        for _ in range(20):
            t += dt
            track = _make_track(track_id=5, bbox=self.box, timestamp=t)
            frame_obs = _make_frame_obs(
                bbox=self.box,
                zone_val=InteractionZoneValue.INSIDE,
                orient_val=HeadOrientationValue.FACING_ROBOT,
                timestamp=t,
            )
            temp_obs = _make_temporal_obs(
                movement_val=MovementValue.STATIONARY,
                approach_val=ApproachValue.NOT_APPROACHING,
                timestamp=t,
                track_id=5,
            )
            res = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)

        self.assertEqual(res.state, ReadinessState.READY)
        self.assertIsNotNone(res.event)
        self.assertEqual(res.event.track_id, 5)

    def test_scenario_c_tracking_loss_and_reacquisition_within_grace_period(self) -> None:
        """Scenario C: Tracking lost during pending accumulation, reacquired within 5.0s."""
        t = 0.0
        dt = 0.1

        # Phase 1: Accumulate 1.0s of continuous eligibility (10 frames at 10 FPS)
        for _ in range(11):
            track = _make_track(track_id=1, bbox=self.box, timestamp=t)
            frame_obs = _make_frame_obs(bbox=self.box, timestamp=t)
            temp_obs = _make_temporal_obs(timestamp=t, track_id=1)
            res = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
            t += dt

        self.assertEqual(res.state, ReadinessState.PENDING)
        self.assertAlmostEqual(res.elapsed_eligibility, 1.0, places=2)
        visit_id = self.engine.visit_manager.active_visit_id
        self.assertIsNotNone(visit_id)

        # Phase 2: Tracking lost for 2.0s (t from 1.1 to 3.0) -> grace period active
        # Advance continuously with dt = 0.1 to avoid inter-frame jumps
        for _ in range(20):
            res_loss = self.engine.evaluate(None, None, None, None, timestamp=t)
            self.assertEqual(res_loss.state, ReadinessState.NOT_READY)
            self.assertTrue(self.engine.visit_manager.is_in_loss_grace_period)
            self.assertEqual(res_loss.elapsed_eligibility, 0.0)
            t += dt

        # Phase 3: Reacquired at t = 3.1s (before 5.0s timeout)
        track_reacq = _make_track(track_id=1, bbox=self.box, timestamp=t)
        frame_obs_reacq = _make_frame_obs(bbox=self.box, timestamp=t)
        temp_obs_reacq = _make_temporal_obs(timestamp=t, track_id=1)
        res_reacq = self.engine.evaluate(track_reacq, frame_obs_reacq, temp_obs_reacq, self.det, timestamp=t)
        # Same visit retained!
        self.assertEqual(self.engine.visit_manager.active_visit_id, visit_id)
        self.assertFalse(self.engine.visit_manager.is_in_loss_grace_period)
        # Continuity was broken, so timer restarted at 0.0s elapsed
        self.assertEqual(res_reacq.state, ReadinessState.PENDING)
        self.assertEqual(res_reacq.elapsed_eligibility, 0.0)

        # Phase 4: Must accumulate full 2.0 continuous seconds from reacquisition (20 frames)
        for _ in range(20):
            t += dt
            track_reacq = _make_track(track_id=1, bbox=self.box, timestamp=t)
            frame_obs_reacq = _make_frame_obs(bbox=self.box, timestamp=t)
            temp_obs_reacq = _make_temporal_obs(timestamp=t, track_id=1)
            res_ready = self.engine.evaluate(track_reacq, frame_obs_reacq, temp_obs_reacq, self.det, timestamp=t)

        self.assertEqual(res_ready.state, ReadinessState.READY)
        self.assertIsNotNone(res_ready.event)
        self.assertEqual(res_ready.event.visit_id, visit_id)

    def test_scenario_d_tracking_loss_timeout_terminates_visit(self) -> None:
        """Scenario D: Tracking loss >= 5.0s terminates visit; subsequent appearance is new visit."""
        t = 0.0
        track = _make_track(track_id=1, bbox=self.box, timestamp=t)
        frame_obs = _make_frame_obs(bbox=self.box, timestamp=t)
        temp_obs = _make_temporal_obs(timestamp=t, track_id=1)

        self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
        visit_id1 = self.engine.visit_manager.active_visit_id

        # Loss starts at t = 1.0s
        self.engine.evaluate(None, None, None, None, timestamp=1.0)
        self.assertTrue(self.engine.visit_manager.has_active_visit)

        # Loss reaches 6.0s (5.0s elapsed) -> visit terminates
        res_term = self.engine.evaluate(None, None, None, None, timestamp=6.0)
        self.assertFalse(self.engine.visit_manager.has_active_visit)
        self.assertEqual(self.engine.visit_manager.last_termination_reason, "TRACKING_LOSS_TIMEOUT")

        # Person reappears at t = 10.0s -> new visit
        track_new = _make_track(track_id=2, bbox=self.box, timestamp=10.0)
        frame_new = _make_frame_obs(bbox=self.box, timestamp=10.0)
        temp_new = _make_temporal_obs(timestamp=10.0, track_id=2)
        det_new = PersonDetection(bbox_xyxy=self.box, confidence=0.91, class_id=0)
        res_new = self.engine.evaluate(track_new, frame_new, temp_new, det_new, timestamp=10.0)

        self.assertTrue(self.engine.visit_manager.has_active_visit)
        self.assertNotEqual(self.engine.visit_manager.active_visit_id, visit_id1)

    def test_scenario_e_revocation_and_recovery_within_same_visit(self) -> None:
        """Scenario E: Revocation and recovery within same visit (no duplicate events)."""
        t = 0.0
        dt = 0.1

        # Phase 1: Accumulate 2.0s -> reach READY and emit event
        for _ in range(21):
            track = _make_track(track_id=1, bbox=self.box, timestamp=t)
            frame_obs = _make_frame_obs(bbox=self.box, timestamp=t)
            temp_obs = _make_temporal_obs(timestamp=t, track_id=1)
            res = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)
            t += dt

        self.assertEqual(res.state, ReadinessState.READY)
        self.assertIsNotNone(res.event)
        visit_id = self.engine.visit_manager.active_visit_id
        self.assertTrue(self.engine.visit_manager.event_emitted)

        # Phase 2: Person temporarily turns head away -> state drops to NOT_READY
        track = _make_track(track_id=1, bbox=self.box, timestamp=t)
        frame_away = _make_frame_obs(
            bbox=self.box,
            orient_val=HeadOrientationValue.NOT_FACING_ROBOT,
            timestamp=t,
        )
        temp_obs = _make_temporal_obs(timestamp=t, track_id=1)
        res_away = self.engine.evaluate(track, frame_away, temp_obs, self.det, timestamp=t)
        self.assertEqual(res_away.state, ReadinessState.NOT_READY)
        self.assertIn(ReasonCode.NOT_FACING_ROBOT, res_away.reasons)
        self.assertIsNone(res_away.event)

        # Phase 3: Person faces robot again for 2.0s -> reaches READY again, but NO DUPLICATE EVENT
        for _ in range(21):
            t += dt
            track = _make_track(track_id=1, bbox=self.box, timestamp=t)
            frame_obs = _make_frame_obs(bbox=self.box, timestamp=t)
            temp_obs = _make_temporal_obs(timestamp=t, track_id=1)
            res_recov = self.engine.evaluate(track, frame_obs, temp_obs, self.det, timestamp=t)

        self.assertEqual(res_recov.state, ReadinessState.READY)
        self.assertIsNone(res_recov.event, "Duplicate event must NOT be emitted for the same visit!")
        self.assertEqual(self.engine.visit_manager.active_visit_id, visit_id)

    def test_scenario_f_active_target_missing_while_other_person_observed(self) -> None:
        """Scenario F: Active target missing while another person is observed.

        Verifies fail-closed target locking:
        1. Another person's OUTSIDE observation does not terminate active person's visit.
        2. Another person's INSIDE observation does not satisfy active person's presence.
        3. Another person does not hijack the active visit.
        """
        t = 0.0
        dt = 0.1

        # Step 1: Active target (track 1) established
        track1 = _make_track(track_id=1, bbox=self.box, timestamp=t)
        frame1 = _make_frame_obs(
            bbox=self.box,
            zone_val=InteractionZoneValue.OUTSIDE,
            timestamp=t,
        )
        temp1 = _make_temporal_obs(
            approach_val=ApproachValue.APPROACHING,
            timestamp=t,
            track_id=1,
        )
        res1 = self.engine.evaluate(track1, frame1, temp1, self.det, timestamp=t)
        visit_id = self.engine.visit_manager.active_visit_id
        self.assertIsNotNone(visit_id)
        self.assertEqual(self.engine.visit_manager.active_track_id, 1)

        # Step 2: Track 1 temporarily lost (grace period starts)
        t += dt
        self.engine.evaluate(None, None, None, None, timestamp=t)
        self.assertTrue(self.engine.visit_manager.is_in_loss_grace_period)

        # Step 3: Different person (track 2) is observed stepping OUTSIDE
        t += dt
        box2 = (700.0, 200.0, 900.0, 600.0)
        det2 = PersonDetection(bbox_xyxy=box2, confidence=0.88, class_id=0)
        frame2 = _make_frame_obs(
            bbox=box2,
            zone_val=InteractionZoneValue.OUTSIDE,
            zone_stat=ObservationStatus.VALID,
            timestamp=t,
        )
        temp2 = _make_temporal_obs(timestamp=t, track_id=2)
        track2 = _make_track(track_id=2, bbox=box2, timestamp=t)

        res2 = self.engine.evaluate(track2, frame2, temp2, det2, timestamp=t)
        # Active target is still track 1, in grace period!
        self.assertTrue(self.engine.visit_manager.has_active_visit)
        self.assertEqual(self.engine.visit_manager.active_track_id, 1)
        self.assertTrue(self.engine.visit_manager.is_in_loss_grace_period)
        self.assertEqual(res2.state, ReadinessState.NOT_READY)
        self.assertIn(ReasonCode.TARGET_MISMATCH, res2.reasons)

        # Step 4: Active person (track 1) reacquires inside zone
        t += dt
        track1_reacq = _make_track(track_id=1, bbox=self.box, timestamp=t)
        frame1_reacq = _make_frame_obs(
            bbox=self.box,
            zone_val=InteractionZoneValue.INSIDE,
            orient_val=HeadOrientationValue.FACING_ROBOT,
            timestamp=t,
        )
        temp1_reacq = _make_temporal_obs(timestamp=t, track_id=1)
        res1_reacq = self.engine.evaluate(track1_reacq, frame1_reacq, temp1_reacq, self.det, timestamp=t)
        self.assertEqual(self.engine.visit_manager.active_visit_id, visit_id)
        self.assertFalse(self.engine.visit_manager.is_in_loss_grace_period)
        self.assertEqual(res1_reacq.state, ReadinessState.PENDING)

    def test_scenario_g_startup_presence_vs_subsequent_visit_presence(self) -> None:
        """Scenario G: Initial presence at startup vs subsequent visit directly inside zone.

        Demonstrates that a person present at initial startup gets approach_or_presence_established,
        and documents the contract for subsequent visits after full termination.
        """
        t = 0.0
        # Person 1 at startup inside zone -> initial presence confirmed
        track1 = _make_track(track_id=1, bbox=self.box, timestamp=t)
        frame1 = _make_frame_obs(bbox=self.box, zone_val=InteractionZoneValue.INSIDE, timestamp=t)
        temp1 = _make_temporal_obs(timestamp=t, track_id=1)
        res1 = self.engine.evaluate(track1, frame1, temp1, self.det, timestamp=t)
        self.assertTrue(self.engine.visit_manager.approach_or_presence_established)
        self.assertTrue(res1.predicates.A)

        # Person 1 exits reliably
        t = 1.0
        frame_exit = _make_frame_obs(
            bbox=self.box,
            zone_val=InteractionZoneValue.OUTSIDE,
            zone_stat=ObservationStatus.VALID,
            timestamp=t,
        )
        self.engine.evaluate(track1, frame_exit, temp1, self.det, timestamp=t)
        self.assertFalse(self.engine.visit_manager.has_active_visit)

        # Person 2 appears at t = 5.0 directly inside zone (second visit of system)
        t = 5.0
        track2 = _make_track(track_id=2, bbox=self.box, timestamp=t)
        frame2 = _make_frame_obs(bbox=self.box, zone_val=InteractionZoneValue.INSIDE, timestamp=t)
        temp2 = _make_temporal_obs(timestamp=t, track_id=2)
        det2 = PersonDetection(bbox_xyxy=self.box, confidence=0.9, class_id=0)
        res2 = self.engine.evaluate(track2, frame2, temp2, det2, timestamp=t)

        self.assertTrue(self.engine.visit_manager.has_active_visit)
        self.assertTrue(self.engine.visit_manager.approach_or_presence_established)
        self.assertTrue(res2.predicates.A)


if __name__ == "__main__":
    unittest.main()
