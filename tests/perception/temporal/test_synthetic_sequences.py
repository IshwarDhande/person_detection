"""Synthetic deterministic sequence tests for Milestone M5 Temporal Behaviour Estimation.

Validates the full M3-M4-M5 temporal observation pipeline on deterministic,
mathematically controlled trajectories according to Milestone M5 specification Section 10.3.
"""

import unittest

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
    MovementValue,
    StabilityValue,
)
from perception.temporal.temporal_config import TemporalConfig
from perception.temporal.temporal_estimator import TemporalEstimator
from perception.tracking.single_person_tracker import Track, TrackState


def _build_frame(
    cx: float,
    cy: float,
    timestamp: float,
    w: float = 100.0,
    h: float = 200.0,
    in_zone: bool = True,
    person_valid: bool = True,
) -> tuple[Track, FrameObservation, PersonDetection]:
    """Helper to generate coordinated M3 Track, M4 FrameObservation, and PersonDetection."""
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    box = (x1, y1, x2, y2)

    det = PersonDetection(
        bbox_xyxy=box,
        confidence=0.92,
        class_id=0,
    )
    track = Track(
        track_id=1,
        bbox_xyxy=box,
        center=(cx, cy),
        velocity=(0.0, 0.0, 0.0, 0.0),
        timestamp=timestamp,
        creation_timestamp=0.0,
        last_seen_timestamp=timestamp,
        hit_count=10,
        miss_count=0,
        state=TrackState.CONFIRMED,
        observed=True,
    )

    metrics = BboxMetrics(
        original_bbox_xyxy=box,
        clipped_bbox_xyxy=box,
        width=w,
        height=h,
        area=w * h,
        area_ratio=(w * h) / (1280 * 720),
        image_width=1280,
        image_height=720,
    )

    p_val = PersonPresenceValue.DETECTED if person_valid else PersonPresenceValue.UNKNOWN
    p_stat = ObservationStatus.VALID if person_valid else ObservationStatus.INVALID
    z_val = InteractionZoneValue.INSIDE if in_zone else InteractionZoneValue.OUTSIDE

    f_obs = FrameObservation(
        person=PersonObservations(
            presence=Observation(value=p_val, confidence=0.92, status=p_stat),
            bounding_box=Observation(value=BboxValidityValue.VALID, confidence=None, status=ObservationStatus.VALID),
            bbox_metrics=metrics,
            interaction_zone=Observation(value=z_val, confidence=None, status=ObservationStatus.VALID),
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

    return track, f_obs, det


class TestSyntheticSequences(unittest.TestCase):
    """Synthetic sequence tests for temporal behaviour estimation."""

    def setUp(self) -> None:
        # Zone is central region [300, 200] to [900, 600]. Centroid is at (600, 400).
        # In normalized coords: (300/1280, 200/720) = (0.234375, 0.277778)
        self.zone_norm = [
            (300.0 / 1280.0, 200.0 / 720.0),
            (900.0 / 1280.0, 200.0 / 720.0),
            (900.0 / 1280.0, 600.0 / 720.0),
            (300.0 / 1280.0, 600.0 / 720.0),
        ]
        self.estimator = TemporalEstimator(zone_vertices_normalized=self.zone_norm)

    def test_sequence_approach_stop_and_stabilize(self) -> None:
        """Full end-to-end sequence:
        1. Approach from (200, 400) towards (600, 400) at 100 px/s for 1.0s -> APPROACHING.
        2. Slow down and stop at (600, 400) inside zone -> STATIONARY.
        3. Remain stationary for 2.1s -> stability timer ACCUMULATING -> COMPLETE.
        """
        dt = 0.05  # 20 FPS
        t = 1.0

        # Phase 1: Moving towards centroid at 100 px/s (outside zone initially)
        # 20 frames = 1.0s
        for i in range(20):
            cx = 200.0 + 100.0 * (i * dt)  # reaches 295.0 px (outside zone)
            cy = 400.0
            track, f_obs, det = _build_frame(cx, cy, t, in_zone=False)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        # By frame 20, approach has persisted > 0.5s -> APPROACHING
        self.assertEqual(obs.approach.value, ApproachValue.APPROACHING)
        self.assertEqual(obs.approach.status, ObservationStatus.VALID)
        self.assertEqual(obs.movement.value, MovementValue.MOVING)
        self.assertEqual(obs.stability.value, StabilityValue.NOT_STARTED)

        # Phase 2: Enter zone and remain stationary at (600, 400)
        # Settle for 25 frames (1.25s) so the earlier movement completely exits the 1.0s history window
        for i in range(25):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        # After history window is filled purely with stationary points at (600, 400),
        # displacement D_t < 20px and speed < 20 px/s -> STATIONARY [VALID]
        self.assertEqual(obs.movement.value, MovementValue.STATIONARY)
        self.assertEqual(obs.movement.status, ObservationStatus.VALID)
        self.assertEqual(obs.approach.value, ApproachValue.NOT_APPROACHING)

        # Phase 3: Remain stationary in zone for 2.1 seconds (42 frames)
        for i in range(42):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        # Stability timer must reach COMPLETE
        self.assertEqual(obs.stability.value, StabilityValue.COMPLETE)
        self.assertEqual(obs.stability.status, ObservationStatus.VALID)
        self.assertGreaterEqual(obs.stability.elapsed_seconds, 2.0)
        self.assertEqual(obs.stability.violation_seconds, 0.0)

    def test_sequence_moving_away(self) -> None:
        """Sequence: Person moves away from zone centroid:
        cx moves from 700 to 1000 at 100 px/s -> NOT_APPROACHING, MOVING.
        """
        dt = 0.05
        t = 10.0
        for i in range(25):
            cx = 700.0 + 100.0 * (i * dt)
            track, f_obs, det = _build_frame(cx, 400.0, t, in_zone=False)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        self.assertEqual(obs.approach.value, ApproachValue.NOT_APPROACHING)
        self.assertEqual(obs.approach.status, ObservationStatus.VALID)
        self.assertEqual(obs.movement.value, MovementValue.MOVING)
        self.assertEqual(obs.stability.value, StabilityValue.NOT_STARTED)

    def test_sequence_lateral_movement(self) -> None:
        """Sequence: Person moves laterally (pure y movement) past the robot."""
        dt = 0.05
        t = 20.0
        for i in range(25):
            cy = 100.0 + 80.0 * (i * dt)
            track, f_obs, det = _build_frame(200.0, cy, t, in_zone=False)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        self.assertEqual(obs.approach.value, ApproachValue.NOT_APPROACHING)
        self.assertEqual(obs.movement.value, MovementValue.MOVING)

    def test_sequence_intermittent_jitter_within_tolerance(self) -> None:
        """Sequence: Person stationary in zone, but 2 frames (0.10s) experience brief
        jitter causing speed to spike, then settles back.
        Total violation = 0.10s <= 0.20s budget -> stability achieves COMPLETE.
        """
        dt = 0.05
        t = 30.0

        # Stabilize for 1.0s so stationary is established
        for i in range(21):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        # Accumulating stability for 0.8s
        for i in range(16):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt
        self.assertEqual(obs.stability.value, StabilityValue.ACCUMULATING)

        # 2 frames jitter: cx jumps by 25px (speed spike -> MOVING), but still in zone
        for i in range(2):
            track, f_obs, det = _build_frame(625.0, 400.0, t, in_zone=True)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt
        # Still ACCUMULATING during active excursion (ongoing violation = 0.05s <= 0.20s)
        self.assertEqual(obs.stability.value, StabilityValue.ACCUMULATING)
        self.assertAlmostEqual(obs.stability.violation_seconds, 0.05, places=2)

        # Settle back for remaining 1.5s
        # Frame 0 of settle back (recovery): full excursion interval charged (0.10s)
        track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
        obs_rec = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
        t += dt
        self.assertEqual(obs_rec.stability.value, StabilityValue.ACCUMULATING)
        self.assertAlmostEqual(obs_rec.stability.violation_seconds, 0.10, places=2)

        for i in range(29):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        self.assertEqual(obs.stability.value, StabilityValue.COMPLETE)

    def test_sequence_excessive_jitter_exhausts_tolerance(self) -> None:
        """Sequence: Jitter lasts for 6 frames (0.25s > 0.20s) -> exhausts budget and resets."""
        dt = 0.05
        t = 40.0

        for i in range(21):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        for i in range(16):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        # 6 frames of violation (span 0.25s > 0.20s)
        for i in range(6):
            track, f_obs, det = _build_frame(630.0, 400.0, t, in_zone=True)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt

        # Exceeded budget -> RESET
        self.assertEqual(obs.stability.value, StabilityValue.RESET)

    def test_sequence_tracking_loss_and_reacquisition(self) -> None:
        """Sequence: Tracking loss mid-attempt triggers immediate reset; reacquisition starts fresh."""
        dt = 0.05
        t = 50.0

        # Accumulate 1.5s
        for i in range(30):
            track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
            obs = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
            t += dt
        self.assertEqual(obs.stability.value, StabilityValue.ACCUMULATING)

        # Tracking loss: track becomes None
        track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
        obs_lost = self.estimator.estimate(None, f_obs, matched_detection=None, timestamp=t)
        self.assertEqual(obs_lost.stability.value, StabilityValue.RESET)
        self.assertEqual(obs_lost.movement.value, MovementValue.UNKNOWN)
        self.assertEqual(obs_lost.movement.status, ObservationStatus.INVALID)
        self.assertFalse(obs_lost.observed)
        t += dt

        # Reacquisition with fresh observations -> COLLECTING count starts from 1
        track, f_obs, det = _build_frame(600.0, 400.0, t, in_zone=True)
        obs_reacq = self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
        self.assertEqual(obs_reacq.initialization_state, InitializationState.COLLECTING)
        self.assertEqual(obs_reacq.initialization_count, 1)


if __name__ == "__main__":
    unittest.main()
