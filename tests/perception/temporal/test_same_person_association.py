"""Integration tests for M3-M4 same-person association, tracking lifecycle, and stability reset semantics."""

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
    MovementValue,
    StabilityValue,
)
from perception.temporal.temporal_estimator import TemporalEstimator
from perception.tracking.single_person_tracker import Track, TrackState


def _make_dummy_frame_observation(
    bbox_xyxy: tuple[float, float, float, float],
    timestamp: float = 10.0,
    w_img: int = 1280,
    h_img: int = 720,
    in_zone: bool = True,
    presence_value: PersonPresenceValue = PersonPresenceValue.DETECTED,
    presence_status: ObservationStatus = ObservationStatus.VALID,
    bbox_value: BboxValidityValue = BboxValidityValue.VALID,
    bbox_status: ObservationStatus = ObservationStatus.VALID,
    zone_value: InteractionZoneValue = InteractionZoneValue.INSIDE,
    zone_status: ObservationStatus = ObservationStatus.VALID,
) -> FrameObservation:
    """Helper to construct a synthetic FrameObservation with explicit observation values and statuses."""
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    area = w * h
    metrics = BboxMetrics(
        original_bbox_xyxy=bbox_xyxy,
        clipped_bbox_xyxy=bbox_xyxy,
        width=w,
        height=h,
        area=area,
        area_ratio=area / (w_img * h_img),
        image_width=w_img,
        image_height=h_img,
    )
    if zone_value == InteractionZoneValue.INSIDE and not in_zone:
        zone_value = InteractionZoneValue.OUTSIDE

    return FrameObservation(
        person=PersonObservations(
            presence=Observation(
                value=presence_value,
                confidence=0.92,
                status=presence_status,
            ),
            bounding_box=Observation(
                value=bbox_value,
                confidence=None,
                status=bbox_status,
            ),
            bbox_metrics=metrics,
            interaction_zone=Observation(
                value=zone_value,
                confidence=None,
                status=zone_status,
            ),
        ),
        face=FaceObservations(
            visibility=Observation(
                value=FaceVisibilityValue.SUFFICIENT,
                confidence=0.9,
                status=ObservationStatus.VALID,
            ),
            metrics=None,
        ),
        orientation=OrientationObservations(
            head_orientation=Observation(
                value=HeadOrientationValue.FACING_ROBOT,
                confidence=None,
                status=ObservationStatus.VALID,
            ),
            metrics=None,
        ),
        frame_metadata=FrameMetadata(
            timestamp=timestamp,
            image_width=w_img,
            image_height=h_img,
        ),
    )


class TestSamePersonAssociation(unittest.TestCase):
    """Test same-person association verification between M3 track and M4 frame observations."""

    def setUp(self) -> None:
        self.zone_vertices = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        self.estimator = TemporalEstimator(zone_vertices_normalized=self.zone_vertices)

    def test_same_detection_matches_successfully(self) -> None:
        """Verify when M3 and M4 observe the same detection, association succeeds."""
        box = (300.0, 200.0, 500.0, 600.0)
        det = PersonDetection(
            bbox_xyxy=box,
            confidence=0.92,
            class_id=0,
        )
        track = Track(
            track_id=1,
            bbox_xyxy=box,
            center=(400.0, 400.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=10.0,
            last_seen_timestamp=10.0,
            hit_count=1,
            miss_count=0,
            state=TrackState.TENTATIVE,
            observed=True,
        )
        frame_obs = _make_dummy_frame_observation(bbox_xyxy=box, timestamp=10.0)

        temp_obs = self.estimator.estimate(
            track=track,
            frame_observation=frame_obs,
            matched_detection=det,
            timestamp=10.0,
        )

        self.assertEqual(temp_obs.track_id, 1)
        self.assertTrue(temp_obs.observed)
        self.assertEqual(self.estimator.history.count, 1)
        self.assertEqual(self.estimator.history.consecutive_valid_count, 1)

    def test_missing_matched_detection_rejected_even_with_high_iou(self) -> None:
        """Correction 3: When matched_detection is None, association fails even if track overlaps M4 box."""
        box = (300.0, 200.0, 500.0, 600.0)
        track = Track(
            track_id=1,
            bbox_xyxy=box,
            center=(400.0, 400.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=10.0,
            last_seen_timestamp=10.0,
            hit_count=5,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )
        frame_obs = _make_dummy_frame_observation(bbox_xyxy=box, timestamp=10.0)

        # matched_detection is None (e.g. tracker coasted or did not provide matched detection)
        temp_obs = self.estimator.estimate(
            track=track,
            frame_observation=frame_obs,
            matched_detection=None,
            timestamp=10.0,
        )

        self.assertEqual(temp_obs.movement.value, MovementValue.UNKNOWN)
        self.assertEqual(temp_obs.movement.status, ObservationStatus.INVALID)
        self.assertEqual(self.estimator.history.count, 0)

    def test_overlapping_people_distinct_detections_rejected(self) -> None:
        """Correction 3: High IoU overlap between two people does not falsely associate."""
        # Person A (M4 selected): (300, 200, 500, 600)
        box_a = (300.0, 200.0, 500.0, 600.0)
        # Person B (M3 matched): (320, 210, 510, 610) -> IoU > 0.80
        box_b = (320.0, 210.0, 510.0, 610.0)

        det_b = PersonDetection(bbox_xyxy=box_b, confidence=0.88, class_id=0)
        track = Track(
            track_id=1,
            bbox_xyxy=box_b,
            center=(415.0, 410.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=10.0,
            last_seen_timestamp=10.0,
            hit_count=3,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )
        frame_obs = _make_dummy_frame_observation(bbox_xyxy=box_a, timestamp=10.0)

        temp_obs = self.estimator.estimate(
            track=track,
            frame_observation=frame_obs,
            matched_detection=det_b,
            timestamp=10.0,
        )

        # Boxes differ: association rejected
        self.assertEqual(temp_obs.movement.value, MovementValue.UNKNOWN)
        self.assertEqual(temp_obs.movement.status, ObservationStatus.INVALID)
        self.assertEqual(self.estimator.history.count, 0)

    def test_mismatched_detection_rejected(self) -> None:
        """Verify when M4 observed Person A but M3 matched Person B, observation is rejected."""
        box_a = (100.0, 100.0, 200.0, 300.0)  # Selected by M4 Option A (conf=0.95)
        box_b = (500.0, 400.0, 650.0, 700.0)  # Tracked by M3 (conf=0.85)

        det_b = PersonDetection(bbox_xyxy=box_b, confidence=0.85, class_id=0)
        track = Track(
            track_id=1,
            bbox_xyxy=box_b,
            center=(575.0, 550.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=10.0,
            last_seen_timestamp=10.0,
            hit_count=3,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )
        frame_obs = _make_dummy_frame_observation(bbox_xyxy=box_a, timestamp=10.0)

        temp_obs = self.estimator.estimate(
            track=track,
            frame_observation=frame_obs,
            matched_detection=det_b,
            timestamp=10.0,
        )

        # Mismatch must reject observation and NOT enter history
        self.assertEqual(temp_obs.movement.value, MovementValue.UNKNOWN)
        self.assertEqual(temp_obs.movement.status, ObservationStatus.INVALID)
        self.assertEqual(self.estimator.history.count, 0)

    def test_reacquisition_after_association_failure(self) -> None:
        """Correction 3: Following association failure, matching detection re-initiates history."""
        box_a = (100.0, 100.0, 200.0, 300.0)
        box_b = (500.0, 400.0, 650.0, 700.0)
        det_b = PersonDetection(bbox_xyxy=box_b, confidence=0.85, class_id=0)
        track = Track(
            track_id=1,
            bbox_xyxy=box_b,
            center=(575.0, 550.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=10.0,
            last_seen_timestamp=10.0,
            hit_count=1,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )

        # Frame 1: Mismatched -> rejected
        f_obs1 = _make_dummy_frame_observation(bbox_xyxy=box_a, timestamp=10.0)
        obs1 = self.estimator.estimate(track, f_obs1, matched_detection=det_b, timestamp=10.0)
        self.assertEqual(obs1.movement.status, ObservationStatus.INVALID)
        self.assertEqual(self.estimator.history.count, 0)

        # Frame 2: Matched detection -> accepted
        f_obs2 = _make_dummy_frame_observation(bbox_xyxy=box_b, timestamp=10.05)
        obs2 = self.estimator.estimate(track, f_obs2, matched_detection=det_b, timestamp=10.05)
        self.assertEqual(obs2.track_id, 1)
        self.assertEqual(self.estimator.history.count, 1)

    def test_predicted_coasting_box_rejected(self) -> None:
        """Verify Kalman predicted / coasting box (observed=False) is rejected."""
        box = (300.0, 200.0, 500.0, 600.0)
        track = Track(
            track_id=1,
            bbox_xyxy=box,
            center=(400.0, 400.0),
            velocity=(10.0, 0.0, 0.0, 0.0),
            timestamp=10.0,
            creation_timestamp=9.0,
            last_seen_timestamp=9.5,
            hit_count=0,
            miss_count=1,
            state=TrackState.LOST,
            observed=False,  # Unobserved Kalman coasting prediction!
        )
        frame_obs = _make_dummy_frame_observation(bbox_xyxy=box, timestamp=10.0)

        temp_obs = self.estimator.estimate(
            track=track,
            frame_observation=frame_obs,
            matched_detection=None,
            timestamp=10.0,
        )

        self.assertFalse(temp_obs.observed)
        self.assertEqual(temp_obs.movement.value, MovementValue.UNKNOWN)
        self.assertEqual(temp_obs.movement.status, ObservationStatus.INVALID)
        self.assertEqual(self.estimator.history.count, 0)

    def test_track_id_change_resets_temporal_history(self) -> None:
        """Verify change in active track_id immediately resets previous history and timers."""
        box = (300.0, 200.0, 500.0, 600.0)
        det = PersonDetection(bbox_xyxy=box, confidence=0.9, class_id=0)

        # Track 1 active for 5 frames -> reach READY
        for i in range(5):
            t = 10.0 + i * 0.033
            tr1 = Track(
                track_id=1,
                bbox_xyxy=box,
                center=(400.0, 400.0),
                velocity=(0.0, 0.0, 0.0, 0.0),
                timestamp=t,
                creation_timestamp=10.0,
                last_seen_timestamp=t,
                hit_count=i + 1,
                miss_count=0,
                state=TrackState.CONFIRMED,
                observed=True,
            )
            f_obs = _make_dummy_frame_observation(bbox_xyxy=box, timestamp=t)
            self.estimator.estimate(tr1, f_obs, matched_detection=det, timestamp=t)

        self.assertTrue(self.estimator.history.is_ready)

        # Track 2 appears (e.g. after track 1 deletion)
        t_new = 10.2
        tr2 = Track(
            track_id=2,
            bbox_xyxy=box,
            center=(400.0, 400.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=t_new,
            creation_timestamp=t_new,
            last_seen_timestamp=t_new,
            hit_count=1,
            miss_count=0,
            state=TrackState.TENTATIVE,
            observed=True,
        )
        f_obs_new = _make_dummy_frame_observation(bbox_xyxy=box, timestamp=t_new)

        temp_obs = self.estimator.estimate(tr2, f_obs_new, matched_detection=det, timestamp=t_new)

        # Should be reinitialized for track 2 (count=1, not READY)
        self.assertEqual(temp_obs.track_id, 2)
        self.assertEqual(self.estimator.history.consecutive_valid_count, 1)
        self.assertFalse(self.estimator.history.is_ready)

    # --- Correction 2: Stability Uncertainty vs Valid Threshold Excursion ---

    def _setup_stable_history(self, duration_s: float = 1.2, dt: float = 0.05) -> tuple[Track, PersonDetection]:
        """Helper to feed stationary observations to establish READY history and start stability."""
        box = (500.0, 300.0, 700.0, 600.0)
        det = PersonDetection(bbox_xyxy=box, confidence=0.95, class_id=0)
        n_steps = int(duration_s / dt)
        track = None
        for i in range(n_steps):
            t = 100.0 + i * dt
            track = Track(
                track_id=1,
                bbox_xyxy=box,
                center=(600.0, 450.0),
                velocity=(0.0, 0.0, 0.0, 0.0),
                timestamp=t,
                creation_timestamp=100.0,
                last_seen_timestamp=t,
                hit_count=i + 1,
                miss_count=0,
                state=TrackState.CONFIRMED,
                observed=True,
            )
            f_obs = _make_dummy_frame_observation(bbox_xyxy=box, timestamp=t)
            self.estimator.estimate(track, f_obs, matched_detection=det, timestamp=t)
        assert track is not None
        return track, det

    def test_stability_stationary_valid_accumulates(self) -> None:
        """Correction 2: STATIONARY + VALID observation accumulates stability time."""
        track, det = self._setup_stable_history(duration_s=1.2)
        self.assertEqual(self.estimator.stability_timer.state, StabilityValue.ACCUMULATING)
        self.assertGreater(self.estimator.stability_timer.elapsed_seconds, 0.0)
        self.assertEqual(self.estimator.stability_timer.violation_seconds, 0.0)

    def test_stability_moving_valid_consumes_tolerance(self) -> None:
        """Correction 2: MOVING + VALID is a valid threshold excursion and consumes tolerance budget."""
        track, det = self._setup_stable_history(duration_s=1.2)
        t_base = 100.0 + 1.2

        # Person moves rapidly within zone for 2 frames (0.10s)
        # Move center from 600 to 650
        box_moved = (550.0, 300.0, 750.0, 600.0)
        det_moved = PersonDetection(bbox_xyxy=box_moved, confidence=0.95, class_id=0)

        t1 = t_base + 0.05
        tr1 = Track(
            track_id=1,
            bbox_xyxy=box_moved,
            center=(650.0, 450.0),
            velocity=(1000.0, 0.0, 0.0, 0.0),
            timestamp=t1,
            creation_timestamp=100.0,
            last_seen_timestamp=t1,
            hit_count=30,
            miss_count=0,
            state=TrackState.CONFIRMED,
            observed=True,
        )
        f_obs1 = _make_dummy_frame_observation(bbox_xyxy=box_moved, timestamp=t1)
        obs1 = self.estimator.estimate(tr1, f_obs1, matched_detection=det_moved, timestamp=t1)

        self.assertEqual(obs1.movement.value, MovementValue.MOVING)
        self.assertEqual(obs1.movement.status, ObservationStatus.VALID)
        # Consumes tolerance: state is still ACCUMULATING, not hard reset
        self.assertEqual(obs1.stability.value, StabilityValue.ACCUMULATING)

    def test_stability_uncertain_evidence_triggers_immediate_hard_reset(self) -> None:
        """Correction 2: UNKNOWN/UNCERTAIN movement evidence triggers hard reset, NOT tolerance consumption."""
        track, det = self._setup_stable_history(duration_s=1.2)
        self.assertEqual(self.estimator.stability_timer.state, StabilityValue.ACCUMULATING)
        t_now = 100.0 + 1.25

        # Frame where interaction zone is UNKNOWN
        f_obs_uncertain = _make_dummy_frame_observation(
            bbox_xyxy=det.bbox_xyxy,
            timestamp=t_now,
            zone_value=InteractionZoneValue.UNKNOWN,
            zone_status=ObservationStatus.UNCERTAIN,
        )
        obs = self.estimator.estimate(track, f_obs_uncertain, matched_detection=det, timestamp=t_now)

        # Must trigger immediate hard reset
        self.assertEqual(obs.stability.value, StabilityValue.RESET)
        self.assertEqual(obs.stability.elapsed_seconds, 0.0)
        self.assertEqual(obs.stability.violation_seconds, 0.0)

    def test_stability_invalid_presence_triggers_hard_reset(self) -> None:
        """Correction 2: INVALID presence status triggers hard reset."""
        track, det = self._setup_stable_history(duration_s=1.2)
        t_now = 100.0 + 1.25

        f_obs_invalid = _make_dummy_frame_observation(
            bbox_xyxy=det.bbox_xyxy,
            timestamp=t_now,
            presence_value=PersonPresenceValue.UNKNOWN,
            presence_status=ObservationStatus.INVALID,
        )
        obs = self.estimator.estimate(track, f_obs_invalid, matched_detection=det, timestamp=t_now)

        self.assertEqual(obs.stability.value, StabilityValue.RESET)
        self.assertEqual(obs.stability.elapsed_seconds, 0.0)
        self.assertEqual(obs.stability.violation_seconds, 0.0)

    def test_tracking_loss_triggers_immediate_hard_reset(self) -> None:
        """Correction 2: Tracking loss (track=None or observed=False) triggers immediate hard reset."""
        track, det = self._setup_stable_history(duration_s=1.2)
        self.assertEqual(self.estimator.stability_timer.state, StabilityValue.ACCUMULATING)

        # Next frame: track is LOST
        lost_track = Track(
            track_id=1,
            bbox_xyxy=det.bbox_xyxy,
            center=(600.0, 450.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            timestamp=102.0,
            creation_timestamp=100.0,
            last_seen_timestamp=101.2,
            hit_count=0,
            miss_count=1,
            state=TrackState.LOST,
            observed=False,
        )
        f_obs = _make_dummy_frame_observation(bbox_xyxy=det.bbox_xyxy, timestamp=102.0)
        obs = self.estimator.estimate(lost_track, f_obs, matched_detection=None, timestamp=102.0)

        self.assertEqual(obs.stability.value, StabilityValue.RESET)
        self.assertEqual(obs.stability.elapsed_seconds, 0.0)
        self.assertEqual(obs.stability.violation_seconds, 0.0)

    def test_reinitialization_following_hard_reset(self) -> None:
        """Correction 2: Following a hard reset, clean stable observation freshly starts stability."""
        track, det = self._setup_stable_history(duration_s=1.2)

        # Trigger hard reset via zone UNKNOWN
        t_reset = 102.0
        f_obs_rst = _make_dummy_frame_observation(
            bbox_xyxy=det.bbox_xyxy,
            timestamp=t_reset,
            zone_value=InteractionZoneValue.UNKNOWN,
            zone_status=ObservationStatus.INVALID,
        )
        obs_rst = self.estimator.estimate(track, f_obs_rst, matched_detection=det, timestamp=t_reset)
        self.assertEqual(obs_rst.stability.value, StabilityValue.RESET)

        # Resume valid stationary frames: history needs to rebuild or continue
        # Feed 1.2s of new valid stationary frames
        box = det.bbox_xyxy
        t_resume = 103.0
        for i in range(25):
            t = t_resume + i * 0.05
            tr = Track(
                track_id=1,
                bbox_xyxy=box,
                center=(600.0, 450.0),
                velocity=(0.0, 0.0, 0.0, 0.0),
                timestamp=t,
                creation_timestamp=100.0,
                last_seen_timestamp=t,
                hit_count=i + 1,
                miss_count=0,
                state=TrackState.CONFIRMED,
                observed=True,
            )
            f_obs = _make_dummy_frame_observation(bbox_xyxy=box, timestamp=t)
            obs = self.estimator.estimate(tr, f_obs, matched_detection=det, timestamp=t)

        self.assertEqual(obs.stability.value, StabilityValue.ACCUMULATING)
        self.assertEqual(obs.stability.violation_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
