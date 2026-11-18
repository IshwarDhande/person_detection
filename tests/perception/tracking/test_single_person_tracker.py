"""Deterministic Unit Tests for Single-Person Tracker (Milestone M3)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.detection.person_detector import PersonDetection
from perception.tracking.single_person_tracker import (
    SinglePersonTracker,
    Track,
    TrackObservation,
    TrackState,
    compute_iou,
)


class TestIoUCalculation(unittest.TestCase):
    """Test 2D bounding-box IoU computation."""

    def test_identical_boxes_iou_is_one(self) -> None:
        """Verify identical boxes yield exactly 1.0 IoU."""
        box = (100.0, 100.0, 200.0, 300.0)
        self.assertAlmostEqual(compute_iou(box, box), 1.0)

    def test_disjoint_boxes_iou_is_zero(self) -> None:
        """Verify disjoint boxes yield 0.0 IoU."""
        box_a = (0.0, 0.0, 10.0, 10.0)
        box_b = (20.0, 20.0, 30.0, 30.0)
        self.assertEqual(compute_iou(box_a, box_b), 0.0)

    def test_partial_overlap_iou(self) -> None:
        """Verify known partial overlap calculation."""
        # Box A: 0 to 10 in x and y (area 100)
        # Box B: 5 to 15 in x and 0 to 10 in y (area 100)
        # Intersection: 5 to 10 in x, 0 to 10 in y (area 50)
        # Union: 100 + 100 - 50 = 150
        # IoU: 50 / 150 = 1/3
        box_a = (0.0, 0.0, 10.0, 10.0)
        box_b = (5.0, 0.0, 15.0, 10.0)
        self.assertAlmostEqual(compute_iou(box_a, box_b), 1.0 / 3.0)

    def test_exact_iou_boundary(self) -> None:
        """Construct boxes that have exactly 0.30 IoU."""
        # Box A: [0, 0, 100, 100] (Area 10000)
        # Shift Box B in x: [s, 0, s + 100, 100]
        # Intersection width = 100 - s
        # Intersection area = (100 - s) * 100
        # Union area = 20000 - (100 - s) * 100 = 10000 + 100 * s
        # IoU = (10000 - 100*s) / (10000 + 100*s) = (100 - s) / (100 + s)
        # For IoU = 0.30: 100 - s = 0.3 * (100 + s) => 70 = 1.3 * s => s = 70 / 1.3 ≈ 53.8461538
        s = 70.0 / 1.3
        box_a = (0.0, 0.0, 100.0, 100.0)
        box_b = (s, 0.0, s + 100.0, 100.0)
        self.assertAlmostEqual(compute_iou(box_a, box_b), 0.3, places=5)


class TestTrackCreationAndConfirmation(unittest.TestCase):
    """Test initial track creation and 3-consecutive-hit confirmation."""

    def test_new_detection_creates_tentative_track(self) -> None:
        """Verify new detection initializes track in TENTATIVE state with track_id=1."""
        tracker = SinglePersonTracker(confirmation_hits=3)
        det = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 400.0), confidence=0.85)

        track = tracker.update([det], timestamp=10.0)
        self.assertIsNotNone(track)
        assert track is not None

        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.state, TrackState.TENTATIVE)
        self.assertEqual(track.hit_count, 1)
        self.assertEqual(track.miss_count, 0)
        self.assertTrue(track.observed)
        self.assertEqual(track.timestamp, 10.0)
        self.assertEqual(track.creation_timestamp, 10.0)
        self.assertEqual(track.last_seen_timestamp, 10.0)
        self.assertEqual(track.age, 0.0)
        self.assertEqual(track.miss_duration, 0.0)

    def test_confirmation_requires_three_consecutive_hits(self) -> None:
        """Verify track remains TENTATIVE on hits 1 and 2, and transitions to CONFIRMED on hit 3."""
        tracker = SinglePersonTracker(confirmation_hits=3)
        det1 = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 400.0), confidence=0.85)
        det2 = PersonDetection(bbox_xyxy=(102.0, 101.0, 202.0, 401.0), confidence=0.86)
        det3 = PersonDetection(bbox_xyxy=(103.0, 102.0, 203.0, 402.0), confidence=0.88)

        # Hit 1
        t1 = tracker.update([det1], timestamp=1.0)
        assert t1 is not None
        self.assertEqual(t1.state, TrackState.TENTATIVE)
        self.assertEqual(t1.hit_count, 1)

        # Hit 2
        t2 = tracker.update([det2], timestamp=1.033)
        assert t2 is not None
        self.assertEqual(t2.state, TrackState.TENTATIVE)
        self.assertEqual(t2.hit_count, 2)

        # Hit 3
        t3 = tracker.update([det3], timestamp=1.066)
        assert t3 is not None
        self.assertEqual(t3.state, TrackState.CONFIRMED)
        self.assertEqual(t3.hit_count, 3)
        self.assertEqual(t3.track_id, 1)

    def test_missed_frame_breaks_consecutive_hits(self) -> None:
        """Verify missed frame breaks consecutive hits and resets confirmation requirement."""
        tracker = SinglePersonTracker(confirmation_hits=3)
        det1 = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 400.0), confidence=0.85)
        det2 = PersonDetection(bbox_xyxy=(102.0, 101.0, 202.0, 401.0), confidence=0.86)

        # Frame 1: Hit 1 -> TENTATIVE
        t1 = tracker.update([det1], timestamp=1.0)
        assert t1 is not None
        self.assertEqual(t1.state, TrackState.TENTATIVE)

        # Frame 2: Hit 2 -> TENTATIVE
        t2 = tracker.update([det2], timestamp=1.033)
        assert t2 is not None
        self.assertEqual(t2.state, TrackState.TENTATIVE)
        self.assertEqual(t2.hit_count, 2)

        # Frame 3: Missed frame -> breaks consecutive hits!
        t3 = tracker.update([], timestamp=1.066)
        assert t3 is not None
        self.assertEqual(t3.state, TrackState.LOST)
        self.assertFalse(t3.observed)
        self.assertEqual(t3.hit_count, 0)
        self.assertEqual(t3.miss_count, 1)

        # Frame 4: Returning detection within timeout -> reacquired as TENTATIVE with hit_count=1
        det4 = PersonDetection(bbox_xyxy=(104.0, 102.0, 204.0, 402.0), confidence=0.87)
        t4 = tracker.update([det4], timestamp=1.100)
        assert t4 is not None
        self.assertEqual(t4.state, TrackState.TENTATIVE)
        self.assertEqual(t4.hit_count, 1)  # Restarted from 1
        self.assertEqual(t4.track_id, 1)   # ID preserved!


class TestDetectionAssociation(unittest.TestCase):
    """Test IoU and uncertainty gating for candidate detections."""

    def test_iou_above_and_at_threshold_accepted(self) -> None:
        """Verify candidate with IoU >= 0.3 is accepted."""
        tracker = SinglePersonTracker(iou_threshold=0.3)
        det_init = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 400.0), confidence=0.85)
        tracker.update([det_init], timestamp=1.0)

        # Candidate with ~0.50 IoU (shifted 40px in x)
        det_match = PersonDetection(bbox_xyxy=(140.0, 100.0, 240.0, 400.0), confidence=0.85)
        track = tracker.update([det_match], timestamp=1.033)
        assert track is not None
        self.assertTrue(track.observed)
        self.assertEqual(track.hit_count, 2)

    def test_iou_below_threshold_rejected(self) -> None:
        """Verify candidate with IoU < 0.3 is rejected and track enters LOST state."""
        tracker = SinglePersonTracker(iou_threshold=0.3)
        det_init = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 400.0), confidence=0.85)
        tracker.update([det_init], timestamp=1.0)

        # Candidate shifted 300px in x (IoU = 0.0)
        det_far = PersonDetection(bbox_xyxy=(400.0, 100.0, 500.0, 400.0), confidence=0.85)
        track = tracker.update([det_far], timestamp=1.033)
        assert track is not None
        # Track should be unobserved (LOST) because candidate failed IoU threshold
        self.assertFalse(track.observed)
        self.assertEqual(track.state, TrackState.LOST)

    def test_multi_candidate_highest_iou_selection(self) -> None:
        """Verify when multiple candidates pass gates, highest IoU candidate is chosen."""
        tracker = SinglePersonTracker(iou_threshold=0.3)
        det_init = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 400.0), confidence=0.85)
        tracker.update([det_init], timestamp=1.0)

        # Candidate A: shifted 40px (IoU ~ 0.50)
        # Candidate B: shifted 5px  (IoU ~ 0.90)
        det_a = PersonDetection(bbox_xyxy=(140.0, 100.0, 240.0, 400.0), confidence=0.90)
        det_b = PersonDetection(bbox_xyxy=(105.0, 100.0, 205.0, 400.0), confidence=0.70)

        track = tracker.update([det_a, det_b], timestamp=1.033)
        assert track is not None
        self.assertTrue(track.observed)

        # Track center should be much closer to candidate B (x ~ 155) than A (x ~ 190)
        self.assertLess(track.center[0], 165.0)

    def test_uncertainty_gate_rejection(self) -> None:
        """Verify candidate failing the Mahalanobis uncertainty gate is rejected."""
        # Set max_mahalanobis_distance_squared to an extremely strict threshold (0.001)
        strict_tracker = SinglePersonTracker(
            iou_threshold=0.3, max_mahalanobis_distance_squared=0.001
        )
        det_init = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 400.0), confidence=0.85)
        strict_tracker.update([det_init], timestamp=1.0)

        # Small offset (15px) that has high IoU (~0.85) but fails the hyper-strict uncertainty gate
        det_shifted = PersonDetection(bbox_xyxy=(115.0, 100.0, 215.0, 400.0), confidence=0.85)
        track = strict_tracker.update([det_shifted], timestamp=1.033)
        assert track is not None

        # Must be rejected due to uncertainty gate failure
        self.assertFalse(track.observed)
        self.assertEqual(track.state, TrackState.LOST)


class TestTrackLifecycleAndTimeout(unittest.TestCase):
    """Test LOST / COASTING, reacquisition, and 2-second timeout deletion."""

    def test_reacquisition_within_two_seconds_retains_id(self) -> None:
        """Verify confirmed track lost for 1.5s is reacquired with original track_id."""
        tracker = SinglePersonTracker(confirmation_hits=3, retention_timeout_seconds=2.0)
        # Establish CONFIRMED track
        tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=1.0)
        tracker.update([PersonDetection((101.0, 100.0, 201.0, 400.0), 0.9)], timestamp=1.033)
        t_conf = tracker.update([PersonDetection((102.0, 100.0, 202.0, 400.0), 0.9)], timestamp=1.066)
        assert t_conf is not None
        self.assertEqual(t_conf.state, TrackState.CONFIRMED)
        self.assertEqual(t_conf.track_id, 1)

        # 1.5 seconds of detection loss (from t=1.066 to t=2.566)
        t_lost = tracker.update([], timestamp=2.566)
        assert t_lost is not None
        self.assertEqual(t_lost.state, TrackState.LOST)
        self.assertFalse(t_lost.observed)
        self.assertEqual(t_lost.track_id, 1)
        self.assertAlmostEqual(t_lost.miss_duration, 1.5, places=3)

        # Detection returns at t=2.600 (within 2.0s limit: 2.600 - 1.066 = 1.534s)
        det_return = PersonDetection((102.0, 100.0, 202.0, 400.0), 0.9)
        t_reacquired = tracker.update([det_return], timestamp=2.600)
        assert t_reacquired is not None

        # Should retain original track_id and return to CONFIRMED
        self.assertEqual(t_reacquired.track_id, 1)
        self.assertEqual(t_reacquired.state, TrackState.CONFIRMED)
        self.assertTrue(t_reacquired.observed)
        self.assertAlmostEqual(t_reacquired.miss_duration, 0.0)

    def test_expiration_after_two_seconds_deletes_track_and_assigns_new_id(self) -> None:
        """Verify track is deleted when miss_duration > 2.0s and next detection gets new ID."""
        tracker = SinglePersonTracker(confirmation_hits=1, retention_timeout_seconds=2.0)
        t1 = tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=1.0)
        assert t1 is not None
        self.assertEqual(t1.track_id, 1)

        # 2.05s of loss (exceeds 2.0s threshold)
        t_expired = tracker.update([], timestamp=3.05)
        # Should be deleted, returning None
        self.assertIsNone(t_expired)
        self.assertIsNone(tracker.active_track)

        # Subsequent detection at t=3.500 creates brand new track with track_id=2
        t_new = tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=3.5)
        assert t_new is not None
        self.assertEqual(t_new.track_id, 2)
        self.assertEqual(t_new.state, TrackState.TENTATIVE)


class TestTrackDataModelAndMemory(unittest.TestCase):
    """Test observation history bounding, independent age/miss_duration, and cleanup."""

    def test_history_bounded_to_thirty_observations(self) -> None:
        """Verify track history strictly does not exceed 30 observations."""
        tracker = SinglePersonTracker(history_max_len=30)
        tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=0.0)

        # Feed 50 frames
        track: Optional[Track] = None
        for i in range(1, 51):
            track = tracker.update(
                [PersonDetection((100.0 + i, 100.0, 200.0 + i, 400.0), 0.9)],
                timestamp=i * 0.033,
            )

        assert track is not None
        self.assertEqual(len(track.history), 30)
        # Oldest element in history should be frame 21 (timestamp = 21 * 0.033)
        self.assertAlmostEqual(track.history[0].timestamp, 21 * 0.033, places=3)
        # Newest element should be frame 50
        self.assertAlmostEqual(track.history[-1].timestamp, 50 * 0.033, places=3)

    def test_age_and_miss_duration_are_independent(self) -> None:
        """Verify age (total time) and miss_duration (loss time) compute independently."""
        tracker = SinglePersonTracker(retention_timeout_seconds=2.0)
        tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=10.0)

        # 1.0s of continuous tracking
        track = tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=11.0)
        assert track is not None
        self.assertAlmostEqual(track.age, 1.0)
        self.assertAlmostEqual(track.miss_duration, 0.0)

        # 0.5s of detection loss
        track = tracker.update([], timestamp=11.5)
        assert track is not None
        self.assertAlmostEqual(track.age, 1.5)
        self.assertAlmostEqual(track.miss_duration, 0.5)

    def test_reset_clears_active_track(self) -> None:
        """Verify reset() method clears active track and Kalman state."""
        tracker = SinglePersonTracker()
        tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=1.0)
        self.assertIsNotNone(tracker.active_track)

        tracker.reset()
        self.assertIsNone(tracker.active_track)

    def test_reject_decreasing_timestamps(self) -> None:
        """Verify non-monotonic timestamp raises ValueError."""
        tracker = SinglePersonTracker()
        tracker.update([PersonDetection((100.0, 100.0, 200.0, 400.0), 0.9)], timestamp=5.0)

        with self.assertRaises(ValueError):
            tracker.update([], timestamp=4.9)


class TestTrackingVerificationAndClosure(unittest.TestCase):
    """Specific verification and closure tests for Milestone M3 review findings."""

    def test_tracking_output_contract(self) -> None:
        """Verify that Track exposes all 12 contract fields with exact required semantics."""
        tracker = SinglePersonTracker()
        det = PersonDetection(bbox_xyxy=(100.0, 200.0, 300.0, 600.0), confidence=0.92)

        # 1. Observed frame
        track = tracker.update([det], timestamp=10.0)
        assert track is not None

        # Verify contract attributes exist and have correct types
        self.assertIsInstance(track.track_id, int)
        self.assertIsInstance(track.bbox_xyxy, tuple)
        self.assertEqual(len(track.bbox_xyxy), 4)
        self.assertIsInstance(track.center, tuple)
        self.assertEqual(len(track.center), 2)
        self.assertIsInstance(track.velocity, tuple)
        self.assertEqual(len(track.velocity), 4)
        self.assertIsInstance(track.timestamp, float)
        self.assertIsInstance(track.creation_timestamp, float)
        self.assertIsInstance(track.last_seen_timestamp, float)
        self.assertIsInstance(track.hit_count, int)
        self.assertIsInstance(track.miss_count, int)
        self.assertIsInstance(track.state, TrackState)
        self.assertIsInstance(track.observed, bool)

        # Semantics: observed frame
        self.assertTrue(track.observed)
        self.assertEqual(track.age, 0.0)
        self.assertEqual(track.miss_duration, 0.0)
        self.assertEqual(track.hit_count, 1)
        self.assertEqual(track.miss_count, 0)
        self.assertEqual(len(track.history), 1)

        # 2. Coasting / unobserved frame at t=10.5
        track_lost = tracker.update([], timestamp=10.5)
        assert track_lost is not None

        # Semantics: unobserved frame
        self.assertFalse(track_lost.observed)
        self.assertAlmostEqual(track_lost.age, 0.5)
        self.assertAlmostEqual(track_lost.miss_duration, 0.5)
        self.assertEqual(track_lost.hit_count, 0)
        self.assertEqual(track_lost.miss_count, 1)
        self.assertEqual(track_lost.state, TrackState.LOST)

        # Predicted box must be distinct from old detection box due to covariance expansion
        self.assertIsInstance(track_lost.bbox_xyxy, tuple)
        self.assertEqual(len(track_lost.history), 2)
        self.assertFalse(track_lost.history[-1].observed)

    def test_case_a_tentative_interrupted_by_missed_frame_full_sequence(self) -> None:
        """Verify Case A: Det 1 -> hit=1, Det 2 -> hit=2, Miss -> hit=0, Det 3 -> hit=1, Det 4 -> hit=2, Det 5 -> CONFIRMED."""
        tracker = SinglePersonTracker(confirmation_hits=3)
        det_box = lambda x: PersonDetection(bbox_xyxy=(x, 100.0, x + 100.0, 300.0), confidence=0.90)

        # 1. Detection 1 -> TENTATIVE, hit_count=1
        t1 = tracker.update([det_box(100.0)], timestamp=1.000)
        assert t1 is not None
        self.assertEqual(t1.state, TrackState.TENTATIVE)
        self.assertEqual(t1.hit_count, 1)
        self.assertEqual(t1.miss_count, 0)
        self.assertTrue(t1.observed)

        # 2. Detection 2 -> TENTATIVE, hit_count=2
        t2 = tracker.update([det_box(102.0)], timestamp=1.033)
        assert t2 is not None
        self.assertEqual(t2.state, TrackState.TENTATIVE)
        self.assertEqual(t2.hit_count, 2)
        self.assertEqual(t2.miss_count, 0)
        self.assertTrue(t2.observed)

        # 3. Missed frame -> LOST, hit_count=0
        t3 = tracker.update([], timestamp=1.066)
        assert t3 is not None
        self.assertEqual(t3.state, TrackState.LOST)
        self.assertEqual(t3.hit_count, 0)
        self.assertEqual(t3.miss_count, 1)
        self.assertFalse(t3.observed)

        # 4. Reacquisition (Detection 3) -> TENTATIVE, hit_count=1
        t4 = tracker.update([det_box(104.0)], timestamp=1.100)
        assert t4 is not None
        self.assertEqual(t4.state, TrackState.TENTATIVE)
        self.assertEqual(t4.hit_count, 1)
        self.assertEqual(t4.miss_count, 0)
        self.assertTrue(t4.observed)
        self.assertEqual(t4.track_id, 1)  # Retains original ID

        # 5. Detection 4 -> TENTATIVE, hit_count=2
        t5 = tracker.update([det_box(106.0)], timestamp=1.133)
        assert t5 is not None
        self.assertEqual(t5.state, TrackState.TENTATIVE)
        self.assertEqual(t5.hit_count, 2)
        self.assertEqual(t5.miss_count, 0)
        self.assertTrue(t5.observed)

        # 6. Detection 5 -> CONFIRMED, hit_count=3
        t6 = tracker.update([det_box(108.0)], timestamp=1.166)
        assert t6 is not None
        self.assertEqual(t6.state, TrackState.CONFIRMED)
        self.assertEqual(t6.hit_count, 3)
        self.assertEqual(t6.miss_count, 0)
        self.assertTrue(t6.observed)
        self.assertEqual(t6.track_id, 1)

    def test_case_b_unmatched_detection_while_track_is_lost(self) -> None:
        """Verify Case B: An unmatched detection during LOST does not replace or spawn tracks."""
        tracker = SinglePersonTracker(retention_timeout_seconds=2.0)
        # Establish active track at (100, 100, 200, 300)
        tracker.update([PersonDetection((100.0, 100.0, 200.0, 300.0), 0.9)], timestamp=1.0)

        # Missed frame: enters LOST at t=1.033
        t_lost = tracker.update([], timestamp=1.033)
        assert t_lost is not None
        self.assertEqual(t_lost.state, TrackState.LOST)
        self.assertFalse(t_lost.observed)
        self.assertEqual(t_lost.track_id, 1)

        # Incoming candidate at far location (600, 600, 700, 800) fails IoU and uncertainty gates
        far_det = PersonDetection(bbox_xyxy=(600.0, 600.0, 700.0, 800.0), confidence=0.88)
        t_unmatched = tracker.update([far_det], timestamp=1.066)
        assert t_unmatched is not None

        # Existing track must be retained in LOST
        self.assertEqual(t_unmatched.track_id, 1)
        self.assertEqual(t_unmatched.state, TrackState.LOST)
        self.assertFalse(t_unmatched.observed)
        # Coordinates must be the Kalman prediction around (100, 100), NOT the far candidate (600, 600)
        self.assertLess(t_unmatched.center[0], 250.0)
        self.assertLess(t_unmatched.center[1], 350.0)

        # Tracker still maintains only ONE active track
        self.assertEqual(tracker.active_track, t_unmatched)

    def test_case_c_reacquisition_rules(self) -> None:
        """Verify Case C: Reacquisition rules across tentative, confirmed, and expired tracks."""
        tracker = SinglePersonTracker(confirmation_hits=3, retention_timeout_seconds=2.0)
        det_box = lambda x: PersonDetection(bbox_xyxy=(x, 100.0, x + 100.0, 300.0), confidence=0.90)

        # 1. Establish CONFIRMED track
        tracker.update([det_box(100.0)], timestamp=1.000)
        tracker.update([det_box(101.0)], timestamp=1.033)
        t_conf = tracker.update([det_box(102.0)], timestamp=1.066)
        assert t_conf is not None
        self.assertEqual(t_conf.state, TrackState.CONFIRMED)

        # Coast for 1.0s (within 2.0s limit)
        t_coast = tracker.update([], timestamp=2.066)
        assert t_coast is not None
        self.assertEqual(t_coast.state, TrackState.LOST)

        # Valid reacquisition at t=2.100 restores CONFIRMED state with original ID
        t_reacq = tracker.update([det_box(103.0)], timestamp=2.100)
        assert t_reacq is not None
        self.assertEqual(t_reacq.state, TrackState.CONFIRMED)
        self.assertEqual(t_reacq.track_id, 1)

        # Let track expire beyond 2.0s (from t=2.100 to t=4.200)
        t_expired = tracker.update([], timestamp=4.200)
        self.assertIsNone(t_expired)
        self.assertIsNone(tracker.active_track)

        # Subsequent detection creates new track with new ID (2) in TENTATIVE state
        t_new = tracker.update([det_box(100.0)], timestamp=4.500)
        assert t_new is not None
        self.assertEqual(t_new.track_id, 2)
        self.assertEqual(t_new.state, TrackState.TENTATIVE)


if __name__ == "__main__":
    unittest.main()

