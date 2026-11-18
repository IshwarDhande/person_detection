"""Unit tests for Condition 1 — Person Presence and Aggregated Pipeline.

Tests person presence states (DETECTED, NOT_DETECTED, UNKNOWN), detection
confidence preservation, highest-confidence selection, and pipeline aggregation.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from perception.detection.person_detector import PersonDetection
from perception.readiness.frame_observer import FrameObserver
from perception.readiness.observation_types import (
    BboxValidityValue,
    FaceVisibilityValue,
    HeadOrientationValue,
    InteractionZoneValue,
    ObservationStatus,
    PersonPresenceValue,
)


class TestPersonPresenceAndObserver(unittest.TestCase):
    """Test suite for Person Presence and FrameObserver aggregation."""

    def setUp(self) -> None:
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.observer = FrameObserver()

    def test_person_presence_detected(self) -> None:
        """When valid detection is supplied, presence must be DETECTED with confidence."""
        det = PersonDetection(
            bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
            confidence=0.88,
            class_id=0,
        )
        obs = self.observer.observe(self.frame, [det], timestamp=10.0)

        self.assertEqual(obs.person.presence.value, PersonPresenceValue.DETECTED)
        self.assertEqual(obs.person.presence.status, ObservationStatus.VALID)
        self.assertAlmostEqual(obs.person.presence.confidence, 0.88)
        self.assertEqual(obs.frame_metadata.image_width, 1280)
        self.assertEqual(obs.frame_metadata.image_height, 720)
        self.assertEqual(obs.frame_metadata.timestamp, 10.0)

    def test_person_presence_not_detected_is_valid_negative(self) -> None:
        """When detector returns empty list, presence is NOT_DETECTED with VALID status."""
        obs = self.observer.observe(self.frame, [], timestamp=12.0)

        self.assertEqual(obs.person.presence.value, PersonPresenceValue.NOT_DETECTED)
        self.assertEqual(obs.person.presence.status, ObservationStatus.VALID)
        self.assertIsNone(obs.person.presence.confidence)

        # Dependent conditions must be UNKNOWN / INVALID without fabricated data
        self.assertEqual(obs.person.bounding_box.value, BboxValidityValue.UNKNOWN)
        self.assertEqual(obs.person.bounding_box.status, ObservationStatus.INVALID)
        self.assertEqual(obs.person.interaction_zone.value, InteractionZoneValue.UNKNOWN)
        self.assertEqual(obs.face.visibility.value, FaceVisibilityValue.UNKNOWN)
        self.assertEqual(obs.orientation.head_orientation.value, HeadOrientationValue.UNKNOWN)

    def test_detector_failure_is_unknown_invalid(self) -> None:
        """When detector is unavailable (detections is None), presence is UNKNOWN / INVALID."""
        obs = self.observer.observe(self.frame, None, timestamp=15.0)  # type: ignore[arg-type]

        self.assertEqual(obs.person.presence.value, PersonPresenceValue.UNKNOWN)
        self.assertEqual(obs.person.presence.status, ObservationStatus.INVALID)
        self.assertIsNone(obs.person.presence.confidence)

    def test_multiple_detections_selects_highest_confidence(self) -> None:
        """In multi-person frames, the highest-confidence candidate is evaluated."""
        det_low = PersonDetection(bbox_xyxy=(50.0, 50.0, 150.0, 300.0), confidence=0.55, class_id=0)
        det_high = PersonDetection(bbox_xyxy=(400.0, 100.0, 800.0, 600.0), confidence=0.92, class_id=0)

        obs = self.observer.observe(self.frame, [det_low, det_high], timestamp=20.0)

        self.assertEqual(obs.person.presence.value, PersonPresenceValue.DETECTED)
        self.assertAlmostEqual(obs.person.presence.confidence, 0.92)
        assert obs.person.bbox_metrics is not None
        self.assertEqual(obs.person.bbox_metrics.original_bbox_xyxy, det_high.bbox_xyxy)

    def test_invalid_bbox_skips_downstream_face_and_orientation(self) -> None:
        """If bounding box is invalid, downstream face/orientation must be UNKNOWN/INVALID."""
        # Malformed box extending outside image
        det_bad = PersonDetection(bbox_xyxy=(-500.0, -500.0, -100.0, -100.0), confidence=0.70, class_id=0)
        obs = self.observer.observe(self.frame, [det_bad])

        self.assertEqual(obs.person.presence.value, PersonPresenceValue.DETECTED)
        self.assertEqual(obs.person.bounding_box.value, BboxValidityValue.INVALID)
        self.assertEqual(obs.person.bounding_box.status, ObservationStatus.INVALID)
        self.assertEqual(obs.face.visibility.value, FaceVisibilityValue.UNKNOWN)
        self.assertEqual(obs.face.visibility.status, ObservationStatus.INVALID)
        self.assertEqual(obs.orientation.head_orientation.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(obs.orientation.head_orientation.status, ObservationStatus.INVALID)


if __name__ == "__main__":
    unittest.main()

