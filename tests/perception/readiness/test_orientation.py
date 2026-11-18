"""Unit tests for Condition 5 — Robot-Facing Head Orientation.

Tests landmark geometric ratios, frontal vs turned classifications,
ambiguous state handling, missing/malformed landmark rejection, and
threshold configurability.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.readiness.observation_types import (
    HeadOrientationValue,
    ObservationStatus,
)
from perception.readiness.orientation_estimator import OrientationEstimator


def _create_mock_landmarks(
    nose_x: float = 100.0,
    nose_y: float = 100.0,
    left_eye_x: float = 70.0,
    left_eye_y: float = 80.0,
    right_eye_x: float = 130.0,
    right_eye_y: float = 80.0,
    forehead_x: float = 100.0,
    forehead_y: float = 60.0,
    chin_x: float = 100.0,
    chin_y: float = 140.0,
    total_count: int = 478,
) -> list[tuple[float, float, float]]:
    """Helper to generate a mock landmark list with key canonical indices populated."""
    lms = [(0.0, 0.0, 0.0)] * max(total_count, 478)
    lms[1] = (nose_x, nose_y, 0.0)          # Nose tip
    lms[33] = (left_eye_x, left_eye_y, 0.0)   # Left eye outer
    lms[263] = (right_eye_x, right_eye_y, 0.0) # Right eye outer
    lms[10] = (forehead_x, forehead_y, 0.0)  # Forehead
    lms[152] = (chin_x, chin_y, 0.0)        # Chin
    return lms


class TestOrientationEstimator(unittest.TestCase):
    """Test suite for OrientationEstimator."""

    def setUp(self) -> None:
        self.estimator = OrientationEstimator(
            yaw_threshold=0.35,
            pitch_threshold=0.60,
            uncertain_multiplier=1.5,
        )

    def test_perfectly_frontal_face(self) -> None:
        """Symmetric nose-to-eyes and vertical landmarks must evaluate to FACING_ROBOT."""
        # Nose at 100, Left eye at 70 (dist 30), Right eye at 130 (dist 30) -> yaw_ratio = 0.0
        # Forehead at 60 (dist 40), Chin at 140 (dist 40) -> pitch_ratio = 0.0
        lms = _create_mock_landmarks(
            nose_x=100.0,
            left_eye_x=70.0,
            right_eye_x=130.0,
            nose_y=100.0,
            forehead_y=60.0,
            chin_y=140.0,
        )
        obs, metrics = self.estimator.estimate(lms)

        self.assertEqual(obs.value, HeadOrientationValue.FACING_ROBOT)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics.yaw_ratio, 0.0)
        self.assertAlmostEqual(metrics.pitch_ratio, 0.0)

    def test_clearly_turned_face_yaw(self) -> None:
        """Face turned horizontally past threshold*1.5 must evaluate to NOT_FACING_ROBOT."""
        # Nose at 120, Left eye at 70 (dist 50), Right eye at 130 (dist 10)
        # yaw_ratio = |50 - 10| / (50 + 10) = 40 / 60 = 0.667 (> 0.35 * 1.5 = 0.525)
        lms = _create_mock_landmarks(
            nose_x=120.0,
            left_eye_x=70.0,
            right_eye_x=130.0,
        )
        obs, metrics = self.estimator.estimate(lms)

        self.assertEqual(obs.value, HeadOrientationValue.NOT_FACING_ROBOT)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        assert metrics is not None
        self.assertGreater(metrics.yaw_ratio, 0.525)

    def test_clearly_tilted_face_pitch(self) -> None:
        """Face tilted vertically past threshold*1.5 must evaluate to NOT_FACING_ROBOT."""
        # Nose at 70, Forehead at 65 (dist 5), Chin at 150 (dist 80)
        # pitch_ratio = |5 - 80| / 85 = 75 / 85 = 0.882 (> 0.60 * 1.5 = 0.90? Let's check 0.60*1.5 = 0.90)
        # Let's make it dist 2 and dist 80 -> |2 - 80| / 82 = 78/82 = 0.951 > 0.90
        lms = _create_mock_landmarks(
            nose_y=67.0,
            forehead_y=65.0,
            chin_y=150.0,
        )
        obs, metrics = self.estimator.estimate(lms)

        self.assertEqual(obs.value, HeadOrientationValue.NOT_FACING_ROBOT)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_ambiguous_orientation_state(self) -> None:
        """Ratios in the ambiguous band must return UNKNOWN with UNCERTAIN status."""
        # yaw_threshold is 0.35, upper bound is 0.35 * 1.5 = 0.525
        # We want yaw_ratio around 0.42:
        # |d1 - d2| / (d1 + d2) = 0.42 => d1 = 42.6, d2 = 17.4 (sum 60)
        # Nose at 112.6, Left eye at 70 (dist 42.6), Right eye at 130 (dist 17.4)
        lms = _create_mock_landmarks(
            nose_x=112.6,
            left_eye_x=70.0,
            right_eye_x=130.0,
        )
        obs, metrics = self.estimator.estimate(lms)

        self.assertEqual(obs.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.UNCERTAIN)
        assert metrics is not None
        self.assertTrue(0.35 <= metrics.yaw_ratio < 0.525)

    def test_insufficient_or_empty_landmarks(self) -> None:
        """Empty or truncated landmark lists must evaluate to UNKNOWN / INVALID."""
        obs1, _ = self.estimator.estimate([])
        self.assertEqual(obs1.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(obs1.status, ObservationStatus.INVALID)

        # Truncated list missing chin (index 152)
        obs2, _ = self.estimator.estimate([(0.0, 0.0, 0.0)] * 50)
        self.assertEqual(obs2.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(obs2.status, ObservationStatus.INVALID)

    def test_non_finite_landmarks(self) -> None:
        """Landmarks with NaN coordinates must evaluate to UNKNOWN / INVALID."""
        lms = _create_mock_landmarks(nose_x=float("nan"))
        obs, _ = self.estimator.estimate(lms)
        self.assertEqual(obs.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)

    def test_degenerate_horizontal_distance_returns_unknown_invalid(self) -> None:
        """Collapsed horizontal landmarks (d_sum < 1e-5) must evaluate to UNKNOWN / INVALID."""
        # Collapse nose and both eyes to identical x
        lms = _create_mock_landmarks(
            nose_x=100.0,
            left_eye_x=100.0,
            right_eye_x=100.0,
        )
        obs, metrics = self.estimator.estimate(lms)
        self.assertEqual(obs.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)
        self.assertIsNone(metrics)

    def test_degenerate_vertical_distance_returns_unknown_invalid(self) -> None:
        """Collapsed vertical landmarks (d_vert_sum < 1e-5) must evaluate to UNKNOWN / INVALID."""
        # Collapse nose, forehead, and chin to identical y
        lms = _create_mock_landmarks(
            nose_y=100.0,
            forehead_y=100.0,
            chin_y=100.0,
        )
        obs, metrics = self.estimator.estimate(lms)
        self.assertEqual(obs.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)
        self.assertIsNone(metrics)

    def test_custom_threshold_configuration(self) -> None:
        """Estimator must strictly respect configured custom thresholds."""
        strict_estimator = OrientationEstimator(
            yaw_threshold=0.10,
            pitch_threshold=0.20,
            uncertain_multiplier=1.2,
        )
        # Yaw ratio of 0.15 is above strict 0.10 threshold
        # nose at 105: |35 - 25| / 60 = 10/60 = 0.167 > 0.10 * 1.2 = 0.12 => NOT_FACING_ROBOT
        lms = _create_mock_landmarks(
            nose_x=105.0,
            left_eye_x=70.0,
            right_eye_x=130.0,
        )
        obs, _ = strict_estimator.estimate(lms)
        self.assertEqual(obs.value, HeadOrientationValue.NOT_FACING_ROBOT)


if __name__ == "__main__":
    unittest.main()

