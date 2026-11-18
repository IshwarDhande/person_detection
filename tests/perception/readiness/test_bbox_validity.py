"""Unit tests for Condition 2 — Person Bounding-Box Validity.

Tests coordinate validation, geometric bounds, image intersection,
clipping, and area-ratio calculation per Milestone M4 specification.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.readiness.bbox_validator import BboxValidator
from perception.readiness.observation_types import (
    BboxValidityValue,
    ObservationStatus,
)


class TestBboxValidity(unittest.TestCase):
    """Test suite for BboxValidator."""

    def test_valid_bounding_box_within_image(self) -> None:
        """A well-formed bounding box inside the frame must be VALID."""
        bbox = (100.0, 150.0, 400.0, 600.0)
        obs, metrics = BboxValidator.validate(bbox, image_width=1280, image_height=720)

        self.assertEqual(obs.value, BboxValidityValue.VALID)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertIsNone(obs.confidence)
        self.assertIsNotNone(metrics)

        assert metrics is not None
        self.assertEqual(metrics.original_bbox_xyxy, bbox)
        self.assertEqual(metrics.clipped_bbox_xyxy, bbox)
        self.assertAlmostEqual(metrics.width, 300.0)
        self.assertAlmostEqual(metrics.height, 450.0)
        self.assertAlmostEqual(metrics.area, 135000.0)
        expected_ratio = 135000.0 / (1280 * 720)
        self.assertAlmostEqual(metrics.area_ratio, expected_ratio)

    def test_reject_non_finite_coordinates(self) -> None:
        """NaN or Inf coordinates must be marked INVALID."""
        for bad_coord in [float("nan"), float("inf"), float("-inf")]:
            bbox = (100.0, bad_coord, 300.0, 400.0)
            obs, metrics = BboxValidator.validate(bbox, image_width=1280, image_height=720)
            self.assertEqual(obs.value, BboxValidityValue.INVALID)
            self.assertEqual(obs.status, ObservationStatus.INVALID)
            self.assertIsNone(metrics)

    def test_reject_inverted_or_zero_dimensions(self) -> None:
        """x2 <= x1 or y2 <= y1 must be marked INVALID."""
        # Inverted x
        obs, metrics = BboxValidator.validate((300.0, 100.0, 200.0, 400.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

        # Inverted y
        obs, metrics = BboxValidator.validate((100.0, 400.0, 300.0, 200.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

        # Zero width
        obs, metrics = BboxValidator.validate((100.0, 100.0, 100.0, 400.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

        # Zero height
        obs, metrics = BboxValidator.validate((100.0, 100.0, 300.0, 100.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

    def test_reject_completely_outside_image(self) -> None:
        """Bounding boxes not intersecting the frame must be INVALID."""
        # Completely to the left
        obs, metrics = BboxValidator.validate((-200.0, 100.0, -10.0, 300.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

        # Completely above
        obs, metrics = BboxValidator.validate((100.0, -200.0, 300.0, -10.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

        # Completely to the right
        obs, metrics = BboxValidator.validate((1300.0, 100.0, 1500.0, 300.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

        # Completely below
        obs, metrics = BboxValidator.validate((100.0, 800.0, 300.0, 1000.0), 1280, 720)
        self.assertEqual(obs.value, BboxValidityValue.INVALID)
        self.assertIsNone(metrics)

    def test_partially_intersecting_bounding_box(self) -> None:
        """Partially intersecting box must be VALID and correctly clipped."""
        # Box extends beyond left and top borders
        bbox = (-50.0, -20.0, 200.0, 300.0)
        obs, metrics = BboxValidator.validate(bbox, image_width=1280, image_height=720)

        self.assertEqual(obs.value, BboxValidityValue.VALID)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertIsNotNone(metrics)

        assert metrics is not None
        self.assertEqual(metrics.original_bbox_xyxy, bbox)
        self.assertEqual(metrics.clipped_bbox_xyxy, (0.0, 0.0, 200.0, 300.0))
        self.assertAlmostEqual(metrics.width, 250.0)
        self.assertAlmostEqual(metrics.height, 320.0)

    def test_box_extending_beyond_right_and_bottom(self) -> None:
        """Box crossing right and bottom edges is clipped to (image_w, image_h)."""
        bbox = (1000.0, 500.0, 1400.0, 800.0)
        obs, metrics = BboxValidator.validate(bbox, image_width=1280, image_height=720)

        self.assertEqual(obs.value, BboxValidityValue.VALID)
        assert metrics is not None
        self.assertEqual(metrics.clipped_bbox_xyxy, (1000.0, 500.0, 1280.0, 720.0))


if __name__ == "__main__":
    unittest.main()

