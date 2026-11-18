"""Unit tests for SinglePersonSmoother and M2 PersonDetection extensions.

Tests verify deterministic mathematical EMA smoothing, detection-count behaviour,
state resets, and detector integration without requiring live webcam hardware.

```text
+-----------------------------------------------------------------------------+
|               UNIT TESTS: SINGLE-PERSON EMA SMOOTHER (M2)                   |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [TestSinglePersonSmootherInit]                                             |
|    - Default alpha == 0.65                                                  |
|    - Alpha bounds: (0.0, 1.0] valid; alpha == 1.0 accepted                  |
|    - Invalid alpha (<=0, >1, NaN, inf, non-numeric) -> ValueError           |
|                                                                             |
|  [TestSinglePersonSmootherMath]                                             |
|    - First detection initializes smoothed box (b_hat_0 = b_0)               |
|    - Multi-step EMA matches exact formula: b_hat_t = a*b_t + (1-a)*b_hat_t-1|
|    - Raw bbox coordinates remain strictly untouched                         |
|    - Output coordinates are finite and properly ordered                     |
|                                                                             |
|  [TestDetectionCountBehaviour]                                              |
|    - Count == 0: Returns [] and resets smoothing state                      |
|    - Count == 1: Applies EMA smoothing                                      |
|    - Count > 1:  Bypasses smoothing, returns raw, and resets state          |
|    - No association across ambiguous multi-person frames                    |
|    - No temporal retention: missed detection reports [] immediately        |
|                                                                             |
|  [TestPersonDetectorSmootherIntegration]                                    |
|    - detector.detect() with smoother applies EMA                            |
|    - detector.detect_raw() returns raw detections unmodified                |
|    - detector.close() resets smoother state                                 |
|                                                                             |
+-----------------------------------------------------------------------------+
```
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Sequence
from unittest.mock import MagicMock, patch
import unittest

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from perception.detection.person_detector import (
    PersonDetection,
    PersonDetector,
    SinglePersonSmoother,
)


class FakeBoxes:
    """Mock Boxes object mimicking Ultralytics engine.results.Boxes."""

    def __init__(
        self,
        xyxy: Sequence[Sequence[float]],
        conf: Sequence[float],
        cls: Sequence[float],
    ) -> None:
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeResult:
    """Mock Results object mimicking Ultralytics engine.results.Results."""

    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes


class TestPersonDetectionM2Extensions(unittest.TestCase):
    """Test suite for PersonDetection dataclass extensions introduced in M2."""

    def test_default_smoothed_bbox_is_none(self) -> None:
        """Verify backward compatibility: smoothed_bbox_xyxy defaults to None."""
        det = PersonDetection(
            bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
            confidence=0.9,
            class_id=0,
        )
        self.assertIsNone(det.smoothed_bbox_xyxy)
        self.assertFalse(det.is_smoothed)
        self.assertEqual(det.effective_bbox_xyxy, (10.0, 20.0, 100.0, 200.0))

    def test_valid_smoothed_bbox_instantiation(self) -> None:
        """Verify PersonDetection holds both raw and smoothed coordinates."""
        det = PersonDetection(
            bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
            confidence=0.9,
            class_id=0,
            smoothed_bbox_xyxy=(12.0, 22.0, 98.0, 198.0),
        )
        self.assertEqual(det.bbox_xyxy, (10.0, 20.0, 100.0, 200.0))
        self.assertEqual(det.smoothed_bbox_xyxy, (12.0, 22.0, 98.0, 198.0))
        self.assertTrue(det.is_smoothed)
        self.assertEqual(det.effective_bbox_xyxy, (12.0, 22.0, 98.0, 198.0))

    def test_reject_invalid_smoothed_bbox(self) -> None:
        """Verify invalid smoothed coordinates are rejected."""
        # Non-finite coordinate
        with self.assertRaises(ValueError):
            PersonDetection(
                bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
                confidence=0.9,
                class_id=0,
                smoothed_bbox_xyxy=(10.0, float("nan"), 100.0, 200.0),
            )
        # Inverted coordinate (sx1 > sx2)
        with self.assertRaises(ValueError):
            PersonDetection(
                bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
                confidence=0.9,
                class_id=0,
                smoothed_bbox_xyxy=(150.0, 20.0, 100.0, 200.0),
            )
        # Invalid length
        with self.assertRaises(ValueError):
            PersonDetection(
                bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
                confidence=0.9,
                class_id=0,
                smoothed_bbox_xyxy=(10.0, 20.0, 100.0),  # type: ignore
            )


class TestSinglePersonSmootherInit(unittest.TestCase):
    """Test suite for SinglePersonSmoother initialization and coefficient validation."""

    def test_default_alpha(self) -> None:
        """Verify default alpha is 0.65 and smoother starts uninitialized."""
        smoother = SinglePersonSmoother()
        self.assertAlmostEqual(smoother.alpha, 0.65)
        self.assertFalse(smoother.is_initialized)
        self.assertIsNone(smoother.current_smoothed_box)

    def test_valid_custom_alpha(self) -> None:
        """Verify valid alpha in (0.0, 1.0] is accepted."""
        smoother = SinglePersonSmoother(alpha=0.5)
        self.assertAlmostEqual(smoother.alpha, 0.5)

        # Boundary alpha == 1.0 (no smoothing)
        smoother_1 = SinglePersonSmoother(alpha=1.0)
        self.assertAlmostEqual(smoother_1.alpha, 1.0)

    def test_reject_invalid_alpha(self) -> None:
        """Verify non-positive, >1, or non-finite alpha raises ValueError."""
        with self.assertRaises(ValueError):
            SinglePersonSmoother(alpha=0.0)
        with self.assertRaises(ValueError):
            SinglePersonSmoother(alpha=-0.2)
        with self.assertRaises(ValueError):
            SinglePersonSmoother(alpha=1.01)
        with self.assertRaises(ValueError):
            SinglePersonSmoother(alpha=float("nan"))
        with self.assertRaises(ValueError):
            SinglePersonSmoother(alpha=float("inf"))
        with self.assertRaises(ValueError):
            SinglePersonSmoother(alpha="0.65")  # type: ignore


class TestSinglePersonSmootherMath(unittest.TestCase):
    """Test suite for mathematical correctness of EMA calculations."""

    def test_first_detection_initializes_box(self) -> None:
        """Verify the first valid detection directly initializes the smoothed box."""
        smoother = SinglePersonSmoother(alpha=0.6)
        raw_det = PersonDetection(
            bbox_xyxy=(100.0, 50.0, 200.0, 300.0),
            confidence=0.92,
            class_id=0,
        )
        results = smoother.smooth([raw_det])

        self.assertEqual(len(results), 1)
        res = results[0]
        # Raw box remains strictly unchanged
        self.assertEqual(res.bbox_xyxy, (100.0, 50.0, 200.0, 300.0))
        # Smoothed box initialized to raw box
        self.assertEqual(res.smoothed_bbox_xyxy, (100.0, 50.0, 200.0, 300.0))
        self.assertAlmostEqual(res.confidence, 0.92)
        self.assertEqual(res.class_id, 0)
        self.assertTrue(smoother.is_initialized)

    def test_multi_step_ema_calculation(self) -> None:
        """Verify multi-step EMA matches: b_hat_t = alpha * b_t + (1 - alpha) * b_hat_(t-1)."""
        alpha = 0.6
        smoother = SinglePersonSmoother(alpha=alpha)

        # Frame 1: Raw = (100.0, 100.0, 200.0, 200.0)
        # Expected smoothed: (100.0, 100.0, 200.0, 200.0)
        det1 = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 200.0), confidence=0.85)
        out1 = smoother.smooth([det1])[0]
        self.assertEqual(out1.smoothed_bbox_xyxy, (100.0, 100.0, 200.0, 200.0))

        # Frame 2: Raw = (110.0, 110.0, 210.0, 210.0)
        # Calculation: 0.6 * 110 + 0.4 * 100 = 66 + 40 = 106.0
        # Expected smoothed: (106.0, 106.0, 206.0, 206.0)
        det2 = PersonDetection(bbox_xyxy=(110.0, 110.0, 210.0, 210.0), confidence=0.88)
        out2 = smoother.smooth([det2])[0]
        self.assertEqual(out2.bbox_xyxy, (110.0, 110.0, 210.0, 210.0))
        self.assertEqual(out2.smoothed_bbox_xyxy, (106.0, 106.0, 206.0, 206.0))

        # Frame 3: Raw = (120.0, 120.0, 220.0, 220.0)
        # Calculation: 0.6 * 120 + 0.4 * 106 = 72 + 42.4 = 114.4
        # Expected smoothed: (114.4, 114.4, 214.4, 214.4)
        det3 = PersonDetection(bbox_xyxy=(120.0, 120.0, 220.0, 220.0), confidence=0.90)
        out3 = smoother.smooth([det3])[0]
        self.assertEqual(out3.bbox_xyxy, (120.0, 120.0, 220.0, 220.0))
        self.assertAlmostEqual(out3.smoothed_bbox_xyxy[0], 114.4, places=5)  # type: ignore
        self.assertAlmostEqual(out3.smoothed_bbox_xyxy[1], 114.4, places=5)  # type: ignore
        self.assertAlmostEqual(out3.smoothed_bbox_xyxy[2], 214.4, places=5)  # type: ignore
        self.assertAlmostEqual(out3.smoothed_bbox_xyxy[3], 214.4, places=5)  # type: ignore


class TestDetectionCountBehaviour(unittest.TestCase):
    """Test suite for handling 0, 1, and >1 person detection frames."""

    def test_zero_detections_resets_smoother(self) -> None:
        """Verify zero detections immediately returns [] and resets smoothing state."""
        smoother = SinglePersonSmoother(alpha=0.6)

        # Initialize with detection
        det1 = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 200.0), confidence=0.9)
        smoother.smooth([det1])
        self.assertTrue(smoother.is_initialized)

        # Zero detection frame
        empty_out = smoother.smooth([])
        self.assertEqual(empty_out, [])
        self.assertFalse(smoother.is_initialized)
        self.assertIsNone(smoother.current_smoothed_box)

        # Subsequent detection must start fresh (not interpolate with frame 1)
        det2 = PersonDetection(bbox_xyxy=(300.0, 300.0, 400.0, 400.0), confidence=0.9)
        out2 = smoother.smooth([det2])[0]
        self.assertEqual(out2.smoothed_bbox_xyxy, (300.0, 300.0, 400.0, 400.0))

    def test_multiple_detections_bypasses_and_resets(self) -> None:
        """Verify multiple detections bypass smoothing and reset smoothing state."""
        smoother = SinglePersonSmoother(alpha=0.6)

        # Initialize with single detection
        det1 = PersonDetection(bbox_xyxy=(100.0, 100.0, 200.0, 200.0), confidence=0.9)
        smoother.smooth([det1])
        self.assertTrue(smoother.is_initialized)

        # Multi-person frame (2 persons)
        person_a = PersonDetection(bbox_xyxy=(50.0, 50.0, 150.0, 250.0), confidence=0.88)
        person_b = PersonDetection(bbox_xyxy=(300.0, 50.0, 400.0, 250.0), confidence=0.85)

        multi_out = smoother.smooth([person_a, person_b])
        # Returns raw detections unmodified
        self.assertEqual(len(multi_out), 2)
        self.assertEqual(multi_out[0].bbox_xyxy, person_a.bbox_xyxy)
        self.assertIsNone(multi_out[0].smoothed_bbox_xyxy)
        self.assertEqual(multi_out[1].bbox_xyxy, person_b.bbox_xyxy)
        self.assertIsNone(multi_out[1].smoothed_bbox_xyxy)
        # Smoother must be reset
        self.assertFalse(smoother.is_initialized)

        # Subsequent single detection starts fresh
        det_single = PersonDetection(bbox_xyxy=(200.0, 200.0, 300.0, 300.0), confidence=0.9)
        single_out = smoother.smooth([det_single])[0]
        self.assertEqual(single_out.smoothed_bbox_xyxy, (200.0, 200.0, 300.0, 300.0))

    def test_explicit_reset_clears_state(self) -> None:
        """Verify reset() method clears current smoothed box."""
        smoother = SinglePersonSmoother()
        smoother.smooth([PersonDetection(bbox_xyxy=(10.0, 10.0, 20.0, 20.0), confidence=0.8)])
        self.assertTrue(smoother.is_initialized)

        smoother.reset()
        self.assertFalse(smoother.is_initialized)
        self.assertIsNone(smoother.current_smoothed_box)


class TestPersonDetectorSmootherIntegration(unittest.TestCase):
    """Test suite for PersonDetector integration with SinglePersonSmoother."""

    def setUp(self) -> None:
        self.mock_yolo_patcher = patch("ultralytics.YOLO")
        self.mock_yolo_cls = self.mock_yolo_patcher.start()
        self.mock_model_instance = MagicMock()
        self.mock_yolo_cls.return_value = self.mock_model_instance

        with patch.object(Path, "is_file", return_value=True):
            self.smoother = SinglePersonSmoother(alpha=0.6)
            self.detector = PersonDetector(
                model_path="dummy.pt",
                confidence_threshold=0.5,
                smoother=self.smoother,
            )

    def tearDown(self) -> None:
        self.mock_yolo_patcher.stop()

    def test_detector_smooths_single_person(self) -> None:
        """Verify detector.detect() routes through smoother when configured."""
        fake_boxes = FakeBoxes(
            xyxy=[[100.0, 100.0, 200.0, 200.0]],
            conf=[0.90],
            cls=[0.0],
        )
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(dummy_frame)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_smoothed)
        self.assertEqual(results[0].bbox_xyxy, (100.0, 100.0, 200.0, 200.0))
        self.assertEqual(results[0].smoothed_bbox_xyxy, (100.0, 100.0, 200.0, 200.0))

    def test_detector_raw_bypasses_smoother(self) -> None:
        """Verify detector.detect_raw() returns unsmoothed PersonDetection."""
        fake_boxes = FakeBoxes(
            xyxy=[[100.0, 100.0, 200.0, 200.0]],
            conf=[0.90],
            cls=[0.0],
        )
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw_results = self.detector.detect_raw(dummy_frame)

        self.assertEqual(len(raw_results), 1)
        self.assertFalse(raw_results[0].is_smoothed)
        self.assertIsNone(raw_results[0].smoothed_bbox_xyxy)

    def test_close_detector_resets_smoother(self) -> None:
        """Verify closing detector resets internal smoother state."""
        self.smoother._smoothed_box = (10.0, 10.0, 20.0, 20.0)
        self.assertTrue(self.smoother.is_initialized)

        self.detector.close()
        self.assertFalse(self.smoother.is_initialized)
        self.assertIsNone(self.detector.model)


if __name__ == "__main__":
    unittest.main()

