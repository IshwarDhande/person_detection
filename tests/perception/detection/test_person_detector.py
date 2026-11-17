"""Unit tests for PersonDetector and PersonDetection (Milestone M1).

These tests run completely deterministically without requiring a physical webcam
or downloading model weights during execution.

```text
+-----------------------------------------------------------------------------+
|                     UNIT TESTS: PERSON DETECTOR & DETECTION                 |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [TestPersonDetection]                                                      |
|    - Valid dataclass construction and properties (w, h, center)             |
|    - Validation: class_id != 0 rejected                                     |
|    - Validation: confidence outside [0.0, 1.0] rejected                      |
|    - Validation: non-finite or inverted bounding box coordinates rejected   |
|                                                                             |
|  [TestPersonDetectorInit]                                                   |
|    - Missing model weights -> FileNotFoundError                             |
|    - Invalid confidence thresholds -> ValueError                            |
|    - Boundary confidence thresholds (0.0, 1.0) accepted                     |
|    - Model initialization error wrapped in contextual RuntimeError          |
|                                                                             |
|  [TestPersonDetectorInference]                                              |
|    - Retain COCO class 0 ('person')                                         |
|    - Filter out non-person classes (car, dog, chair)                        |
|    - Filter out below-threshold confidence detections                       |
|    - Boundary confidence threshold (conf == threshold) retained             |
|    - Empty model results -> empty list []                                   |
|    - Coordinate and confidence precision preservation                       |
|    - Invalid input frame validation (None, wrong type, empty ndarray)       |
|    - Resource cleanup on close()                                            |
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

from perception.detection.person_detector import PersonDetection, PersonDetector


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


class TestPersonDetection(unittest.TestCase):
    """Test suite for PersonDetection dataclass and validation rules."""

    def test_valid_detection_instantiation(self) -> None:
        """Verify normal creation with valid coordinates and confidence."""
        det = PersonDetection(
            bbox_xyxy=(100.0, 50.0, 200.0, 300.0),
            confidence=0.88,
            class_id=0,
        )
        self.assertEqual(det.bbox_xyxy, (100.0, 50.0, 200.0, 300.0))
        self.assertAlmostEqual(det.confidence, 0.88)
        self.assertEqual(det.class_id, 0)
        self.assertAlmostEqual(det.width, 100.0)
        self.assertAlmostEqual(det.height, 250.0)
        self.assertEqual(det.center_xy, (150.0, 175.0))

    def test_reject_non_person_class_id(self) -> None:
        """Verify class_id != 0 is rejected with ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PersonDetection(
                bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
                confidence=0.9,
                class_id=1,  # Non-person (bicycle)
            )
        self.assertIn("class_id=0", str(ctx.exception))

    def test_reject_out_of_bound_confidence(self) -> None:
        """Verify confidence outside [0.0, 1.0] is rejected."""
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), confidence=-0.01, class_id=0)
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), confidence=1.01, class_id=0)
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), confidence=float("nan"), class_id=0)

    def test_reject_invalid_bbox_length(self) -> None:
        """Verify bbox_xyxy requires exactly 4 coordinates."""
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(0.0, 0.0, 10.0), confidence=0.5, class_id=0)  # type: ignore

    def test_reject_non_finite_coordinates(self) -> None:
        """Verify NaN or Inf coordinates are rejected."""
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(0.0, 0.0, float("nan"), 10.0), confidence=0.5, class_id=0)
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(0.0, 0.0, float("inf"), 10.0), confidence=0.5, class_id=0)

    def test_reject_inverted_coordinates(self) -> None:
        """Verify inverted bounding boxes (x1 > x2 or y1 > y2) are rejected."""
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(100.0, 0.0, 50.0, 50.0), confidence=0.5, class_id=0)
        with self.assertRaises(ValueError):
            PersonDetection(bbox_xyxy=(0.0, 100.0, 50.0, 50.0), confidence=0.5, class_id=0)


class TestPersonDetectorInit(unittest.TestCase):
    """Test suite for PersonDetector initialization and configuration."""

    def test_missing_model_path_raises_file_not_found(self) -> None:
        """Verify missing model file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            PersonDetector(model_path="non_existent_weights_xyz123.pt")
        self.assertIn("Model weights file not found", str(ctx.exception))

    def test_invalid_confidence_threshold_raises_value_error(self) -> None:
        """Verify invalid confidence thresholds are rejected."""
        # Use existing yolov8n.pt or dummy file
        valid_model = "yolov8n.pt"
        if not Path(valid_model).is_file():
            self.skipTest(f"{valid_model} not present for threshold test")

        with self.assertRaises(ValueError):
            PersonDetector(model_path=valid_model, confidence_threshold=-0.1)
        with self.assertRaises(ValueError):
            PersonDetector(model_path=valid_model, confidence_threshold=1.05)
        with self.assertRaises(ValueError):
            PersonDetector(model_path=valid_model, confidence_threshold=float("nan"))
        with self.assertRaises(ValueError):
            PersonDetector(model_path=valid_model, confidence_threshold="0.5")  # type: ignore

    @patch("ultralytics.YOLO")
    def test_boundary_confidence_thresholds(self, mock_yolo: MagicMock) -> None:
        """Verify boundary thresholds 0.0 and 1.0 are accepted."""
        # Create a temporary dummy file to satisfy Path.is_file()
        with patch.object(Path, "is_file", return_value=True):
            det0 = PersonDetector(model_path="dummy.pt", confidence_threshold=0.0)
            self.assertEqual(det0.confidence_threshold, 0.0)

            det1 = PersonDetector(model_path="dummy.pt", confidence_threshold=1.0)
            self.assertEqual(det1.confidence_threshold, 1.0)

    @patch("ultralytics.YOLO", side_effect=Exception("Corrupted model file"))
    def test_model_init_failure_raises_runtime_error(self, mock_yolo: MagicMock) -> None:
        """Verify underlying YOLO initialization errors are wrapped in RuntimeError."""
        with patch.object(Path, "is_file", return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                PersonDetector(model_path="corrupted.pt")
            self.assertIn("Failed to initialize YOLO model", str(ctx.exception))


class TestPersonDetectorInference(unittest.TestCase):
    """Test suite for inference post-processing and filtering logic."""

    def setUp(self) -> None:
        """Create detector instance with mocked YOLO backend."""
        self.mock_yolo_patcher = patch("ultralytics.YOLO")
        self.mock_yolo_cls = self.mock_yolo_patcher.start()
        self.mock_model_instance = MagicMock()
        self.mock_yolo_cls.return_value = self.mock_model_instance

        with patch.object(Path, "is_file", return_value=True):
            self.detector = PersonDetector(model_path="dummy.pt", confidence_threshold=0.5)

    def tearDown(self) -> None:
        self.mock_yolo_patcher.stop()

    def test_person_class_retained(self) -> None:
        """Verify detections with class_id 0 and conf >= 0.5 are retained."""
        fake_boxes = FakeBoxes(
            xyxy=[[10.0, 20.0, 100.0, 200.0]],
            conf=[0.85],
            cls=[0.0],
        )
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(dummy_frame)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].class_id, 0)
        self.assertEqual(results[0].bbox_xyxy, (10.0, 20.0, 100.0, 200.0))
        self.assertAlmostEqual(results[0].confidence, 0.85, places=5)

    def test_non_person_classes_filtered_out(self) -> None:
        """Verify detections with class_id != 0 are strictly excluded."""
        fake_boxes = FakeBoxes(
            xyxy=[
                [10.0, 20.0, 100.0, 200.0],  # Person (class 0)
                [15.0, 25.0, 50.0, 60.0],    # Bicycle (class 1)
                [200.0, 300.0, 400.0, 500.0], # Car (class 2)
                [50.0, 50.0, 90.0, 90.0],     # Chair (class 56)
            ],
            conf=[0.90, 0.95, 0.88, 0.75],
            cls=[0.0, 1.0, 2.0, 56.0],
        )
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(dummy_frame)

        # Only the class 0 (person) should survive
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].class_id, 0)
        self.assertEqual(results[0].bbox_xyxy, (10.0, 20.0, 100.0, 200.0))

    def test_below_threshold_detections_filtered_out(self) -> None:
        """Verify detections below confidence_threshold are excluded."""
        fake_boxes = FakeBoxes(
            xyxy=[
                [10.0, 20.0, 100.0, 200.0],
                [30.0, 40.0, 80.0, 90.0],
            ],
            conf=[0.49, 0.25],  # Both below threshold 0.50
            cls=[0.0, 0.0],
        )
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(dummy_frame)

        self.assertEqual(len(results), 0)

    def test_boundary_confidence_threshold_retained(self) -> None:
        """Verify detection with confidence exactly equal to threshold is retained."""
        fake_boxes = FakeBoxes(
            xyxy=[[10.0, 20.0, 100.0, 200.0]],
            conf=[0.50],  # Exactly equal to threshold 0.50
            cls=[0.0],
        )
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(dummy_frame)

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].confidence, 0.50, places=5)

    def test_empty_model_results_produce_empty_list(self) -> None:
        """Verify empty boxes produce [] without error."""
        # Case A: Empty boxes tensor
        fake_boxes = FakeBoxes(xyxy=[], conf=[], cls=[])
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(self.detector.detect(dummy_frame), [])

        # Case B: boxes is None
        self.mock_model_instance.predict.return_value = [FakeResult(None)]
        self.assertEqual(self.detector.detect(dummy_frame), [])

        # Case C: results is empty list
        self.mock_model_instance.predict.return_value = []
        self.assertEqual(self.detector.detect(dummy_frame), [])

    def test_multiple_persons_detected(self) -> None:
        """Verify multi-person frame returns all qualified person detections."""
        fake_boxes = FakeBoxes(
            xyxy=[
                [10.0, 20.0, 100.0, 200.0],
                [300.0, 150.0, 450.0, 400.0],
                [500.0, 100.0, 600.0, 350.0],
            ],
            conf=[0.85, 0.72, 0.91],
            cls=[0.0, 0.0, 0.0],
        )
        self.mock_model_instance.predict.return_value = [FakeResult(fake_boxes)]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(dummy_frame)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].bbox_xyxy, (10.0, 20.0, 100.0, 200.0))
        self.assertEqual(results[1].bbox_xyxy, (300.0, 150.0, 450.0, 400.0))
        self.assertEqual(results[2].bbox_xyxy, (500.0, 100.0, 600.0, 350.0))

    def test_invalid_input_frame_raises_value_error(self) -> None:
        """Verify None, non-array, or empty array raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.detect(None)  # type: ignore

        with self.assertRaises(ValueError):
            self.detector.detect("not_an_image")  # type: ignore

        with self.assertRaises(ValueError):
            self.detector.detect(np.array([]))

    def test_inference_failure_raises_runtime_error(self) -> None:
        """Verify inference exceptions are raised as RuntimeError."""
        self.mock_model_instance.predict.side_effect = RuntimeError("GPU out of memory")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with self.assertRaises(RuntimeError) as ctx:
            self.detector.detect(dummy_frame)
        self.assertIn("YOLO inference failed", str(ctx.exception))

    def test_close_detector_releases_resources(self) -> None:
        """Verify close() clears model reference and prevents subsequent calls."""
        self.detector.close()
        self.assertIsNone(self.detector.model)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError) as ctx:
            self.detector.detect(dummy_frame)
        self.assertIn("closed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
