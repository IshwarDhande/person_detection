"""Unit tests for Condition 3 — Face Visibility and Quality.

Tests MediaPipe Face Landmarker integration, dimension thresholds (40x40),
landmark quality checks, aspect-ratio consistency, failure modes, and mocks.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from perception.readiness.face_analyzer import FaceAnalyzer
from perception.readiness.observation_types import (
    FaceVisibilityValue,
    ObservationStatus,
)


class MockNormalizedLandmark:
    """Mock MediaPipe NormalizedLandmark."""
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class TestFaceAnalyzer(unittest.TestCase):
    """Test suite for FaceAnalyzer."""

    def setUp(self) -> None:
        # Create an instance with dummy or mocked landmarker
        self.dummy_crop = np.zeros((200, 200, 3), dtype=np.uint8)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_face_sufficient_quality(self, mock_create: MagicMock) -> None:
        """Face detected with adequate dimensions and landmarks must be SUFFICIENT / VALID."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        # Return 478 mock landmarks spanning a 100x100 box in a 200x200 crop
        # (normalized coords from 0.25 to 0.75 => 50px to 150px = 100px width/height)
        lms = [
            MockNormalizedLandmark(
                0.25 + 0.5 * (i % 20) / 20.0,
                0.25 + 0.5 * (i // 20) / 25.0,
            )
            for i in range(478)
        ]
        mock_result = MagicMock(spec=["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"])
        mock_result.face_landmarks = [lms]
        mock_landmarker.detect.return_value = mock_result

        analyzer = FaceAnalyzer(
            model_path="models/face_landmarker.task",
            min_face_width=40,
            min_face_height=40,
        )
        obs, metrics, full_lms = analyzer.analyze(self.dummy_crop, crop_offset_x=10.0, crop_offset_y=20.0)

        self.assertEqual(obs.value, FaceVisibilityValue.SUFFICIENT)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertIsNone(obs.confidence)  # FaceLandmarker API does not expose detection score; must be None
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertGreaterEqual(metrics.face_width, 40)
        self.assertGreaterEqual(metrics.face_height, 40)
        self.assertEqual(metrics.landmark_count, 478)
        self.assertIsNone(metrics.detection_confidence)
        self.assertIsNotNone(full_lms)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_face_confidence_uses_actual_model_score_when_available(self, mock_create: MagicMock) -> None:
        """When the MediaPipe result exposes an actual detection score, it must be used."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        lms = [
            MockNormalizedLandmark(0.25 + 0.5 * (i % 20) / 20.0, 0.25 + 0.5 * (i // 20) / 25.0)
            for i in range(478)
        ]
        mock_result = MagicMock()
        mock_result.face_landmarks = [lms]
        # Simulate an actual model-provided category detection score of 0.93
        mock_cat = MagicMock()
        mock_cat.score = 0.93
        mock_det = MagicMock()
        mock_det.categories = [mock_cat]
        mock_result.face_detections = [mock_det]
        mock_landmarker.detect.return_value = mock_result

        analyzer = FaceAnalyzer(model_path="models/face_landmarker.task", min_detection_confidence=0.50)
        obs, metrics, _ = analyzer.analyze(self.dummy_crop)

        self.assertEqual(obs.value, FaceVisibilityValue.SUFFICIENT)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertAlmostEqual(obs.confidence, 0.93)
        assert metrics is not None
        self.assertAlmostEqual(metrics.detection_confidence, 0.93)
        # Ensure configured threshold 0.50 is NOT fabricated as confidence
        self.assertNotEqual(obs.confidence, 0.50)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_missing_required_landmarks_returns_unknown_invalid(self, mock_create: MagicMock) -> None:
        """Face result missing required landmarks must evaluate to UNKNOWN / INVALID."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        # Only 50 landmarks returned (missing required chin at index 152 and right eye at 263)
        lms = [
            MockNormalizedLandmark(0.25 + 0.5 * (i % 5) / 5.0, 0.25 + 0.5 * (i // 5) / 10.0)
            for i in range(50)
        ]
        mock_result = MagicMock(spec=["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"])
        mock_result.face_landmarks = [lms]
        mock_landmarker.detect.return_value = mock_result

        analyzer = FaceAnalyzer(model_path="models/face_landmarker.task", min_face_width=20, min_face_height=20)
        obs, metrics, _ = analyzer.analyze(self.dummy_crop)

        self.assertEqual(obs.value, FaceVisibilityValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_non_finite_required_landmarks_returns_unknown_invalid(self, mock_create: MagicMock) -> None:
        """Required landmarks with NaN coordinates must return UNKNOWN / INVALID."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        lms = [
            MockNormalizedLandmark(0.25 + 0.5 * (i % 20) / 20.0, 0.25 + 0.5 * (i // 20) / 25.0)
            for i in range(478)
        ]
        # Set nose tip (index 1) to NaN
        lms[1] = MockNormalizedLandmark(float("nan"), 0.5)

        mock_result = MagicMock(spec=["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"])
        mock_result.face_landmarks = [lms]
        mock_landmarker.detect.return_value = mock_result

        analyzer = FaceAnalyzer(model_path="models/face_landmarker.task")
        obs, metrics, _ = analyzer.analyze(self.dummy_crop)

        self.assertEqual(obs.value, FaceVisibilityValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_degenerate_landmark_geometry_returns_unknown_uncertain(self, mock_create: MagicMock) -> None:
        """Collapsed eye distance (< min_eye_distance) must return UNKNOWN / UNCERTAIN."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        lms = [
            MockNormalizedLandmark(0.25 + 0.5 * (i % 20) / 20.0, 0.25 + 0.5 * (i // 20) / 25.0)
            for i in range(478)
        ]
        # Collapse left eye (33) and right eye (263) to same horizontal coordinate
        lms[33] = MockNormalizedLandmark(0.50, 0.40)
        lms[263] = MockNormalizedLandmark(0.501, 0.40)  # Only 0.2px apart in a 200px crop (< 4px)

        mock_result = MagicMock(spec=["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"])
        mock_result.face_landmarks = [lms]
        mock_landmarker.detect.return_value = mock_result

        analyzer = FaceAnalyzer(model_path="models/face_landmarker.task", min_eye_distance=4.0)
        obs, metrics, _ = analyzer.analyze(self.dummy_crop)

        self.assertEqual(obs.value, FaceVisibilityValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.UNCERTAIN)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_no_face_detected_is_insufficient(self, mock_create: MagicMock) -> None:
        """When detector finds no face, result is INSUFFICIENT with VALID status."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_result = MagicMock()
        mock_result.face_landmarks = []  # No face
        mock_landmarker.detect.return_value = mock_result

        analyzer = FaceAnalyzer(model_path="models/face_landmarker.task")
        obs, metrics, full_lms = analyzer.analyze(self.dummy_crop)

        self.assertEqual(obs.value, FaceVisibilityValue.INSUFFICIENT)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertIsNone(metrics)
        self.assertIsNone(full_lms)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_face_smaller_than_minimum_dimensions(self, mock_create: MagicMock) -> None:
        """Face smaller than 40x40 must evaluate to INSUFFICIENT with VALID status."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        # Span only 0.05 to 0.10 in a 200x200 crop => 10px by 10px
        lms = [
            MockNormalizedLandmark(0.05 + 0.05 * (i % 5) / 5.0, 0.05 + 0.05 * (i // 5) / 5.0)
            for i in range(25)
        ]
        mock_result = MagicMock(spec=["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"])
        mock_result.face_landmarks = [lms]
        mock_landmarker.detect.return_value = mock_result

        analyzer = FaceAnalyzer(
            model_path="models/face_landmarker.task",
            min_face_width=40,
            min_face_height=40,
        )
        obs, metrics, _ = analyzer.analyze(self.dummy_crop)

        self.assertEqual(obs.value, FaceVisibilityValue.INSUFFICIENT)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertLess(metrics.face_width, 40)

    @patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options")
    def test_face_inference_exception_is_unknown_invalid(self, mock_create: MagicMock) -> None:
        """Model execution failure must return UNKNOWN with INVALID status."""
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.side_effect = RuntimeError("MediaPipe crash")

        analyzer = FaceAnalyzer(model_path="models/face_landmarker.task")
        obs, metrics, lms = analyzer.analyze(self.dummy_crop)

        self.assertEqual(obs.value, FaceVisibilityValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)
        self.assertIsNone(metrics)
        self.assertIsNone(lms)

    def test_empty_or_none_crop(self) -> None:
        """Empty or None image crop evaluates to INSUFFICIENT / VALID."""
        with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options"):
            analyzer = FaceAnalyzer(model_path="models/face_landmarker.task")
            obs, metrics, _ = analyzer.analyze(np.zeros((0, 0, 3), dtype=np.uint8))
            self.assertEqual(obs.value, FaceVisibilityValue.INSUFFICIENT)
            self.assertEqual(obs.status, ObservationStatus.VALID)


if __name__ == "__main__":
    unittest.main()

