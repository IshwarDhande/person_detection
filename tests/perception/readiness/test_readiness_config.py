"""Unit tests for Milestone M4 Configuration Loading and Validation.

Tests parsing of YAML configuration, default values, bounds validation,
and invalid parameter rejection.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.readiness.readiness_config import (
    CameraConfig,
    DetectorConfig,
    FaceAnalyzerConfig,
    InteractionZoneConfig,
    OrientationConfig,
    ReadinessConfig,
    load_readiness_config,
)


class TestReadinessConfig(unittest.TestCase):
    """Test suite for readiness configuration."""

    def test_load_default_m4_config(self) -> None:
        """Loading default configs/m4/stationary_readiness.yaml must succeed."""
        config_path = "configs/m4/stationary_readiness.yaml"
        cfg = load_readiness_config(config_path)

        self.assertIsInstance(cfg, ReadinessConfig)
        self.assertEqual(cfg.camera.width, 1280)
        self.assertEqual(cfg.camera.height, 720)
        self.assertEqual(cfg.camera.fps, 30)
        self.assertAlmostEqual(cfg.detector.confidence_threshold, 0.50)
        self.assertEqual(cfg.detector.class_id, 0)
        self.assertEqual(cfg.face_analyzer.min_face_width, 40)
        self.assertEqual(cfg.face_analyzer.min_face_height, 40)
        self.assertAlmostEqual(cfg.orientation.yaw_threshold, 0.35)
        self.assertTrue(cfg.interaction_zone.enabled)
        self.assertEqual(len(cfg.interaction_zone.vertices), 4)

    def test_missing_config_file_raises_filenotfound(self) -> None:
        """Non-existent config file must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_readiness_config("non_existent_config_file.yaml")

    def test_camera_config_validation(self) -> None:
        """Negative or zero camera dimensions must raise ValueError."""
        with self.assertRaises(ValueError):
            CameraConfig(width=0)
        with self.assertRaises(ValueError):
            CameraConfig(height=-10)
        with self.assertRaises(ValueError):
            CameraConfig(fps=0)

    def test_detector_config_validation(self) -> None:
        """Invalid confidence or non-zero class_id must raise ValueError."""
        with self.assertRaises(ValueError):
            DetectorConfig(confidence_threshold=-0.1)
        with self.assertRaises(ValueError):
            DetectorConfig(confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            DetectorConfig(class_id=1)

    def test_face_analyzer_config_validation(self) -> None:
        """Negative min dimensions or out-of-range confidence must raise ValueError."""
        with self.assertRaises(ValueError):
            FaceAnalyzerConfig(min_face_width=0)
        with self.assertRaises(ValueError):
            FaceAnalyzerConfig(min_detection_confidence=1.2)

    def test_orientation_config_validation(self) -> None:
        """Non-positive thresholds or uncertain_multiplier <= 1.0 must raise ValueError."""
        with self.assertRaises(ValueError):
            OrientationConfig(yaw_threshold=-0.1)
        with self.assertRaises(ValueError):
            OrientationConfig(uncertain_multiplier=0.9)

    def test_interaction_zone_config_validation(self) -> None:
        """Vertices count < 3 or out of [0, 1] bounds must raise ValueError."""
        with self.assertRaises(ValueError):
            InteractionZoneConfig(vertices=((0.1, 0.1), (0.2, 0.2)))
        with self.assertRaises(ValueError):
            InteractionZoneConfig(vertices=((0.1, 0.1), (1.5, 0.2), (0.3, 0.3)))

    def test_interaction_zone_rejects_degenerate_and_collinear_polygon(self) -> None:
        """Collinear vertices producing zero area must raise ValueError."""
        with self.assertRaises(ValueError):
            # Collinear points forming a line with zero area
            InteractionZoneConfig(vertices=((0.1, 0.1), (0.3, 0.3), (0.8, 0.8)))

    def test_interaction_zone_rejects_self_intersecting_polygon(self) -> None:
        """Self-intersecting (bow-tie / figure-8) polygon must raise ValueError."""
        with self.assertRaises(ValueError):
            # Bow-tie crossing edges
            InteractionZoneConfig(vertices=(
                (0.1, 0.1),
                (0.9, 0.9),
                (0.9, 0.1),
                (0.1, 0.9),
            ))

    def test_interaction_zone_rejects_duplicate_consecutive_vertices(self) -> None:
        """Consecutive duplicate vertices must raise ValueError."""
        with self.assertRaises(ValueError):
            InteractionZoneConfig(vertices=(
                (0.2, 0.2),
                (0.2, 0.2),
                (0.8, 0.8),
                (0.2, 0.8),
            ))

    def test_interaction_zone_rejects_malformed_coordinate_dimensions(self) -> None:
        """Vertices not having exactly 2 coordinates must raise ValueError."""
        with self.assertRaises(ValueError):
            InteractionZoneConfig(vertices=(
                (0.1, 0.2, 0.3),  # 3 coords
                (0.8, 0.2),
                (0.8, 0.8),
            ))


if __name__ == "__main__":
    unittest.main()

