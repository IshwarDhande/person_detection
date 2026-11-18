"""Unit tests for M4 Demo Runner CLI Parsing and Lifecycle (Milestone M4)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.readiness.run_readiness_demo import (
    DEFAULT_CONFIG_PATH,
    compute_application_fps,
    parse_args,
    run_readiness_pipeline,
)


class TestReadinessDemoRunner(unittest.TestCase):
    """Test suite for readiness demo runner CLI argument parsing and lifecycle."""

    def test_default_cli_arguments(self) -> None:
        """Verify default CLI arguments."""
        args = parse_args([])
        self.assertEqual(args.config, DEFAULT_CONFIG_PATH)
        self.assertIsNone(args.camera)
        self.assertIsNone(args.model)
        self.assertIsNone(args.conf)
        self.assertIsNone(args.max_frames)
        self.assertFalse(args.headless)

    def test_custom_cli_arguments(self) -> None:
        """Verify CLI overrides for camera, dimensions, and bounds."""
        args = parse_args([
            "--config", "custom_config.yaml",
            "--camera", "/dev/video1",
            "--model", "custom.pt",
            "--conf", "0.75",
            "--width", "640",
            "--height", "480",
            "--fps", "15",
            "--max-frames", "50",
            "--headless",
        ])
        self.assertEqual(args.config, "custom_config.yaml")
        self.assertEqual(args.camera, "/dev/video1")
        self.assertEqual(args.model, "custom.pt")
        self.assertAlmostEqual(args.conf, 0.75)
        self.assertEqual(args.width, 640)
        self.assertEqual(args.height, 480)
        self.assertEqual(args.fps, 15)
        self.assertEqual(args.max_frames, 50)
        self.assertTrue(args.headless)

    def test_reject_invalid_conf(self) -> None:
        """Verify invalid confidence raises error."""
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "-0.1"])
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "1.2"])

    def test_reject_invalid_dimensions(self) -> None:
        """Verify non-positive dimensions raise error."""
        with self.assertRaises(SystemExit):
            parse_args(["--width", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--height", "-10"])
        with self.assertRaises(SystemExit):
            parse_args(["--fps", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--max-frames", "0"])

    def test_compute_application_fps(self) -> None:
        """Verify FPS calculation."""
        timestamps = [i * 0.05 for i in range(21)]  # 20 intervals of 0.05 = 1.0s => 20 FPS
        self.assertAlmostEqual(compute_application_fps(timestamps), 20.0)
        self.assertEqual(compute_application_fps([]), 0.0)
        self.assertEqual(compute_application_fps([1.0]), 0.0)

    @patch("perception.readiness.run_readiness_demo.initialize_camera")
    @patch("perception.readiness.run_readiness_demo.PersonDetector")
    @patch("perception.readiness.run_readiness_demo.FaceAnalyzer")
    def test_cleanup_on_camera_failure(
        self,
        mock_face: MagicMock,
        mock_det: MagicMock,
        mock_init_cam: MagicMock,
    ) -> None:
        """Resources must be cleanly released if camera initialization fails."""
        mock_det_instance = MagicMock()
        mock_det.return_value = mock_det_instance
        mock_face_instance = MagicMock()
        mock_face.return_value = mock_face_instance

        mock_init_cam.side_effect = RuntimeError("Camera open failed")

        args = parse_args(["--headless"])
        exit_code = run_readiness_pipeline(args)

        self.assertEqual(exit_code, 1)
        mock_det_instance.close.assert_called_once()
        mock_face_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

