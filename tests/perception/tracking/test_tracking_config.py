"""Tests for Tracking CLI Argument Parsing and Demo Seams (Milestone M3)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.tracking.run_tracking_demo import (
    DEFAULT_CAMERA_DEVICE,
    DEFAULT_CAPTURE_FPS,
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_WIDTH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_RETENTION_TIMEOUT,
    compute_application_fps,
    parse_args,
    run_tracking_pipeline,
)


class TestTrackingConfigParsing(unittest.TestCase):
    """Test CLI argument parsing and validation for tracking demo runner."""

    def test_default_cli_arguments(self) -> None:
        """Verify default CLI arguments match M3 design specification."""
        args = parse_args([])
        self.assertEqual(args.camera, DEFAULT_CAMERA_DEVICE)
        self.assertEqual(args.model, DEFAULT_MODEL_PATH)
        self.assertEqual(args.conf, DEFAULT_CONFIDENCE_THRESHOLD)
        self.assertEqual(args.iou_thresh, DEFAULT_IOU_THRESHOLD)
        self.assertEqual(args.timeout, DEFAULT_RETENTION_TIMEOUT)
        self.assertEqual(args.width, DEFAULT_CAPTURE_WIDTH)
        self.assertEqual(args.height, DEFAULT_CAPTURE_HEIGHT)
        self.assertEqual(args.fps, DEFAULT_CAPTURE_FPS)
        self.assertIsNone(args.max_frames)
        self.assertFalse(args.headless)

    def test_custom_cli_arguments(self) -> None:
        """Verify custom CLI arguments are parsed and typed correctly."""
        args = parse_args([
            "--camera", "/dev/video2",
            "--model", "custom_weights.pt",
            "--conf", "0.65",
            "--iou-thresh", "0.45",
            "--timeout", "3.5",
            "--width", "640",
            "--height", "480",
            "--fps", "15",
            "--max-frames", "100",
            "--headless",
        ])
        self.assertEqual(args.camera, "/dev/video2")
        self.assertEqual(args.model, "custom_weights.pt")
        self.assertAlmostEqual(args.conf, 0.65)
        self.assertAlmostEqual(args.iou_thresh, 0.45)
        self.assertAlmostEqual(args.timeout, 3.5)
        self.assertEqual(args.width, 640)
        self.assertEqual(args.height, 480)
        self.assertEqual(args.fps, 15)
        self.assertEqual(args.max_frames, 100)
        self.assertTrue(args.headless)

    def test_reject_invalid_iou_threshold(self) -> None:
        """Verify iou_thresh outside (0.0, 1.0] raises error."""
        with self.assertRaises(SystemExit):
            parse_args(["--iou-thresh", "0.0"])

        with self.assertRaises(SystemExit):
            parse_args(["--iou-thresh", "-0.2"])

        with self.assertRaises(SystemExit):
            parse_args(["--iou-thresh", "1.1"])

    def test_reject_invalid_timeout(self) -> None:
        """Verify non-positive timeout raises error."""
        with self.assertRaises(SystemExit):
            parse_args(["--timeout", "0.0"])

        with self.assertRaises(SystemExit):
            parse_args(["--timeout", "-1.5"])

    def test_compute_application_fps_steady_state(self) -> None:
        """Verify application FPS computation from timestamps."""
        # 10 timestamps separated by 0.1s = 9 intervals of 0.1s = 0.9s elapsed => (10 - 1) / 0.9 = 10.0 FPS
        timestamps = [i * 0.1 for i in range(10)]
        self.assertAlmostEqual(compute_application_fps(timestamps), 10.0)

        # Single timestamp yields 0.0 FPS
        self.assertEqual(compute_application_fps([1.0]), 0.0)


class TestTrackingPipelineResourceCleanup(unittest.TestCase):
    """Test guaranteed resource release on normal and exceptional paths."""

    @patch("perception.tracking.run_tracking_demo.initialize_camera")
    @patch("perception.tracking.run_tracking_demo.PersonDetector")
    def test_detector_closed_if_camera_init_fails(
        self, mock_detector_cls: MagicMock, mock_init_cam: MagicMock
    ) -> None:
        """Verify PersonDetector is safely closed if camera initialization raises RuntimeError."""
        mock_detector_inst = MagicMock()
        mock_detector_cls.return_value = mock_detector_inst
        mock_init_cam.side_effect = RuntimeError("Camera open failed")

        args = parse_args(["--headless"])
        exit_code = run_tracking_pipeline(args)

        self.assertEqual(exit_code, 1)
        mock_detector_inst.close.assert_called_once()

    @patch("perception.tracking.run_tracking_demo.initialize_camera")
    @patch("perception.tracking.run_tracking_demo.PersonDetector")
    def test_resources_released_when_loop_crashes(
        self, mock_detector_cls: MagicMock, mock_init_cam: MagicMock
    ) -> None:
        """Verify camera, detector, and tracker are released when processing loop raises error."""
        mock_detector_inst = MagicMock()
        mock_detector_cls.return_value = mock_detector_inst

        mock_cap = MagicMock()
        mock_cap.read.side_effect = RuntimeError("Frame acquisition crash")
        mock_init_cam.return_value = (mock_cap, 1280, 720, 30.0)

        args = parse_args(["--headless"])
        exit_code = run_tracking_pipeline(args)

        self.assertEqual(exit_code, 1)
        mock_cap.release.assert_called_once()
        mock_detector_inst.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
