"""Unit tests for M1 detection configuration and camera abstraction.

```text
+-----------------------------------------------------------------------------+
|                  UNIT TESTS: DETECTION CONFIG & CAMERA SEAMS                |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [TestDetectionConfigParsing]                                               |
|    - Verify default arguments match M1 specification                        |
|    - Verify custom argument parsing (--camera, --conf, --width, etc.)       |
|    - Verify invalid argument boundaries reject with SystemExit              |
|                                                                             |
|  [TestCameraStreamSeam]                                                     |
|    - Camera open failure -> RuntimeError (clear diagnostics, no hang)       |
|    - Camera initial frame read failure -> RuntimeError (releases camera)    |
|    - Successful negotiation -> returns (cap, w, h, fps)                     |
|                                                                             |
+-----------------------------------------------------------------------------+
```
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import unittest

# Ensure repository root is on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from perception.detection.run_detection_demo import (
    compute_application_fps,
    open_camera_stream,
    parse_args,
    run_detection_pipeline,
)


class TestDetectionConfigParsing(unittest.TestCase):
    """Test suite for command-line arguments and configuration validation."""

    def test_default_cli_arguments(self) -> None:
        """Verify default CLI arguments match M1 design specification."""
        args = parse_args([])
        self.assertEqual(args.camera, "/dev/video0")
        self.assertEqual(args.model, "yolov8n.pt")
        self.assertEqual(args.conf, 0.5)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertEqual(args.fps, 30)
        self.assertEqual(args.fourcc, "MJPG")
        self.assertIsNone(args.device)
        self.assertIsNone(args.max_frames)
        self.assertFalse(args.headless)
        self.assertAlmostEqual(args.ema_alpha, 0.65)
        self.assertFalse(args.no_smoothing)

    def test_custom_cli_arguments(self) -> None:
        """Verify custom CLI arguments are parsed and typed correctly."""
        custom_argv = [
            "--camera", "1",
            "--model", "custom_yolo.pt",
            "--conf", "0.75",
            "--width", "640",
            "--height", "480",
            "--fps", "15",
            "--fourcc", "YUYV",
            "--device", "cpu",
            "--max-frames", "100",
            "--headless",
            "--ema-alpha", "0.85",
            "--no-smoothing",
        ]
        args = parse_args(custom_argv)
        self.assertEqual(args.camera, "1")
        self.assertEqual(args.model, "custom_yolo.pt")
        self.assertEqual(args.conf, 0.75)
        self.assertEqual(args.width, 640)
        self.assertEqual(args.height, 480)
        self.assertEqual(args.fps, 15)
        self.assertEqual(args.fourcc, "YUYV")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.max_frames, 100)
        self.assertTrue(args.headless)
        self.assertAlmostEqual(args.ema_alpha, 0.85)
        self.assertTrue(args.no_smoothing)

    def test_reject_invalid_confidence_arg(self) -> None:
        """Verify confidence outside [0.0, 1.0] causes parse error."""
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "-0.1"])
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "1.2"])

    def test_reject_invalid_ema_alpha_arg(self) -> None:
        """Verify ema_alpha outside (0.0, 1.0] causes parse error."""
        with self.assertRaises(SystemExit):
            parse_args(["--ema-alpha", "0.0"])
        with self.assertRaises(SystemExit):
            parse_args(["--ema-alpha", "-0.2"])
        with self.assertRaises(SystemExit):
            parse_args(["--ema-alpha", "1.05"])

    def test_reject_non_positive_dimensions(self) -> None:
        """Verify non-positive width, height, or fps cause parse errors."""
        with self.assertRaises(SystemExit):
            parse_args(["--width", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--height", "-10"])
        with self.assertRaises(SystemExit):
            parse_args(["--fps", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--max-frames", "0"])


class TestCameraStreamSeam(unittest.TestCase):
    """Test suite for camera opening and initial frame validation seams."""

    @patch("cv2.VideoCapture")
    def test_camera_open_failure_raises_runtime_error(self, mock_vidcap: MagicMock) -> None:
        """Verify unopenable camera raises RuntimeError immediately without infinite loop."""
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_vidcap.return_value = mock_cap_instance

        with self.assertRaises(RuntimeError) as ctx:
            open_camera_stream("/dev/video_non_existent")

        self.assertIn("Failed to open video capture device", str(ctx.exception))

    @patch("cv2.VideoCapture")
    def test_initial_frame_read_failure_releases_camera(self, mock_vidcap: MagicMock) -> None:
        """Verify failed initial frame read raises RuntimeError and releases resources."""
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (False, None)
        mock_vidcap.return_value = mock_cap_instance

        with self.assertRaises(RuntimeError) as ctx:
            open_camera_stream("/dev/video0")

        self.assertIn("failed to yield a valid video frame", str(ctx.exception))
        mock_cap_instance.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_successful_camera_open(self, mock_vidcap: MagicMock) -> None:
        """Verify successful camera open returns capture object and negotiated dimensions."""
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap_instance.read.return_value = (True, test_frame)

        # Mock CAP_PROP returns
        def mock_get(prop: int) -> float:
            if prop == 3:  # CAP_PROP_FRAME_WIDTH
                return 1280.0
            if prop == 4:  # CAP_PROP_FRAME_HEIGHT
                return 720.0
            if prop == 5:  # CAP_PROP_FPS
                return 30.0
            return 0.0

        mock_cap_instance.get.side_effect = mock_get
        mock_vidcap.return_value = mock_cap_instance

        cap, w, h, fps = open_camera_stream("/dev/video0", width=1280, height=720, fps=30)
        self.assertEqual(w, 1280)
        self.assertEqual(h, 720)
        self.assertEqual(fps, 30.0)
        self.assertEqual(cap, mock_cap_instance)


class TestFPSComputation(unittest.TestCase):
    """Test suite for controlled steady-state application FPS calculation."""

    def test_empty_or_single_timestamp_returns_zero(self) -> None:
        """Verify fewer than 2 frames yields 0.0 FPS."""
        self.assertEqual(compute_application_fps([]), 0.0)
        self.assertEqual(compute_application_fps([100.0]), 0.0)

    def test_controlled_timestamps_yield_exact_fps(self) -> None:
        """Verify (N - 1) / (tau_N - tau_1) calculation with controlled monotonic timestamps."""
        # 4 timestamps spaced by 0.1s -> 3 intervals in 0.3s -> 10.0 FPS
        timestamps = [1.0, 1.1, 1.2, 1.3]
        self.assertAlmostEqual(compute_application_fps(timestamps), 10.0, places=5)

        # 31 timestamps spaced by 0.033333s (30 FPS) -> 30 intervals in 1.0s -> 30.0 FPS
        timestamps_30fps = [i * (1.0 / 30.0) for i in range(31)]
        self.assertAlmostEqual(compute_application_fps(timestamps_30fps), 30.0, places=4)

    def test_non_positive_elapsed_time_returns_zero(self) -> None:
        """Verify identical or reversed timestamps do not cause division by zero."""
        self.assertEqual(compute_application_fps([1.0, 1.0]), 0.0)
        self.assertEqual(compute_application_fps([1.0, 0.9]), 0.0)

    def test_explicit_frame_count_parameter(self) -> None:
        """Verify explicit frame_count N across first and last timestamp endpoints."""
        # 30 frames completed between tau_1=10.0 and tau_30=12.9 -> (30-1)/2.9 = 29/2.9 = 10.0 FPS
        self.assertAlmostEqual(
            compute_application_fps([10.0, 12.9], frame_count=30), 10.0, places=5
        )
        # 1 frame count yields 0.0 FPS
        self.assertEqual(compute_application_fps([10.0, 12.9], frame_count=1), 0.0)


class TestGuaranteedResourceCleanup(unittest.TestCase):
    """Test suite verifying robust resource cleanup under all execution paths."""

    @patch("perception.detection.run_detection_demo.open_camera_stream")
    @patch("perception.detection.run_detection_demo.PersonDetector")
    def test_detector_closed_when_camera_init_fails(
        self, mock_detector_cls: MagicMock, mock_open_camera: MagicMock
    ) -> None:
        """Verify detector is closed even when camera stream initialization fails."""
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance
        mock_open_camera.side_effect = RuntimeError("V4L2 camera busy")

        with self.assertRaises(RuntimeError) as ctx:
            run_detection_pipeline(camera="/dev/video0", headless=True)

        self.assertIn("V4L2 camera busy", str(ctx.exception))
        mock_detector_instance.close.assert_called_once()

    @patch("cv2.destroyAllWindows")
    @patch("perception.detection.run_detection_demo.open_camera_stream")
    @patch("perception.detection.run_detection_demo.PersonDetector")
    def test_resources_released_when_loop_crashes(
        self,
        mock_detector_cls: MagicMock,
        mock_open_camera: MagicMock,
        mock_destroy_windows: MagicMock,
    ) -> None:
        """Verify cap, detector, and windows are released when loop raises an unhandled error."""
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance
        mock_detector_instance.detect.side_effect = RuntimeError("Inference memory fault")

        mock_cap = MagicMock()
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_open_camera.return_value = (mock_cap, 1280, 720, 30.0)

        with self.assertRaises(RuntimeError) as ctx:
            run_detection_pipeline(camera="/dev/video0", headless=False)

        self.assertIn("Inference memory fault", str(ctx.exception))
        mock_cap.release.assert_called_once()
        mock_detector_instance.close.assert_called_once()
        mock_destroy_windows.assert_called_once()

    @patch("perception.detection.run_detection_demo.open_camera_stream")
    @patch("perception.detection.run_detection_demo.PersonDetector")
    def test_resources_released_on_normal_completion(
        self, mock_detector_cls: MagicMock, mock_open_camera: MagicMock
    ) -> None:
        """Verify clean release of cap and detector upon reaching max_frames."""
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance
        mock_detector_instance.detect.return_value = []

        mock_cap = MagicMock()
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_open_camera.return_value = (mock_cap, 1280, 720, 30.0)

        metrics = run_detection_pipeline(camera="/dev/video0", max_frames=3, headless=True)

        self.assertEqual(metrics["processed_frames"], 3)
        mock_cap.release.assert_called_once()
        mock_detector_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
