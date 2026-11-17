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

from perception.detection.run_detection_demo import open_camera_stream, parse_args


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

    def test_reject_invalid_confidence_arg(self) -> None:
        """Verify confidence outside [0.0, 1.0] causes parse error."""
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "-0.1"])
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "1.2"])

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


if __name__ == "__main__":
    unittest.main()
