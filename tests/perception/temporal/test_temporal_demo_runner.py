"""Unit tests for M5 temporal behaviour demonstration runner."""

from unittest.mock import MagicMock, patch
import unittest

import numpy as np

from perception.temporal.run_temporal_demo import parse_args, run_temporal_pipeline


class TestTemporalDemoRunner(unittest.TestCase):
    """Test CLI argument parsing and demo runner execution seams."""

    def test_default_cli_args(self) -> None:
        """Verify default CLI options for the temporal demo runner."""
        args = parse_args([])
        self.assertEqual(args.config, "configs/m4/stationary_readiness.yaml")
        self.assertEqual(args.temporal_config, "configs/m5/temporal_behaviour.yaml")
        self.assertIsNone(args.camera)
        self.assertIsNone(args.model)
        self.assertIsNone(args.conf)
        self.assertIsNone(args.max_frames)
        self.assertFalse(args.headless)

    def test_custom_cli_args(self) -> None:
        """Verify custom CLI arguments are parsed correctly."""
        args = parse_args([
            "--camera", "2",
            "--conf", "0.65",
            "--max-frames", "100",
            "--headless",
        ])
        self.assertEqual(args.camera, "2")
        self.assertEqual(args.conf, 0.65)
        self.assertEqual(args.max_frames, 100)
        self.assertTrue(args.headless)

    def test_reject_invalid_confidence(self) -> None:
        """Verify confidence outside [0.0, 1.0] causes parse error."""
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "1.5"])
        with self.assertRaises(SystemExit):
            parse_args(["--conf", "-0.1"])

    def test_reject_non_positive_max_frames(self) -> None:
        """Verify non-positive max-frames causes parse error."""
        with self.assertRaises(SystemExit):
            parse_args(["--max-frames", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--max-frames", "-10"])

    @patch("perception.temporal.run_temporal_demo.initialize_camera")
    @patch("perception.temporal.run_temporal_demo.PersonDetector")
    @patch("perception.temporal.run_temporal_demo.FaceAnalyzer")
    def test_camera_failure_releases_detector_and_analyzer(
        self,
        mock_face_cls: MagicMock,
        mock_detector_cls: MagicMock,
        mock_init_camera: MagicMock,
    ) -> None:
        """Verify camera open failure raises error and releases all initialized models."""
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_analyzer = MagicMock()
        mock_face_cls.return_value = mock_analyzer

        mock_init_camera.side_effect = RuntimeError("Camera device busy")

        args = parse_args(["--headless"])
        with self.assertRaises(RuntimeError):
            run_temporal_pipeline(args)

    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=-1)
    @patch("perception.temporal.run_temporal_demo.initialize_camera")
    @patch("perception.temporal.run_temporal_demo.PersonDetector")
    @patch("perception.temporal.run_temporal_demo.FaceAnalyzer")
    def test_pipeline_runs_n_frames_and_terminates(
        self,
        mock_face_cls: MagicMock,
        mock_detector_cls: MagicMock,
        mock_init_camera: MagicMock,
        mock_wait_key: MagicMock,
        mock_imshow: MagicMock,
    ) -> None:
        """Verify runner processes exactly max-frames and exits cleanly."""
        mock_detector = MagicMock()
        mock_detector.detect_raw.return_value = []
        mock_detector_cls.return_value = mock_detector

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = MagicMock(face_detected=False)
        mock_face_cls.return_value = mock_analyzer

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, dummy_frame)
        mock_init_camera.return_value = (mock_cap, 1280, 720, 30.0)

        args = parse_args(["--max-frames", "5", "--headless"])
        exit_code = run_temporal_pipeline(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_detector.detect_raw.call_count, 5)
        mock_cap.release.assert_called_once()
        mock_detector.close.assert_called_once()
        mock_analyzer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

