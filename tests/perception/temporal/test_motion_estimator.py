"""Unit tests for linear regression velocity and displacement estimation."""

import unittest

from perception.readiness.observation_types import ObservationStatus
from perception.temporal.history import ObservationHistory
from perception.temporal.motion_estimator import MotionEstimator


class TestMotionEstimator(unittest.TestCase):
    """Test MotionEstimator least-squares velocity and displacement calculations."""

    def test_unready_history_returns_uncertain(self) -> None:
        """Verify uninitialized or < 2 observations history returns UNCERTAIN."""
        h = ObservationHistory(initial_valid_observations=5)
        estimator = MotionEstimator()

        # 0 observations
        v, speed, status = estimator.estimate_velocity(h)
        self.assertIsNone(v)
        self.assertEqual(status, ObservationStatus.UNCERTAIN)

        disp, disp_status = estimator.estimate_displacement(h)
        self.assertIsNone(disp)
        self.assertEqual(disp_status, ObservationStatus.UNCERTAIN)

        # 3 observations (< 5 needed for READY)
        for i in range(3):
            h.add_observation(10.0 + i * 0.033, (100.0 + i, 200.0))

        v, speed, status = estimator.estimate_velocity(h)
        self.assertIsNone(v)
        self.assertEqual(status, ObservationStatus.UNCERTAIN)

    def test_constant_velocity_motion(self) -> None:
        """Verify constant horizontal movement vx=100 px/s, vy=0 px/s."""
        h = ObservationHistory(history_window_seconds=1.0, initial_valid_observations=5)
        estimator = MotionEstimator()

        # Simulate 30 frames at 30 FPS moving at vx = 100 px/s
        dt = 1.0 / 30.0
        t0 = 50.0
        for i in range(30):
            t = t0 + i * dt
            cx = 100.0 + 100.0 * (i * dt)
            cy = 200.0
            h.add_observation(t, (cx, cy))

        self.assertTrue(h.is_ready)
        v, speed, status = estimator.estimate_velocity(h)

        self.assertEqual(status, ObservationStatus.VALID)
        self.assertIsNotNone(v)
        self.assertIsNotNone(speed)
        vx, vy = v
        self.assertAlmostEqual(vx, 100.0, places=2)
        self.assertAlmostEqual(vy, 0.0, places=2)
        self.assertAlmostEqual(speed, 100.0, places=2)

    def test_zero_velocity_stationary(self) -> None:
        """Verify stationary observations produce ~0 px/s velocity."""
        h = ObservationHistory(history_window_seconds=1.0, initial_valid_observations=5)
        estimator = MotionEstimator()

        dt = 1.0 / 30.0
        for i in range(30):
            h.add_observation(10.0 + i * dt, (320.0, 240.0))

        v, speed, status = estimator.estimate_velocity(h)
        self.assertEqual(status, ObservationStatus.VALID)
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v[0], 0.0, places=3)
        self.assertAlmostEqual(v[1], 0.0, places=3)
        self.assertAlmostEqual(speed, 0.0, places=3)

    def test_displacement_window_coverage(self) -> None:
        """Verify displacement status is UNCERTAIN until ~1.0s is covered, then VALID."""
        h = ObservationHistory(history_window_seconds=1.5, initial_valid_observations=5)
        estimator = MotionEstimator(min_displacement_window_seconds=1.0, displacement_tolerance_seconds=0.05)

        # 10 frames = ~0.3s -> duration < 0.95s -> UNCERTAIN
        for i in range(10):
            h.add_observation(1.0 + i * 0.033, (100.0 + i * 2.0, 100.0))

        disp, status = estimator.estimate_displacement(h)
        self.assertIsNotNone(disp)
        self.assertEqual(status, ObservationStatus.UNCERTAIN)

        # Add frames up to 31 frames = 1.0s (30 intervals of 0.0333s = ~1.0s)
        for i in range(10, 31):
            h.add_observation(1.0 + i * 0.0333333, (100.0 + i * 2.0, 100.0))

        disp, status = estimator.estimate_displacement(h)
        self.assertIsNotNone(disp)
        self.assertEqual(status, ObservationStatus.VALID)
        # Expected displacement: 30 * 2.0 = 60.0 px
        self.assertAlmostEqual(disp, 60.0, places=1)


if __name__ == "__main__":
    unittest.main()

