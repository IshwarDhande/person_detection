"""Unit tests for approach behaviour estimation towards interaction-zone centroid."""

import unittest

from perception.readiness.observation_types import ObservationStatus
from perception.temporal.approach_estimator import ApproachEstimator
from perception.temporal.history import ObservationHistory
from perception.temporal.observation_types import ApproachValue


class TestApproachEstimator(unittest.TestCase):
    """Test ApproachEstimator directional cosine, closing speed, and 500ms confirmation."""

    def setUp(self) -> None:
        self.estimator = ApproachEstimator(
            min_direction_cosine=0.5,
            min_closing_speed_px_s=10.0,
            confirmation_seconds=0.5,
        )
        self.zone_centroid = (500.0, 500.0)

    def test_direct_approach_with_confirmation(self) -> None:
        """Verify person moving directly towards zone centroid confirms APPROACHING after 500ms."""
        h = ObservationHistory(history_window_seconds=1.0, initial_valid_observations=5)

        # Move from (100, 500) towards (500, 500) at 100 px/s along x-axis
        # Centroid is at (500, 500), so vector r is positive x. Velocity vx is +100.
        # cos(theta) = 1.0 > 0.5. Closing speed = 100 > 10.
        dt = 0.05  # 20 FPS
        t0 = 10.0

        # Feed 10 frames (up to t=10.45s): h becomes ready at t=10.20s.
        # Elapsed confirmation at t=10.45 is 0.25s (< 0.50s) -> UNKNOWN [UNCERTAIN]
        last_obs = None
        for i in range(10):
            t = t0 + i * dt
            cx = 100.0 + 100.0 * (i * dt)
            h.add_observation(t, (cx, 500.0))

            if h.is_ready:
                obs, cosine, close_spd, dist = self.estimator.evaluate(
                    history=h,
                    zone_centroid=self.zone_centroid,
                    velocity_xy_px_s=(100.0, 0.0),
                    speed_px_s=100.0,
                    velocity_status=ObservationStatus.VALID,
                    timestamp=t,
                )
                self.assertIsNotNone(cosine)
                self.assertAlmostEqual(cosine, 1.0, places=3)
                self.assertEqual(obs.value, ApproachValue.UNKNOWN)
                self.assertEqual(obs.status, ObservationStatus.UNCERTAIN)
                last_obs = obs

        # Feed frames up to i=15 (t=10.75s): elapsed from t=10.20 is 0.55s (>= 0.50s)
        for i in range(10, 16):
            t = t0 + i * dt
            cx = 100.0 + 100.0 * (i * dt)
            h.add_observation(t, (cx, 500.0))

            obs, cosine, close_spd, dist = self.estimator.evaluate(
                history=h,
                zone_centroid=self.zone_centroid,
                velocity_xy_px_s=(100.0, 0.0),
                speed_px_s=100.0,
                velocity_status=ObservationStatus.VALID,
                timestamp=t,
            )

        # Should now be confirmed APPROACHING [VALID]
        self.assertEqual(obs.value, ApproachValue.APPROACHING)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_moving_away_evaluates_not_approaching(self) -> None:
        """Verify person moving away from centroid evaluates NOT_APPROACHING."""
        h = ObservationHistory(initial_valid_observations=5)
        # Move from (600, 500) away to the right: vx = +100, rx = 500 - 600 = -100
        # cos(theta) = -1.0. Closing speed is negative.
        dt = 0.05
        t0 = 20.0
        for i in range(10):
            t = t0 + i * dt
            cx = 600.0 + 100.0 * (i * dt)
            h.add_observation(t, (cx, 500.0))

        obs, cosine, close_spd, dist = self.estimator.evaluate(
            history=h,
            zone_centroid=self.zone_centroid,
            velocity_xy_px_s=(100.0, 0.0),
            speed_px_s=100.0,
            velocity_status=ObservationStatus.VALID,
            timestamp=t0 + 9 * dt,
        )

        self.assertIsNotNone(cosine)
        self.assertAlmostEqual(cosine, -1.0, places=3)
        self.assertEqual(obs.value, ApproachValue.NOT_APPROACHING)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_lateral_movement_evaluates_not_approaching(self) -> None:
        """Verify lateral movement perpendicular to centroid evaluates NOT_APPROACHING."""
        h = ObservationHistory(initial_valid_observations=5)
        # Person at x=100 moving vertically from cy=455 to cy=500.
        # Centroid is at (500, 500). At final frame, person is at (100, 500).
        # Vector r = z - c_t = (400, 0). Movement velocity v = (0, 100).
        # Dot product v . r = 0.0 -> cos(theta) = 0.0 < 0.5.
        dt = 0.05
        t0 = 30.0
        for i in range(10):
            t = t0 + i * dt
            cy = 455.0 + 5.0 * i
            h.add_observation(t, (100.0, cy))

        obs, cosine, close_spd, dist = self.estimator.evaluate(
            history=h,
            zone_centroid=self.zone_centroid,
            velocity_xy_px_s=(0.0, 100.0),
            speed_px_s=100.0,
            velocity_status=ObservationStatus.VALID,
            timestamp=t0 + 9 * dt,
        )

        self.assertIsNotNone(cosine)
        self.assertAlmostEqual(cosine, 0.0, places=2)
        self.assertEqual(obs.value, ApproachValue.NOT_APPROACHING)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_reset_clears_confirmation_state(self) -> None:
        """Verify reset clears ongoing approach confirmation."""
        self.estimator._approach_start_time = 123.0
        self.estimator.reset()
        self.assertIsNone(self.estimator._approach_start_time)


if __name__ == "__main__":
    unittest.main()
