"""Unit tests for stationary and movement classification."""

import unittest

from perception.readiness.observation_types import ObservationStatus
from perception.temporal.observation_types import MovementValue
from perception.temporal.stationary_estimator import StationaryEstimator


class TestStationaryEstimator(unittest.TestCase):
    """Test StationaryEstimator strict velocity and displacement criteria."""

    def setUp(self) -> None:
        self.estimator = StationaryEstimator(
            velocity_threshold_px_s=20.0,
            displacement_threshold_px=20.0,
        )

    def test_both_conditions_met_is_stationary(self) -> None:
        """Verify speed < 20 px/s AND disp < 20 px evaluates to STATIONARY."""
        obs = self.estimator.evaluate(
            speed_px_s=10.0,
            speed_status=ObservationStatus.VALID,
            displacement_px=12.0,
            displacement_status=ObservationStatus.VALID,
        )
        self.assertEqual(obs.value, MovementValue.STATIONARY)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_strict_boundary_conditions(self) -> None:
        """Verify strict inequality: 19.99 is stationary, 20.00 is moving."""
        # Just below thresholds: STATIONARY
        obs1 = self.estimator.evaluate(
            speed_px_s=19.99,
            speed_status=ObservationStatus.VALID,
            displacement_px=19.99,
            displacement_status=ObservationStatus.VALID,
        )
        self.assertEqual(obs1.value, MovementValue.STATIONARY)

        # Speed exactly at threshold: MOVING
        obs2 = self.estimator.evaluate(
            speed_px_s=20.0,
            speed_status=ObservationStatus.VALID,
            displacement_px=5.0,
            displacement_status=ObservationStatus.VALID,
        )
        self.assertEqual(obs2.value, MovementValue.MOVING)

        # Displacement exactly at threshold: MOVING
        obs3 = self.estimator.evaluate(
            speed_px_s=5.0,
            speed_status=ObservationStatus.VALID,
            displacement_px=20.0,
            displacement_status=ObservationStatus.VALID,
        )
        self.assertEqual(obs3.value, MovementValue.MOVING)

    def test_high_velocity_or_high_displacement_is_moving(self) -> None:
        """Verify high velocity or displacement produces MOVING."""
        obs = self.estimator.evaluate(
            speed_px_s=45.0,
            speed_status=ObservationStatus.VALID,
            displacement_px=50.0,
            displacement_status=ObservationStatus.VALID,
        )
        self.assertEqual(obs.value, MovementValue.MOVING)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_uncertain_inputs_produce_unknown_uncertain(self) -> None:
        """Verify UNCERTAIN speed or displacement produces UNKNOWN with UNCERTAIN status."""
        obs = self.estimator.evaluate(
            speed_px_s=5.0,
            speed_status=ObservationStatus.VALID,
            displacement_px=5.0,
            displacement_status=ObservationStatus.UNCERTAIN,  # e.g. < 1.0s window
        )
        self.assertEqual(obs.value, MovementValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.UNCERTAIN)

    def test_invalid_inputs_produce_unknown_invalid(self) -> None:
        """Verify INVALID inputs produce UNKNOWN with INVALID status."""
        obs = self.estimator.evaluate(
            speed_px_s=None,
            speed_status=ObservationStatus.INVALID,
            displacement_px=None,
            displacement_status=ObservationStatus.INVALID,
        )
        self.assertEqual(obs.value, MovementValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)


if __name__ == "__main__":
    unittest.main()

