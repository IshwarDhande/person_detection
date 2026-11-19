"""Unit tests for M5 temporal observation types and schemas."""

import unittest

from perception.readiness.observation_types import Observation, ObservationStatus
from perception.temporal.observation_types import (
    ApproachValue,
    InitializationState,
    MotionMetrics,
    MovementValue,
    StabilityObservation,
    StabilityValue,
    TemporalObservation,
)


class TestObservationTypes(unittest.TestCase):
    """Test M5 temporal observation enums and dataclass schemas."""

    def test_enums_values(self) -> None:
        """Verify enum members match specification."""
        self.assertEqual(ApproachValue.APPROACHING.value, "APPROACHING")
        self.assertEqual(ApproachValue.NOT_APPROACHING.value, "NOT_APPROACHING")
        self.assertEqual(ApproachValue.UNKNOWN.value, "UNKNOWN")

        self.assertEqual(MovementValue.MOVING.value, "MOVING")
        self.assertEqual(MovementValue.STATIONARY.value, "STATIONARY")
        self.assertEqual(MovementValue.UNKNOWN.value, "UNKNOWN")

        self.assertEqual(StabilityValue.NOT_STARTED.value, "NOT_STARTED")
        self.assertEqual(StabilityValue.ACCUMULATING.value, "ACCUMULATING")
        self.assertEqual(StabilityValue.COMPLETE.value, "COMPLETE")
        self.assertEqual(StabilityValue.RESET.value, "RESET")
        self.assertEqual(StabilityValue.UNKNOWN.value, "UNKNOWN")

        self.assertEqual(InitializationState.UNINITIALIZED.value, "UNINITIALIZED")
        self.assertEqual(InitializationState.COLLECTING.value, "COLLECTING")
        self.assertEqual(InitializationState.READY.value, "READY")
        self.assertEqual(InitializationState.RESET.value, "RESET")

    def test_motion_metrics_defaults(self) -> None:
        """Verify MotionMetrics default fields and immutability."""
        m = MotionMetrics()
        self.assertIsNone(m.velocity_xy_px_s)
        self.assertIsNone(m.speed_px_s)
        self.assertIsNone(m.displacement_px)
        self.assertIsNone(m.distance_to_zone_centroid_px)
        self.assertIsNone(m.direction_cosine)
        self.assertIsNone(m.closing_speed_px_s)
        self.assertEqual(m.history_count, 0)
        self.assertEqual(m.history_duration_seconds, 0.0)

    def test_temporal_observation_schema(self) -> None:
        """Verify TemporalObservation container instantiation and contract fields."""
        app_obs = Observation(
            value=ApproachValue.APPROACHING,
            confidence=None,
            status=ObservationStatus.VALID,
        )
        mov_obs = Observation(
            value=MovementValue.STATIONARY,
            confidence=None,
            status=ObservationStatus.VALID,
        )
        stab_obs = StabilityObservation(
            value=StabilityValue.ACCUMULATING,
            elapsed_seconds=1.25,
            violation_seconds=0.05,
            status=ObservationStatus.VALID,
        )
        metrics = MotionMetrics(
            velocity_xy_px_s=(8.4, 0.0),
            speed_px_s=8.4,
            displacement_px=6.2,
            distance_to_zone_centroid_px=125.0,
            direction_cosine=0.91,
            closing_speed_px_s=18.2,
            history_count=15,
            history_duration_seconds=0.5,
        )

        obs = TemporalObservation(
            timestamp=123.456,
            track_id=1,
            observed=True,
            approach=app_obs,
            movement=mov_obs,
            stability=stab_obs,
            motion=metrics,
            initialization_state=InitializationState.READY,
            initialization_count=5,
        )

        self.assertEqual(obs.timestamp, 123.456)
        self.assertEqual(obs.track_id, 1)
        self.assertTrue(obs.observed)
        self.assertEqual(obs.approach.value, ApproachValue.APPROACHING)
        self.assertEqual(obs.movement.value, MovementValue.STATIONARY)
        self.assertEqual(obs.stability.value, StabilityValue.ACCUMULATING)
        self.assertEqual(obs.stability.elapsed_seconds, 1.25)
        self.assertEqual(obs.motion.speed_px_s, 8.4)
        self.assertEqual(obs.initialization_state, InitializationState.READY)
        self.assertEqual(obs.initialization_count, 5)


if __name__ == "__main__":
    unittest.main()

