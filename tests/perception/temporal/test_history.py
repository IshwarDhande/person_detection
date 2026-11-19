"""Unit tests for ObservationHistory and temporal continuity management."""

import unittest

from perception.temporal.history import ObservationHistory
from perception.temporal.observation_types import InitializationState


class TestObservationHistory(unittest.TestCase):
    """Test ObservationHistory buffering, pruning, and initialization state machine."""

    def test_initial_state_is_uninitialized(self) -> None:
        """Verify new history buffer starts UNINITIALIZED with zero count."""
        h = ObservationHistory(history_window_seconds=1.0, initial_valid_observations=5)
        self.assertEqual(h.state, InitializationState.UNINITIALIZED)
        self.assertEqual(h.consecutive_valid_count, 0)
        self.assertFalse(h.is_ready)
        self.assertEqual(h.count, 0)

    def test_five_consecutive_observations_reach_ready(self) -> None:
        """Verify exactly 5 consecutive valid observations transition to READY."""
        h = ObservationHistory(
            history_window_seconds=1.0,
            max_observation_gap_seconds=0.2,
            initial_valid_observations=5,
        )

        # Observations 1 to 4: COLLECTING
        t0 = 100.0
        for i in range(4):
            accepted = h.add_observation(t0 + i * 0.033, (100.0 + i, 200.0))
            self.assertTrue(accepted)
            self.assertEqual(h.state, InitializationState.COLLECTING)
            self.assertEqual(h.consecutive_valid_count, i + 1)
            self.assertFalse(h.is_ready)

        # Observation 5: READY
        accepted = h.add_observation(t0 + 4 * 0.033, (104.0, 200.0))
        self.assertTrue(accepted)
        self.assertEqual(h.state, InitializationState.READY)
        self.assertEqual(h.consecutive_valid_count, 5)
        self.assertTrue(h.is_ready)

    def test_observations_pruned_outside_window(self) -> None:
        """Verify observations older than history_window_seconds are pruned."""
        h = ObservationHistory(history_window_seconds=0.5, max_observation_gap_seconds=0.1)

        t0 = 10.0
        for i in range(20):  # 20 frames at 30 FPS = ~0.63s
            h.add_observation(t0 + i * 0.033, (float(i), 50.0))

        # Oldest observation should be >= latest - 0.5
        latest_t = h.latest_timestamp
        self.assertIsNotNone(latest_t)
        earliest_t = h.observations[0][0]
        self.assertGreaterEqual(earliest_t, latest_t - 0.5)

    def test_temporal_gap_greater_than_200ms_triggers_reset(self) -> None:
        """Verify Delta t > 200ms causes continuity failure, resets history, and returns False."""
        h = ObservationHistory(
            history_window_seconds=1.0,
            max_observation_gap_seconds=0.2,
            initial_valid_observations=5,
        )

        # Add 5 frames to reach READY
        for i in range(5):
            h.add_observation(10.0 + i * 0.033, (10.0, 10.0))
        self.assertTrue(h.is_ready)

        # Next frame arrives after 0.25 seconds (> 0.20s gap)
        last_t = h.latest_timestamp
        accepted = h.add_observation(last_t + 0.25, (20.0, 20.0))

        # Should return False (discontinuity occurred)
        self.assertFalse(accepted)
        # Should be reset to COLLECTING with count=1 (fresh start)
        self.assertEqual(h.state, InitializationState.COLLECTING)
        self.assertEqual(h.consecutive_valid_count, 1)
        self.assertEqual(h.count, 1)
        self.assertFalse(h.is_ready)

    def test_non_monotonic_timestamp_raises_value_error(self) -> None:
        """Verify non-increasing timestamps raise ValueError."""
        h = ObservationHistory()
        h.add_observation(10.0, (100.0, 100.0))

        with self.assertRaises(ValueError):
            h.add_observation(10.0, (105.0, 100.0))  # Equal timestamp

        with self.assertRaises(ValueError):
            h.add_observation(9.5, (105.0, 100.0))  # Decreasing timestamp

    def test_non_finite_inputs_raise_value_error(self) -> None:
        """Verify non-finite timestamps or coordinates raise ValueError."""
        h = ObservationHistory()
        with self.assertRaises(ValueError):
            h.add_observation(float("nan"), (100.0, 100.0))
        with self.assertRaises(ValueError):
            h.add_observation(10.0, (float("inf"), 100.0))

    def test_manual_reset(self) -> None:
        """Verify manual reset clears buffer and count."""
        h = ObservationHistory()
        for i in range(5):
            h.add_observation(1.0 + i * 0.033, (10.0, 10.0))
        self.assertTrue(h.is_ready)

        h.reset()
        self.assertEqual(h.state, InitializationState.RESET)
        self.assertEqual(h.consecutive_valid_count, 0)
        self.assertEqual(h.count, 0)
        self.assertFalse(h.is_ready)


if __name__ == "__main__":
    unittest.main()

