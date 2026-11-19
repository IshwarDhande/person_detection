"""Unit tests for stability duration timer and cumulative noise tolerance."""

import unittest

from perception.readiness.observation_types import ObservationStatus
from perception.temporal.observation_types import StabilityValue
from perception.temporal.stability_timer import StabilityTimer


class TestStabilityTimer(unittest.TestCase):
    """Test StabilityTimer continuous timing, 200ms tolerance, and hard reset conditions."""

    def setUp(self) -> None:
        self.timer = StabilityTimer(
            required_duration_seconds=2.0,
            violation_tolerance_seconds=0.2,
        )

    def test_initial_state_not_started(self) -> None:
        """Verify timer begins in NOT_STARTED state."""
        self.assertEqual(self.timer.state, StabilityValue.NOT_STARTED)
        self.assertEqual(self.timer.elapsed_seconds, 0.0)
        self.assertEqual(self.timer.violation_seconds, 0.0)
        self.assertFalse(self.timer.in_excursion)

    def test_continuous_accumulation_to_complete(self) -> None:
        """Verify uninterrupted eligibility for 2.0s transitions to COMPLETE."""
        dt = 0.05  # 20 FPS
        t0 = 100.0

        for i in range(40):
            t = t0 + i * dt
            obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=t)
            if i < 39:
                self.assertEqual(obs.value, StabilityValue.ACCUMULATING)

        obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=t0 + 2.05)
        self.assertEqual(obs.value, StabilityValue.COMPLETE)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    # --- Correction 1 Deterministic Suite (10 Required Scenarios) ---

    def test_1_one_frame_excursion(self) -> None:
        """Scenario 1: One-frame excursion charges dt (0.05s) upon recovery."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)
        self.assertFalse(self.timer.in_excursion)

        # Ineligible at t=0.05
        obs1 = self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.05)
        self.assertTrue(self.timer.in_excursion)
        self.assertEqual(obs1.value, StabilityValue.ACCUMULATING)

        # Recovery at t=0.10: span = 0.10 - 0.05 = 0.05s
        obs2 = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.10)
        self.assertFalse(self.timer.in_excursion)
        self.assertAlmostEqual(obs2.violation_seconds, 0.05, places=4)
        self.assertEqual(obs2.value, StabilityValue.ACCUMULATING)

    def test_2_two_frame_excursion(self) -> None:
        """Scenario 2: Two-frame excursion charges 0.10s upon recovery."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)

        # Ineligible at 0.05 and 0.10
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.05)
        obs_mid = self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.10)
        self.assertTrue(self.timer.in_excursion)
        self.assertAlmostEqual(obs_mid.violation_seconds, 0.05, places=4)

        # Recovery at 0.15: span = 0.15 - 0.05 = 0.10s
        obs_rec = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.15)
        self.assertFalse(self.timer.in_excursion)
        self.assertAlmostEqual(obs_rec.violation_seconds, 0.10, places=4)
        self.assertEqual(obs_rec.value, StabilityValue.ACCUMULATING)

    def test_3_multiple_separate_excursions(self) -> None:
        """Scenario 3: Multiple separate excursions accumulate correctly."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)

        # Excursion 1: 0.05 -> 0.10 (0.05s)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.05)
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.10)
        self.assertAlmostEqual(self.timer.violation_seconds, 0.05, places=4)

        # Intervening eligible frame
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.15)

        # Excursion 2: 0.20 -> 0.25 (0.05s)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.20)
        obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.25)
        self.assertAlmostEqual(obs.violation_seconds, 0.10, places=4)
        self.assertEqual(obs.value, StabilityValue.ACCUMULATING)

    def test_4_excursion_followed_by_recovery(self) -> None:
        """Scenario 4: Excursion followed by recovery allows accumulation to continue."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)

        # Excursion 0.05 -> 0.20 (0.15s)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.05)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.10)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.15)
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.20)
        self.assertAlmostEqual(self.timer.violation_seconds, 0.15, places=4)

        # Continue eligible frames
        obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.50)
        self.assertAlmostEqual(obs.violation_seconds, 0.15, places=4)
        self.assertEqual(obs.value, StabilityValue.ACCUMULATING)

    def test_5_excursion_exactly_at_200ms(self) -> None:
        """Scenario 5: Excursion span exactly equal to 0.20s is permitted and does not reset."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)

        # Ineligible at 0.05, 0.10, 0.15, 0.20
        for t in [0.05, 0.10, 0.15, 0.20]:
            self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=t)

        # Recovery at t=0.25: excursion span = 0.25 - 0.05 = 0.20s (<= 0.20s)
        obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.25)
        self.assertAlmostEqual(obs.violation_seconds, 0.20, places=4)
        self.assertEqual(obs.value, StabilityValue.ACCUMULATING)

    def test_6_excursion_just_above_200ms(self) -> None:
        """Scenario 6: Excursion exceeding 200ms triggers immediate reset."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)

        for t in [0.05, 0.10, 0.15, 0.20]:
            self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=t)

        # Ineligible at t=0.26: active excursion span = 0.26 - 0.05 = 0.21s > 0.20s
        obs = self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.26)
        self.assertEqual(obs.value, StabilityValue.RESET)
        self.assertEqual(obs.elapsed_seconds, 0.0)
        self.assertEqual(obs.violation_seconds, 0.0)

    def test_7_repeated_excursions_cumulative_limit(self) -> None:
        """Scenario 7: Cumulative duration reaching limit is permitted; exceeding resets."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)

        # Excursion 1: 0.10 -> 0.20 (0.10s)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.10)
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.20)
        self.assertAlmostEqual(self.timer.violation_seconds, 0.10, places=4)

        # Excursion 2: 0.30 -> 0.40 (0.10s) -> cumulative 0.20s
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.30)
        obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.40)
        self.assertAlmostEqual(obs.violation_seconds, 0.20, places=4)
        self.assertEqual(obs.value, StabilityValue.ACCUMULATING)

        # Excursion 3: any additional violation exhausts budget
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.50)
        obs_fail = self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.55)
        self.assertEqual(obs_fail.value, StabilityValue.RESET)

    def test_8_recovery_after_valid_excursion(self) -> None:
        """Scenario 8: in_excursion flag clears on recovery and violation_seconds stabilizes."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.10)
        self.assertTrue(self.timer.in_excursion)

        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.18)
        self.assertFalse(self.timer.in_excursion)
        frozen_violation = self.timer.violation_seconds
        self.assertAlmostEqual(frozen_violation, 0.08, places=4)

        # Subsequent eligible frames do not alter violation_seconds
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.30)
        self.assertAlmostEqual(self.timer.violation_seconds, frozen_violation, places=4)

    def test_9_stability_completion_after_permitted_tolerance_consumption(self) -> None:
        """Scenario 9: Timer achieves COMPLETE after consuming permitted tolerance budget."""
        t0 = 0.0
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=t0)

        # Excursion consuming 0.10s (0.10 to 0.20)
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.10)
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.20)

        # Continue eligible up to 2.10s total elapsed time
        for i in range(5, 43):
            t = i * 0.05
            obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=t)

        self.assertEqual(obs.value, StabilityValue.COMPLETE)
        self.assertAlmostEqual(obs.violation_seconds, 0.10, places=4)
        self.assertGreaterEqual(obs.elapsed_seconds, 2.0)

    def test_10_correct_timer_reset_after_tolerance_exhaustion(self) -> None:
        """Scenario 10: After tolerance exhaustion reset, new accumulation can start freshly."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=0.0)

        # Exhaust tolerance
        self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.05)
        obs_reset = self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=0.30)
        self.assertEqual(obs_reset.value, StabilityValue.RESET)

        # New fresh attempt at t=1.00
        obs_start = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=1.00)
        self.assertEqual(obs_start.value, StabilityValue.ACCUMULATING)
        self.assertEqual(obs_start.violation_seconds, 0.0)
        self.assertEqual(obs_start.elapsed_seconds, 0.0)

    # --- Immediate Hard Reset & Completion Termination Tests ---

    def test_immediate_hard_reset(self) -> None:
        """Verify is_hard_reset=True immediately resets timer without consuming tolerance."""
        t0 = 20.0
        for i in range(15):
            self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=t0 + i * 0.05)

        self.assertEqual(self.timer.state, StabilityValue.ACCUMULATING)
        self.assertGreater(self.timer.elapsed_seconds, 0.5)

        # Hard reset event (e.g. tracking loss)
        obs = self.timer.update(is_eligible=False, is_hard_reset=True, timestamp=t0 + 0.8)
        self.assertEqual(obs.value, StabilityValue.RESET)
        self.assertEqual(obs.elapsed_seconds, 0.0)
        self.assertEqual(obs.violation_seconds, 0.0)

    def test_conditions_cease_after_complete_resets(self) -> None:
        """Verify when eligibility ceases after COMPLETE, timer transitions to RESET."""
        self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=10.0)
        obs = self.timer.update(is_eligible=True, is_hard_reset=False, timestamp=12.1)
        self.assertEqual(obs.value, StabilityValue.COMPLETE)

        obs2 = self.timer.update(is_eligible=False, is_hard_reset=False, timestamp=12.15)
        self.assertEqual(obs2.value, StabilityValue.RESET)


if __name__ == "__main__":
    unittest.main()
