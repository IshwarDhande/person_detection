"""Unit tests for M6 eligibility continuous duration timer."""

from __future__ import annotations

import unittest

from perception.contact.eligibility_timer import EligibilityTimer


class TestEligibilityTimer(unittest.TestCase):
    """Test 2.0s continuous eligibility timing, reset semantics, and continuity limits."""

    def setUp(self) -> None:
        self.timer = EligibilityTimer(
            required_duration_seconds=2.0,
            max_inter_frame_gap_seconds=0.5,
        )

    def test_initial_state(self) -> None:
        """Verify new timer is not running, not complete, and zero elapsed."""
        self.assertFalse(self.timer.is_running)
        self.assertFalse(self.timer.is_complete)
        self.assertEqual(self.timer.elapsed_seconds, 0.0)
        self.assertIsNone(self.timer.start_time)

    def test_less_than_two_seconds(self) -> None:
        """Verify continuous eligibility for < 2.0s returns is_complete=False."""
        t0 = 10.0
        # 30 frames at ~30 FPS (dt = 0.0333s), total ~0.97s elapsed
        for i in range(30):
            t = t0 + i * 0.0333
            is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=t)
            self.assertFalse(is_comp)

        self.assertTrue(self.timer.is_running)
        self.assertFalse(self.timer.is_complete)
        self.assertAlmostEqual(elapsed, 29 * 0.0333, places=2)

    def test_exactly_two_seconds_with_continuous_frames(self) -> None:
        """Verify reaching 2.0s continuous validity with contiguous frames completes eligibility."""
        t0 = 100.0
        dt = 0.05  # 20 FPS (within 0.5s gap threshold)

        # Feed 40 frames: t = 100.00 to 101.95 (1.95s elapsed) -> not complete
        for i in range(40):
            is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=t0 + i * dt)
            self.assertFalse(is_comp)

        # 41st frame: t = 102.00 (2.00s elapsed) -> completes eligibility
        is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=t0 + 40 * dt)
        self.assertTrue(is_comp)
        self.assertAlmostEqual(elapsed, 2.00, places=2)
        self.assertTrue(self.timer.is_complete)

    def test_more_than_two_seconds(self) -> None:
        """Verify continuing eligibility beyond 2.0s remains complete."""
        t0 = 0.0
        dt = 0.1  # 10 FPS
        # 21 frames (2.0s) -> complete
        for i in range(21):
            self.timer.update(is_eligible=True, timestamp=t0 + i * dt)
        self.assertTrue(self.timer.is_complete)

        # Subsequent continuous frames stay complete
        is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=2.5)
        self.assertTrue(is_comp)
        self.assertAlmostEqual(elapsed, 2.5, places=2)

    def test_inter_frame_gap_below_limit_preserves_continuity(self) -> None:
        """Verify gap below max_inter_frame_gap_seconds (0.3s < 0.5s) preserves accumulated time."""
        self.timer.update(is_eligible=True, timestamp=10.0)
        self.timer.update(is_eligible=True, timestamp=10.2)
        # Gap of 0.3s (within 0.5s limit)
        is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=10.5)
        self.assertFalse(is_comp)
        self.assertAlmostEqual(elapsed, 0.5, places=2)

    def test_inter_frame_gap_above_limit_resets_eligibility(self) -> None:
        """Verify gap above max_inter_frame_gap_seconds (0.6s > 0.5s) resets eligibility."""
        self.timer.update(is_eligible=True, timestamp=10.0)
        self.timer.update(is_eligible=True, timestamp=10.4)
        self.assertAlmostEqual(self.timer.elapsed_seconds, 0.4, places=2)

        # Gap of 0.6s exceeds 0.5s limit: resets continuity and restarts from 11.0
        is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=11.0)
        self.assertFalse(is_comp)
        self.assertEqual(elapsed, 0.0)
        self.assertEqual(self.timer.start_time, 11.0)

    def test_large_timestamp_jump_cannot_complete_eligibility(self) -> None:
        """Verify two sparse frames with a large time gap cannot satisfy 2.0s eligibility."""
        # Frame 1 at t = 1.0s
        self.timer.update(is_eligible=True, timestamp=1.0)
        # Frame 2 at t = 10.0s (9.0s gap)
        is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=10.0)
        self.assertFalse(is_comp)
        self.assertEqual(elapsed, 0.0)
        self.assertEqual(self.timer.start_time, 10.0)

    def test_condition_failure_resets_timer(self) -> None:
        """Verify single condition failure resets accumulated time and timer state."""
        t0 = 50.0
        dt = 0.1
        # Accumulate 1.5s
        for i in range(16):
            self.timer.update(is_eligible=True, timestamp=t0 + i * dt)
        self.assertAlmostEqual(self.timer.elapsed_seconds, 1.5, places=2)

        # Frame 17 fails condition
        is_comp, elapsed = self.timer.update(is_eligible=False, timestamp=t0 + 16 * dt)
        self.assertFalse(is_comp)
        self.assertEqual(elapsed, 0.0)
        self.assertEqual(self.timer.elapsed_seconds, 0.0)
        self.assertFalse(self.timer.is_running)
        self.assertFalse(self.timer.is_complete)

    def test_no_preservation_of_accumulated_time_after_reset(self) -> None:
        """Verify after reset, previously accumulated time is not resumed."""
        dt = 0.1
        # Attempt 1: accumulate 1.0s then break
        for i in range(11):
            self.timer.update(is_eligible=True, timestamp=10.0 + i * dt)
        self.timer.update(is_eligible=False, timestamp=11.1)

        # Attempt 2: fresh start at t=20.0
        self.timer.update(is_eligible=True, timestamp=20.0)
        is_comp, elapsed = self.timer.update(is_eligible=True, timestamp=20.1)

        # Must report 0.1s elapsed, NOT 1.0s + 0.1s = 1.1s
        self.assertFalse(is_comp)
        self.assertAlmostEqual(elapsed, 0.1, places=2)

    def test_non_monotonic_timestamp_raises_and_resets(self) -> None:
        """Verify non-monotonic timestamp raises ValueError and resets timer."""
        self.timer.update(is_eligible=True, timestamp=10.0)
        self.timer.update(is_eligible=True, timestamp=10.1)

        # Decreasing timestamp
        with self.assertRaises(ValueError):
            self.timer.update(is_eligible=True, timestamp=9.5)

        # Timer must be reset
        self.assertEqual(self.timer.elapsed_seconds, 0.0)

    def test_non_finite_timestamp_raises_and_resets(self) -> None:
        """Verify NaN or Inf timestamp raises ValueError and resets timer."""
        self.timer.update(is_eligible=True, timestamp=10.0)
        with self.assertRaises(ValueError):
            self.timer.update(is_eligible=True, timestamp=float("nan"))
        self.assertEqual(self.timer.elapsed_seconds, 0.0)

        self.timer.update(is_eligible=True, timestamp=10.0)
        with self.assertRaises(ValueError):
            self.timer.update(is_eligible=True, timestamp=float("inf"))
        self.assertEqual(self.timer.elapsed_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
