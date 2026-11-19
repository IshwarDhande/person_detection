"""Unit tests for TemporalConfig validation and loading."""

import tempfile
import unittest

from perception.temporal.temporal_config import TemporalConfig, load_temporal_config


class TestTemporalConfig(unittest.TestCase):
    """Test TemporalConfig validation rules and YAML file loading."""

    def test_default_config_is_valid(self) -> None:
        """Verify default configuration passes validation."""
        cfg = TemporalConfig()
        cfg.validate()
        self.assertEqual(cfg.history_window_seconds, 1.0)
        self.assertEqual(cfg.max_observation_gap_seconds, 0.2)
        self.assertEqual(cfg.initial_valid_observations, 5)
        self.assertEqual(cfg.stationary_velocity_threshold_px_s, 20.0)
        self.assertEqual(cfg.stationary_displacement_threshold_px, 20.0)
        self.assertEqual(cfg.approach_confirmation_seconds, 0.5)
        self.assertEqual(cfg.stability_duration_seconds, 2.0)
        self.assertEqual(cfg.stability_violation_tolerance_seconds, 0.2)

    def test_reject_non_positive_history_window(self) -> None:
        """Verify non-positive history window raises ValueError."""
        cfg = TemporalConfig(history_window_seconds=0.0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_reject_non_positive_observation_gap(self) -> None:
        """Verify non-positive observation gap raises ValueError."""
        cfg = TemporalConfig(max_observation_gap_seconds=-0.1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_reject_negative_thresholds(self) -> None:
        """Verify negative velocity or displacement thresholds raise ValueError."""
        with self.assertRaises(ValueError):
            TemporalConfig(stationary_velocity_threshold_px_s=-1.0).validate()
        with self.assertRaises(ValueError):
            TemporalConfig(stationary_displacement_threshold_px=-5.0).validate()

    def test_reject_invalid_direction_cosine(self) -> None:
        """Verify direction cosine outside [-1.0, 1.0] raises ValueError."""
        with self.assertRaises(ValueError):
            TemporalConfig(approach_min_direction_cosine=1.2).validate()
        with self.assertRaises(ValueError):
            TemporalConfig(approach_min_direction_cosine=-1.5).validate()

    def test_reject_inconsistent_displacement_window(self) -> None:
        """Verify displacement window > history window raises ValueError."""
        with self.assertRaises(ValueError):
            TemporalConfig(
                history_window_seconds=1.0,
                stationary_displacement_window_seconds=1.5,
            ).validate()

    def test_reject_tolerance_greater_than_duration(self) -> None:
        """Verify stability tolerance >= duration raises ValueError."""
        with self.assertRaises(ValueError):
            TemporalConfig(
                stability_duration_seconds=2.0,
                stability_violation_tolerance_seconds=2.0,
            ).validate()

    def test_load_from_yaml_file(self) -> None:
        """Verify loading from YAML file."""
        content = """
temporal:
  history_window_seconds: 1.2
  max_observation_gap_seconds: 0.15
  initial_valid_observations: 4
  stationary_velocity_threshold_px_s: 18.0
  stationary_displacement_threshold_px: 15.0
  stationary_displacement_window_seconds: 1.0
  approach_confirmation_seconds: 0.4
  approach_min_direction_cosine: 0.6
  approach_min_closing_speed_px_s: 12.0
  stability_duration_seconds: 2.5
  stability_violation_tolerance_seconds: 0.25
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(content)
            temp_path = f.name

        cfg = load_temporal_config(temp_path)
        self.assertEqual(cfg.history_window_seconds, 1.2)
        self.assertEqual(cfg.max_observation_gap_seconds, 0.15)
        self.assertEqual(cfg.initial_valid_observations, 4)
        self.assertEqual(cfg.stationary_velocity_threshold_px_s, 18.0)
        self.assertEqual(cfg.approach_min_direction_cosine, 0.6)
        self.assertEqual(cfg.stability_duration_seconds, 2.5)


if __name__ == "__main__":
    unittest.main()

