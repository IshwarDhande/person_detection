"""Unit tests for M6 configuration loading and parameter validation."""

from pathlib import Path
import unittest

from perception.contact.contact_config import ContactConfig, load_contact_config


class TestContactConfig(unittest.TestCase):
    """Test ContactConfig validation rules and YAML loader."""

    def test_default_config_valid(self) -> None:
        """Verify default configuration passes validation."""
        cfg = ContactConfig()
        cfg.validate()
        self.assertAlmostEqual(cfg.distance_max_m, 1.5)
        self.assertEqual(cfg.distance_mode, "mock")
        self.assertAlmostEqual(cfg.mock_distance_m, 1.2)
        self.assertAlmostEqual(cfg.eligibility_duration_seconds, 2.0)
        self.assertAlmostEqual(cfg.tracking_loss_timeout_seconds, 5.0)

    def test_reject_non_positive_distance_max(self) -> None:
        """Verify non-positive distance_max_m raises ValueError."""
        with self.assertRaises(ValueError):
            ContactConfig(distance_max_m=0.0).validate()
        with self.assertRaises(ValueError):
            ContactConfig(distance_max_m=-1.0).validate()

    def test_reject_invalid_distance_mode(self) -> None:
        """Verify unknown distance_mode raises ValueError."""
        with self.assertRaises(ValueError):
            ContactConfig(distance_mode="lidar").validate()

    def test_reject_negative_mock_distance(self) -> None:
        """Verify negative mock_distance_m raises ValueError."""
        with self.assertRaises(ValueError):
            ContactConfig(mock_distance_m=-0.5).validate()

    def test_reject_non_positive_eligibility_duration(self) -> None:
        """Verify non-positive eligibility_duration_seconds raises ValueError."""
        with self.assertRaises(ValueError):
            ContactConfig(eligibility_duration_seconds=0.0).validate()

    def test_reject_non_positive_tracking_loss_timeout(self) -> None:
        """Verify non-positive tracking_loss_timeout_seconds raises ValueError."""
        with self.assertRaises(ValueError):
            ContactConfig(tracking_loss_timeout_seconds=-2.0).validate()

    def test_production_mode_rejects_mock_distance(self) -> None:
        """Verify require_metric_distance_in_production rejects distance_mode='mock'."""
        with self.assertRaises(ValueError):
            ContactConfig(
                distance_mode="mock",
                require_metric_distance_in_production=True,
            ).validate()

    def test_reject_non_positive_max_inter_frame_gap(self) -> None:
        """Verify non-positive max_inter_frame_gap_seconds raises ValueError."""
        with self.assertRaises(ValueError):
            ContactConfig(max_inter_frame_gap_seconds=0.0).validate()
        with self.assertRaises(ValueError):
            ContactConfig(max_inter_frame_gap_seconds=-0.5).validate()

    def test_reject_nan_and_inf_configuration_values(self) -> None:
        """Verify NaN and infinity are rejected across numerical configurations."""
        nan_val = float("nan")
        inf_val = float("inf")
        neg_inf = float("-inf")

        for val in (nan_val, inf_val, neg_inf):
            with self.assertRaises(ValueError):
                ContactConfig(distance_max_m=val).validate()
            with self.assertRaises(ValueError):
                ContactConfig(mock_distance_m=val).validate()
            with self.assertRaises(ValueError):
                ContactConfig(eligibility_duration_seconds=val).validate()
            with self.assertRaises(ValueError):
                ContactConfig(tracking_loss_timeout_seconds=val).validate()
            with self.assertRaises(ValueError):
                ContactConfig(observation_freshness_seconds=val).validate()
            with self.assertRaises(ValueError):
                ContactConfig(max_inter_frame_gap_seconds=val).validate()

    def test_yaml_config_loading(self) -> None:
        """Verify loading from configs/m6/first_contact.yaml."""
        yaml_path = Path("configs/m6/first_contact.yaml")
        if yaml_path.is_file():
            cfg = load_contact_config(yaml_path)
            self.assertIsInstance(cfg, ContactConfig)
            self.assertAlmostEqual(cfg.distance_max_m, 1.5)


if __name__ == "__main__":
    unittest.main()

