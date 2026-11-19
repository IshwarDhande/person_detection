"""Unit tests for M6 distance provider implementations and factory."""

import unittest

from perception.contact.contact_config import ContactConfig
from perception.contact.contact_types import DistanceSource
from perception.contact.distance_provider import (
    MetricDistanceProvider,
    MockDistanceProvider,
    create_distance_provider,
)
from perception.readiness.observation_types import ObservationStatus


class TestDistanceProvider(unittest.TestCase):
    """Test MockDistanceProvider, MetricDistanceProvider, and factory."""

    def test_mock_distance_provider_defaults(self) -> None:
        """Verify MockDistanceProvider supplies configured distance and MOCK tag."""
        provider = MockDistanceProvider(default_distance_m=1.2)
        obs = provider.get_distance(timestamp=10.0)
        self.assertAlmostEqual(obs.distance_m, 1.2)
        self.assertEqual(obs.source, DistanceSource.MOCK)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertEqual(obs.timestamp, 10.0)

    def test_mock_distance_provider_mutation(self) -> None:
        """Verify dynamic mutation of mock distance and status."""
        provider = MockDistanceProvider(default_distance_m=1.0)
        provider.distance_m = 2.5
        provider.status = ObservationStatus.UNCERTAIN

        obs = provider.get_distance(timestamp=20.0)
        self.assertAlmostEqual(obs.distance_m, 2.5)
        self.assertEqual(obs.status, ObservationStatus.UNCERTAIN)

        with self.assertRaises(ValueError):
            provider.distance_m = -1.0

    def test_metric_distance_provider_uncalibrated_fails_closed(self) -> None:
        """Verify MetricDistanceProvider without sensor returns status=INVALID fail-closed."""
        provider = MetricDistanceProvider(is_calibrated=False)
        obs = provider.get_distance(timestamp=15.0)
        self.assertEqual(obs.source, DistanceSource.METRIC)
        self.assertEqual(obs.status, ObservationStatus.INVALID)

    def test_create_distance_provider_factory(self) -> None:
        """Verify factory creates appropriate provider for configured mode."""
        cfg_mock = ContactConfig(distance_mode="mock", mock_distance_m=1.4)
        p_mock = create_distance_provider(cfg_mock)
        self.assertIsInstance(p_mock, MockDistanceProvider)
        obs_m = p_mock.get_distance(timestamp=5.0)
        self.assertAlmostEqual(obs_m.distance_m, 1.4)

        cfg_metric = ContactConfig(distance_mode="metric")
        p_metric = create_distance_provider(cfg_metric)
        self.assertIsInstance(p_metric, MetricDistanceProvider)


if __name__ == "__main__":
    unittest.main()

