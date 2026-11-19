"""Unit tests for M6 contact data types, event schemas, and predicates."""

import unittest

from perception.contact.contact_types import (
    ContactTimings,
    DistanceInfo,
    DistanceObservation,
    DistanceSource,
    FirstContactReadyEvent,
    FirstContactResult,
    ReadinessPredicates,
    ReadinessState,
    ReasonCode,
)
from perception.readiness.observation_types import ObservationStatus


class TestContactTypes(unittest.TestCase):
    """Test M6 data models, schemas, and immutability."""

    def test_readiness_state_values(self) -> None:
        """Verify the three required readiness state enum values."""
        self.assertEqual(ReadinessState.NOT_READY.value, "NOT_READY")
        self.assertEqual(ReadinessState.PENDING.value, "PENDING")
        self.assertEqual(ReadinessState.READY.value, "READY")

    def test_distance_source_values(self) -> None:
        """Verify MOCK and METRIC distance sources."""
        self.assertEqual(DistanceSource.MOCK.value, "MOCK")
        self.assertEqual(DistanceSource.METRIC.value, "METRIC")

    def test_distance_observation_validation(self) -> None:
        """Verify DistanceObservation validates distance and timestamp."""
        obs = DistanceObservation(
            distance_m=1.2,
            source=DistanceSource.MOCK,
            status=ObservationStatus.VALID,
            timestamp=10.0,
        )
        self.assertAlmostEqual(obs.distance_m, 1.2)
        self.assertEqual(obs.source, DistanceSource.MOCK)
        self.assertEqual(obs.status, ObservationStatus.VALID)

        # Reject negative distance
        with self.assertRaises(ValueError):
            DistanceObservation(
                distance_m=-0.5,
                source=DistanceSource.MOCK,
                status=ObservationStatus.VALID,
                timestamp=10.0,
            )

        # Reject non-finite distance
        with self.assertRaises(ValueError):
            DistanceObservation(
                distance_m=float("inf"),
                source=DistanceSource.MOCK,
                status=ObservationStatus.VALID,
                timestamp=10.0,
            )

    def test_first_contact_ready_event_schema(self) -> None:
        """Verify structured FIRST_CONTACT_READY event serialization."""
        evt = FirstContactReadyEvent(
            event_id="evt_12345",
            visit_id="visit_abc",
            track_id=1,
            timestamp=12.5,
            distance=DistanceInfo(value_m=1.2, source="MOCK"),
        )
        d = evt.to_dict()
        self.assertEqual(d["event_type"], "FIRST_CONTACT_READY")
        self.assertEqual(d["event_id"], "evt_12345")
        self.assertEqual(d["visit_id"], "visit_abc")
        self.assertEqual(d["track_id"], 1)
        self.assertEqual(d["timestamp"], 12.5)
        self.assertEqual(d["readiness_duration"], 2.0)
        self.assertEqual(d["distance"]["value_m"], 1.2)
        self.assertEqual(d["distance"]["source"], "MOCK")

    def test_readiness_predicates_all_valid(self) -> None:
        """Verify is_all_valid requires all seven predicates to be simultaneously True."""
        all_true = ReadinessPredicates(
            p_presence=True,
            b_bbox=True,
            a_approach=True,
            z_zone=True,
            d_distance=True,
            s_stationary=True,
            o_orientation=True,
        )
        self.assertTrue(all_true.is_all_valid)

        # Single condition false makes is_all_valid False
        for field_name in [
            "p_presence",
            "b_bbox",
            "a_approach",
            "z_zone",
            "d_distance",
            "s_stationary",
            "o_orientation",
        ]:
            kwargs = {f: True for f in [
                "p_presence", "b_bbox", "a_approach", "z_zone", "d_distance", "s_stationary", "o_orientation"
            ]}
            kwargs[field_name] = False
            preds = ReadinessPredicates(**kwargs)
            self.assertFalse(preds.is_all_valid, f"Expected is_all_valid=False when {field_name}=False")

    def test_reason_codes_defined(self) -> None:
        """Verify all required explainable reason codes are present."""
        expected = [
            "NO_PERSON", "INVALID_BBOX", "APPROACH_NOT_CONFIRMED",
            "OUTSIDE_INTERACTION_ZONE", "DISTANCE_UNAVAILABLE",
            "DISTANCE_UNRELIABLE", "TOO_FAR", "PERSON_MOVING",
            "NOT_FACING_ROBOT", "PERCEPTION_UNCERTAIN", "TRACKING_LOST",
            "STALE_OBSERVATION", "INVALID_TIMESTAMP", "TIMESTAMP_MISMATCH",
            "TARGET_MISMATCH", "PIPELINE_FAILURE",
        ]
        for name in expected:
            self.assertTrue(hasattr(ReasonCode, name), f"Missing ReasonCode: {name}")

    def test_distance_observation_track_id(self) -> None:
        """Verify DistanceObservation supports optional track_id field."""
        obs1 = DistanceObservation(
            distance_m=1.2,
            source=DistanceSource.MOCK,
            status=ObservationStatus.VALID,
            timestamp=10.0,
            track_id=5,
        )
        self.assertEqual(obs1.track_id, 5)

        obs2 = DistanceObservation(
            distance_m=1.2,
            source=DistanceSource.MOCK,
            status=ObservationStatus.VALID,
            timestamp=10.0,
        )
        self.assertIsNone(obs2.track_id)

    def test_first_contact_ready_event_timestamp_and_duration_validation(self) -> None:
        """Verify FirstContactReadyEvent rejects invalid or non-finite timestamps and durations."""
        dist = DistanceInfo(value_m=1.2, source="MOCK")

        # Reject negative timestamp
        with self.assertRaises(ValueError):
            FirstContactReadyEvent(
                event_id="e1", visit_id="v1", track_id=1, timestamp=-1.0, distance=dist
            )

        # Reject NaN timestamp
        with self.assertRaises(ValueError):
            FirstContactReadyEvent(
                event_id="e1", visit_id="v1", track_id=1, timestamp=float("nan"), distance=dist
            )

        # Reject inf timestamp
        with self.assertRaises(ValueError):
            FirstContactReadyEvent(
                event_id="e1", visit_id="v1", track_id=1, timestamp=float("inf"), distance=dist
            )

        # Reject zero or negative readiness duration
        with self.assertRaises(ValueError):
            FirstContactReadyEvent(
                event_id="e1",
                visit_id="v1",
                track_id=1,
                timestamp=10.0,
                distance=dist,
                readiness_duration=0.0,
            )

        with self.assertRaises(ValueError):
            FirstContactReadyEvent(
                event_id="e1",
                visit_id="v1",
                track_id=1,
                timestamp=10.0,
                distance=dist,
                readiness_duration=float("nan"),
            )

        # Verify utc_timestamp included in to_dict() if present
        evt_utc = FirstContactReadyEvent(
            event_id="e2",
            visit_id="v2",
            track_id=2,
            timestamp=15.0,
            distance=dist,
            utc_timestamp=1700000000.0,
        )
        d = evt_utc.to_dict()
        self.assertEqual(d["utc_timestamp"], 1700000000.0)


if __name__ == "__main__":
    unittest.main()

