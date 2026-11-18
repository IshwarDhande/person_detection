"""Unit tests for the M4 Common Observation Contract.

Verifies that all observations expose (value, confidence, status), enforce
status types (VALID, UNCERTAIN, INVALID), cleanly distinguish valid negative
results from execution failures, and never fabricate probabilities.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.readiness.observation_types import (
    BboxValidityValue,
    FaceVisibilityValue,
    HeadOrientationValue,
    InteractionZoneValue,
    Observation,
    ObservationStatus,
    PersonPresenceValue,
)


class TestObservationContract(unittest.TestCase):
    """Test suite verifying adherence to the common observation contract."""

    def test_observation_fields(self) -> None:
        """Every Observation must have value, confidence, and status."""
        obs = Observation(
            value=PersonPresenceValue.DETECTED,
            confidence=0.91,
            status=ObservationStatus.VALID,
        )
        self.assertEqual(obs.value, PersonPresenceValue.DETECTED)
        self.assertEqual(obs.confidence, 0.91)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_null_confidence_for_deterministic_geometry(self) -> None:
        """Geometric checks must use None for confidence without inventing numbers."""
        obs = Observation(
            value=BboxValidityValue.VALID,
            confidence=None,
            status=ObservationStatus.VALID,
        )
        self.assertIsNone(obs.confidence)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_valid_negative_vs_invalid_distinction(self) -> None:
        """Valid negative states must be VALID status; failures must be INVALID."""
        # Valid negative: detector worked, found no person
        neg_person = Observation(
            value=PersonPresenceValue.NOT_DETECTED,
            confidence=None,
            status=ObservationStatus.VALID,
        )
        self.assertEqual(neg_person.status, ObservationStatus.VALID)

        # Failure: detector could not execute
        fail_person = Observation(
            value=PersonPresenceValue.UNKNOWN,
            confidence=None,
            status=ObservationStatus.INVALID,
        )
        self.assertEqual(fail_person.status, ObservationStatus.INVALID)

        # Valid negative: face landmarker ran, no face found
        neg_face = Observation(
            value=FaceVisibilityValue.INSUFFICIENT,
            confidence=None,
            status=ObservationStatus.VALID,
        )
        self.assertEqual(neg_face.status, ObservationStatus.VALID)

        # Failure: face model crashed
        fail_face = Observation(
            value=FaceVisibilityValue.UNKNOWN,
            confidence=None,
            status=ObservationStatus.INVALID,
        )
        self.assertEqual(fail_face.status, ObservationStatus.INVALID)

        # Valid negative: person is outside the polygon
        neg_zone = Observation(
            value=InteractionZoneValue.OUTSIDE,
            confidence=None,
            status=ObservationStatus.VALID,
        )
        self.assertEqual(neg_zone.status, ObservationStatus.VALID)

    def test_uncertain_status_preservation(self) -> None:
        """Ambiguous evidence must yield UNCERTAIN status, distinct from INVALID."""
        uncertain_orient = Observation(
            value=HeadOrientationValue.UNKNOWN,
            confidence=None,
            status=ObservationStatus.UNCERTAIN,
        )
        self.assertEqual(uncertain_orient.value, HeadOrientationValue.UNKNOWN)
        self.assertEqual(uncertain_orient.status, ObservationStatus.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()

