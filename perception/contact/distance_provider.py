"""Distance provider interface and implementations for Milestone M6.

Consumes or generates distance observations conforming to Section 10 of
the M6 specification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from perception.contact.contact_config import ContactConfig
from perception.contact.contact_types import DistanceObservation, DistanceSource
from perception.readiness.observation_types import ObservationStatus


class DistanceProvider(ABC):
    """Abstract base class for distance providers consumed by M6."""

    @property
    def supports_target_association(self) -> bool:
        """True if this provider associates distance measurements with target track IDs."""
        return True

    @abstractmethod
    def get_distance(
        self,
        timestamp: float,
        bbox_xyxy: Optional[tuple[float, float, float, float]] = None,
        track_id: Optional[int] = None,
    ) -> DistanceObservation:
        """Retrieve distance observation for the current frame.

        Args:
            timestamp: Monotonic frame timestamp in seconds.
            bbox_xyxy: Optional observed bounding box of the active track.
            track_id: Optional active target track identifier.

        Returns:
            DistanceObservation instance.
        """


class MockDistanceProvider(DistanceProvider):
    """Deterministic mock distance provider for testing, replay, and laptop demonstrations.

    Explicitly tagged with DistanceSource.MOCK. Must never be presented as camera-derived distance.
    """

    def __init__(
        self,
        default_distance_m: float = 1.2,
        default_status: ObservationStatus = ObservationStatus.VALID,
    ) -> None:
        """Initialize mock distance provider.

        Args:
            default_distance_m: Distance in metres to supply.
            default_status: Observation status (VALID, UNCERTAIN, INVALID).
        """
        if default_distance_m < 0.0:
            raise ValueError(f"default_distance_m must be non-negative, got {default_distance_m}")
        self._distance_m: float = float(default_distance_m)
        self._status: ObservationStatus = default_status

    @property
    def distance_m(self) -> float:
        """Current mock distance value in metres."""
        return self._distance_m

    @distance_m.setter
    def distance_m(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"distance_m must be non-negative, got {value}")
        self._distance_m = float(value)

    @property
    def status(self) -> ObservationStatus:
        """Current observation status."""
        return self._status

    @status.setter
    def status(self, value: ObservationStatus) -> None:
        self._status = value

    def get_distance(
        self,
        timestamp: float,
        bbox_xyxy: Optional[tuple[float, float, float, float]] = None,
        track_id: Optional[int] = None,
    ) -> DistanceObservation:
        """Return configured mock distance observation."""
        return DistanceObservation(
            distance_m=self._distance_m,
            source=DistanceSource.MOCK,
            status=self._status,
            timestamp=timestamp,
            track_id=track_id,
        )


class MetricDistanceProvider(DistanceProvider):
    """Interface for validated camera-derived or sensor-derived metric distance.

    Tagged with DistanceSource.METRIC. Used for calibrated physical robot operations.
    When uncalibrated or no sensor is connected, returns status=INVALID fail-closed.
    """

    def __init__(
        self,
        is_calibrated: bool = False,
    ) -> None:
        """Initialize metric distance provider."""
        self._is_calibrated = is_calibrated

    def get_distance(
        self,
        timestamp: float,
        bbox_xyxy: Optional[tuple[float, float, float, float]] = None,
        track_id: Optional[int] = None,
    ) -> DistanceObservation:
        """Retrieve camera-derived metric distance.

        Fails closed with INVALID if metric calibration or sensor is unavailable.
        """
        return DistanceObservation(
            distance_m=0.0,
            source=DistanceSource.METRIC,
            status=ObservationStatus.INVALID,
            timestamp=timestamp,
            track_id=track_id,
        )


def create_distance_provider(config: ContactConfig) -> DistanceProvider:
    """Instantiate appropriate DistanceProvider according to configuration."""
    config.validate()
    mode = config.distance_mode.strip().lower()

    if mode == "mock":
        return MockDistanceProvider(
            default_distance_m=config.mock_distance_m,
            default_status=ObservationStatus.VALID,
        )
    elif mode == "metric":
        return MetricDistanceProvider(is_calibrated=False)
    else:
        raise ValueError(f"Unknown distance mode: '{config.distance_mode}'")

