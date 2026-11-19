"""Observation types and output contract for Milestone M5 Temporal Behaviour Estimation.

Defines enums, metric dataclasses, and output containers for temporal behaviour
estimation according to Milestone M5 specification.

Architecture & Output Schema:
+-----------------------------------------------------------------------------+
|                           M5 TEMPORAL OBSERVATION                           |
+-----------------------------------------------------------------------------+
|                                                                             |
|  TemporalObservation                                                        |
|  |                                                                          |
|  +-- timestamp: float (monotonic capture timestamp)                         |
|  +-- track_id: Optional[int] (M3 active track identifier)                   |
|  +-- observed: bool (True only when matched to real YOLO detection)         |
|  |                                                                          |
|  +-- approach: Observation[ApproachValue]                                   |
|  |   +-- value:      APPROACHING / NOT_APPROACHING / UNKNOWN                |
|  |   +-- confidence: None (deterministic geometric predicate)               |
|  |   +-- status:     VALID / UNCERTAIN / INVALID                            |
|  |                                                                          |
|  +-- movement: Observation[MovementValue]                                   |
|  |   +-- value:      MOVING / STATIONARY / UNKNOWN                          |
|  |   +-- confidence: None                                                   |
|  |   +-- status:     VALID / UNCERTAIN / INVALID                            |
|  |                                                                          |
|  +-- stability: StabilityObservation                                        |
|  |   +-- value:              NOT_STARTED / ACCUMULATING / COMPLETE / RESET   |
|  |   +-- elapsed_seconds:    float (accumulated stability duration)         |
|  |   +-- violation_seconds:  float (tolerated threshold excursions)         |
|  |   +-- status:             VALID / UNCERTAIN / INVALID                    |
|  |                                                                          |
|  +-- motion: MotionMetrics                                                  |
|  |   +-- velocity_px_s:                tuple[float, float] (vx, vy)         |
|  |   +-- speed_px_s:                   float (||v||_2 in px/s)              |
|  |   +-- displacement_px:              float (net displacement in 1s)       |
|  |   +-- distance_to_zone_centroid_px: float (distance to polygon centroid) |
|  |   +-- direction_cosine:             Optional[float] (cos theta)          |
|  |   +-- closing_speed_px_s:           Optional[float] (-b_d px/s)          |
|  |                                                                          |
|  +-- initialization_state: InitializationState                              |
|  |   +-- UNINITIALIZED / COLLECTING / READY / RESET                         |
|  +-- initialization_count: int (0 to 5)                                     |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from perception.readiness.observation_types import Observation, ObservationStatus


class ApproachValue(str, Enum):
    """Approach observation values."""

    APPROACHING = "APPROACHING"
    NOT_APPROACHING = "NOT_APPROACHING"
    UNKNOWN = "UNKNOWN"


class MovementValue(str, Enum):
    """Movement observation values."""

    MOVING = "MOVING"
    STATIONARY = "STATIONARY"
    UNKNOWN = "UNKNOWN"


class StabilityValue(str, Enum):
    """Stability timer observation values."""

    NOT_STARTED = "NOT_STARTED"
    ACCUMULATING = "ACCUMULATING"
    COMPLETE = "COMPLETE"
    RESET = "RESET"
    UNKNOWN = "UNKNOWN"


class InitializationState(str, Enum):
    """Temporal observation initialization state."""

    UNINITIALIZED = "UNINITIALIZED"
    COLLECTING = "COLLECTING"
    READY = "READY"
    RESET = "RESET"


@dataclass(frozen=True)
class TemporalTimings:
    """Detailed component execution latencies for temporal estimation."""

    history_ms: float
    motion_ms: float
    stationary_ms: float
    approach_ms: float
    stability_ms: float
    total_ms: float


@dataclass(frozen=True)
class MotionMetrics:
    """Supplementary image-space motion metrics.

    Attributes:
        velocity_xy_px_s: Estimated velocity vector (vx, vy) in pixels/second.
        speed_px_s: Magnitude of velocity vector in pixels/second.
        displacement_px: Net Euclidean displacement over the comparison window in pixels.
        distance_to_zone_centroid_px: Distance from person center to zone centroid in pixels.
        direction_cosine: Cosine of angle between velocity and direction to zone centroid.
        closing_speed_px_s: Rate of distance reduction towards zone centroid (-b_d) in px/s.
        history_count: Number of observations currently in the history window.
        history_duration_seconds: Time span between earliest and latest observations in history.
    """

    velocity_xy_px_s: Optional[tuple[float, float]] = None
    speed_px_s: Optional[float] = None
    displacement_px: Optional[float] = None
    distance_to_zone_centroid_px: Optional[float] = None
    direction_cosine: Optional[float] = None
    closing_speed_px_s: Optional[float] = None
    history_count: int = 0
    history_duration_seconds: float = 0.0


@dataclass(frozen=True)
class StabilityObservation:
    """Structured observation for temporal stability timing.

    Attributes:
        value: Current stability state (NOT_STARTED, ACCUMULATING, COMPLETE, RESET, UNKNOWN).
        elapsed_seconds: Elapsed continuous stability duration in seconds.
        violation_seconds: Cumulative duration of tolerated valid threshold excursions in seconds.
        status: Observation reliability status (VALID, UNCERTAIN, INVALID).
    """

    value: StabilityValue
    elapsed_seconds: float
    violation_seconds: float
    status: ObservationStatus


@dataclass(frozen=True)
class TemporalObservation:
    """Aggregated temporal behaviour observation for Milestone M5.

    Attributes:
        timestamp: Monotonic capture timestamp of the current frame in seconds.
        track_id: Active M3 track identifier, or None if no active track.
        observed: True if supported by a genuine YOLO detection; False if coasted/predicted.
        approach: Approach observation (value, confidence=None, status).
        movement: Movement/stationary observation (value, confidence=None, status).
        stability: Continuous stability timer observation.
        motion: Detailed image-space motion metrics.
        initialization_state: State of five-observation initialization.
        initialization_count: Number of consecutive valid observations collected (0 to 5).
    """

    timestamp: float
    track_id: Optional[int]
    observed: bool
    approach: Observation[ApproachValue]
    movement: Observation[MovementValue]
    stability: StabilityObservation
    motion: MotionMetrics
    initialization_state: InitializationState
    initialization_count: int

