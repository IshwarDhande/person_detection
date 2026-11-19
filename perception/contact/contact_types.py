"""Type definitions and data models for Milestone M6: Readiness-Based Single-Person First Contact.

Defines readiness states, distance observation models, structured first-contact events,
predicates, and explainable reason codes.

Architecture & State Machine:
+-----------------------------------------------------------------------------+
|                          M6 READINESS STATE MACHINE                         |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [NOT_READY]                                                                |
|       |                                                                     |
|       +--(All 7 Predicates Valid)--> [PENDING]                              |
|       ^                                 |                                   |
|       |                                 | (2.0s continuous valid time)      |
|       |                                 v                                   |
|       |                              [READY] (Emits FIRST_CONTACT_READY)    |
|       |                                 |                                   |
|       +--(Any Condition Invalid)--------+                                   |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Optional
import uuid

from perception.readiness.observation_types import ObservationStatus


class ReadinessState(str, Enum):
    """Externally visible readiness states for the single-person target."""

    NOT_READY = "NOT_READY"
    PENDING = "PENDING"
    READY = "READY"


class DistanceSource(str, Enum):
    """Source of the distance observation."""

    MOCK = "MOCK"
    METRIC = "METRIC"


class ReasonCode(str, Enum):
    """Deterministic, explainable reason codes for why readiness is not satisfied."""

    NO_PERSON = "NO_PERSON"
    INVALID_BBOX = "INVALID_BBOX"
    APPROACH_NOT_CONFIRMED = "APPROACH_NOT_CONFIRMED"
    OUTSIDE_INTERACTION_ZONE = "OUTSIDE_INTERACTION_ZONE"
    DISTANCE_UNAVAILABLE = "DISTANCE_UNAVAILABLE"
    DISTANCE_UNRELIABLE = "DISTANCE_UNRELIABLE"
    TOO_FAR = "TOO_FAR"
    PERSON_MOVING = "PERSON_MOVING"
    NOT_FACING_ROBOT = "NOT_FACING_ROBOT"
    PERCEPTION_UNCERTAIN = "PERCEPTION_UNCERTAIN"
    TRACKING_LOST = "TRACKING_LOST"
    STALE_OBSERVATION = "STALE_OBSERVATION"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    TIMESTAMP_MISMATCH = "TIMESTAMP_MISMATCH"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    PIPELINE_FAILURE = "PIPELINE_FAILURE"


@dataclass(frozen=True)
class DistanceObservation:
    """Distance observation consumed by M6 from an independent provider.

    Attributes:
        distance_m: Estimated or supplied distance in metres.
        source: Source type (MOCK or METRIC).
        status: Reliability status (VALID, UNCERTAIN, INVALID).
        timestamp: Associated frame timestamp.
        track_id: Optional tracked target identifier associated with this measurement.
    """

    distance_m: float
    source: DistanceSource
    status: ObservationStatus
    timestamp: float
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError(f"distance_m must be a non-negative finite float, got {self.distance_m}")
        if not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp}")


@dataclass(frozen=True)
class DistanceInfo:
    """Distance payload included in the FIRST_CONTACT_READY event."""

    value_m: float
    source: str


@dataclass(frozen=True)
class FirstContactReadyEvent:
    """Structured one-shot event emitted when the person achieves first-contact readiness.

    Attributes:
        event_type: Fixed event identifier "FIRST_CONTACT_READY".
        event_id: Unique event UUID string.
        visit_id: Identifier of the visit in which readiness was achieved.
        track_id: Active M3 track identifier.
        timestamp: Monotonic event/frame timestamp in seconds.
        readiness_duration: Completed continuous eligibility duration (typically 2.0s).
        distance: Distance measurement payload at event generation.
    """

    event_id: str
    visit_id: str
    track_id: int
    timestamp: float
    distance: DistanceInfo
    event_type: str = "FIRST_CONTACT_READY"
    readiness_duration: float = 2.0
    utc_timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp <= 0.0:
            raise ValueError(f"timestamp must be a positive finite float, got {self.timestamp}")
        if not math.isfinite(self.readiness_duration) or self.readiness_duration <= 0.0:
            raise ValueError(
                f"readiness_duration must be a positive finite float, got {self.readiness_duration}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert event to structured dictionary representation."""
        data: dict[str, Any] = {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "visit_id": self.visit_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "readiness_duration": self.readiness_duration,
            "distance": {
                "value_m": self.distance.value_m,
                "source": self.distance.source,
            },
        }
        if self.utc_timestamp is not None:
            data["utc_timestamp"] = self.utc_timestamp
        return data


@dataclass(frozen=True)
class ReadinessPredicates:
    """Evaluation status of the seven required readiness conditions:
    E = P & B & A & Z & D & S & O.

    Attributes:
        p_presence: Current-frame person presence is valid.
        b_bbox: Valid bounding box matched to active M3 track.
        a_approach: Confirmed approach OR reliable initial presence established.
        z_zone: Interaction zone membership (INSIDE + VALID).
        d_distance: Distance observation is valid and <= d_max.
        s_stationary: Movement state is STATIONARY + VALID.
        o_orientation: Head orientation is FACING_ROBOT + VALID.
    """

    p_presence: bool
    b_bbox: bool
    a_approach: bool
    z_zone: bool
    d_distance: bool
    s_stationary: bool
    o_orientation: bool

    @property
    def is_all_valid(self) -> bool:
        """True if all seven readiness predicates are simultaneously satisfied."""
        return (
            self.p_presence
            and self.b_bbox
            and self.a_approach
            and self.z_zone
            and self.d_distance
            and self.s_stationary
            and self.o_orientation
        )

    @property
    def P(self) -> bool:
        """Alias for p_presence."""
        return self.p_presence

    @property
    def B(self) -> bool:
        """Alias for b_bbox."""
        return self.b_bbox

    @property
    def A(self) -> bool:
        """Alias for a_approach."""
        return self.a_approach

    @property
    def Z(self) -> bool:
        """Alias for z_zone."""
        return self.z_zone

    @property
    def D(self) -> bool:
        """Alias for d_distance."""
        return self.d_distance

    @property
    def S(self) -> bool:
        """Alias for s_stationary."""
        return self.s_stationary

    @property
    def O(self) -> bool:
        """Alias for o_orientation."""
        return self.o_orientation


@dataclass(frozen=True)
class ContactTimings:
    """Execution latency breakdown in milliseconds for M6 decision operations."""

    predicate_eval_ms: float
    visit_management_ms: float
    state_machine_ms: float
    total_ms: float


@dataclass(frozen=True)
class FirstContactResult:
    """Complete output record produced by the M6 Readiness Engine for a single frame.

    Attributes:
        state: Current persistent readiness state (NOT_READY, PENDING, READY).
        event: FIRST_CONTACT_READY event if emitted on this frame, or None.
        predicates: Detailed status of all seven individual readiness predicates.
        reasons: Active reason codes explaining any unsatisfied conditions.
        visit_id: Current active visit identifier, or None if no active visit.
        track_id: Active M3 track identifier, or None if no active track.
        elapsed_eligibility: Accumulated continuous eligibility time in seconds.
        distance: Consumed distance observation for this frame, or None.
        timings: Decision latency instrumentation.
    """

    state: ReadinessState
    event: Optional[FirstContactReadyEvent]
    predicates: ReadinessPredicates
    reasons: list[ReasonCode]
    visit_id: Optional[str]
    track_id: Optional[int]
    elapsed_eligibility: float
    distance: Optional[DistanceObservation]
    timings: ContactTimings
