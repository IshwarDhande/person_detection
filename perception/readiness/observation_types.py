"""Observation data types for M4 frame-level readiness perception.

Defines enums for observation values and statuses, and dataclasses for
structured observation output following the common observation contract.

Common observation contract:
    Every observation exposes three fields:
    - value:      Estimated property value (enum member).
    - confidence: Model-provided confidence (float or None).
    - status:     Evaluation validity (VALID, UNCERTAIN, INVALID).

Output structure:
    FrameObservation
    |
    +-- person
    |   +-- presence     (Observation[PersonPresenceValue])
    |   +-- bounding_box (Observation[BboxValidityValue] + BboxMetrics)
    |   +-- interaction_zone (Observation[InteractionZoneValue])
    |
    +-- face
    |   +-- visibility   (Observation[FaceVisibilityValue] + FaceMetrics)
    |
    +-- orientation
    |   +-- head_orientation (Observation[HeadOrientationValue] + OrientationMetrics)
    |
    +-- frame_metadata
        +-- timestamp
        +-- image_width
        +-- image_height
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Optional, TypeVar


# ---------------------------------------------------------------------------
# Status enum (evaluation validity)
# ---------------------------------------------------------------------------

class ObservationStatus(str, Enum):
    """Evaluation validity status for an observation.

    VALID:     Property was evaluated successfully with sufficient evidence.
    UNCERTAIN: Evaluation succeeded, but available evidence is ambiguous.
    INVALID:   Property could not be evaluated.
    """
    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    INVALID = "INVALID"


# ---------------------------------------------------------------------------
# Value enums (per-condition observation values)
# ---------------------------------------------------------------------------

class PersonPresenceValue(str, Enum):
    """Condition 1 — Person Presence observation values."""
    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    UNKNOWN = "UNKNOWN"


class BboxValidityValue(str, Enum):
    """Condition 2 — Bounding-Box Validity observation values."""
    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class FaceVisibilityValue(str, Enum):
    """Condition 3 — Face Visibility and Quality observation values."""
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    UNKNOWN = "UNKNOWN"


class InteractionZoneValue(str, Enum):
    """Condition 4 — Interaction-Zone Membership observation values."""
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    UNKNOWN = "UNKNOWN"


class HeadOrientationValue(str, Enum):
    """Condition 5 — Head Orientation observation values."""
    FACING_ROBOT = "FACING_ROBOT"
    NOT_FACING_ROBOT = "NOT_FACING_ROBOT"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Generic observation dataclass
# ---------------------------------------------------------------------------

V = TypeVar("V")


@dataclass(frozen=True)
class Observation(Generic[V]):
    """Common observation contract: value + confidence + status.

    Attributes:
        value:      Estimated property value (enum member).
        confidence: Model-provided confidence when available; None otherwise.
        status:     Evaluation validity (VALID, UNCERTAIN, INVALID).
    """
    value: V
    confidence: Optional[float]
    status: ObservationStatus


# ---------------------------------------------------------------------------
# Supplementary metrics dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BboxMetrics:
    """Supplementary bounding-box geometry metrics.

    Attributes:
        original_bbox_xyxy: Original detector bounding box (x1, y1, x2, y2).
        clipped_bbox_xyxy:  Bounding box clipped to image bounds.
        width:              Bounding-box width in pixels.
        height:             Bounding-box height in pixels.
        area:               Bounding-box area in pixels squared.
        area_ratio:         Fraction of image area covered by the bbox.
        image_width:        Image width used for clipping/ratio.
        image_height:       Image height used for clipping/ratio.
    """
    original_bbox_xyxy: tuple[float, float, float, float]
    clipped_bbox_xyxy: tuple[float, float, float, float]
    width: float
    height: float
    area: float
    area_ratio: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class FaceMetrics:
    """Supplementary face analysis metrics.

    Attributes:
        face_bbox_xyxy: Bounding box of detected face in pixel coords, or None.
        face_width:     Width of detected face in pixels, or None.
        face_height:    Height of detected face in pixels, or None.
        detection_confidence: MediaPipe face detection confidence, or None.
        landmark_count: Number of landmarks returned, or None.
    """
    face_bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    face_width: Optional[float] = None
    face_height: Optional[float] = None
    detection_confidence: Optional[float] = None
    landmark_count: Optional[int] = None


@dataclass(frozen=True)
class OrientationMetrics:
    """Supplementary head orientation metrics.

    Attributes:
        yaw_ratio:   Nose-to-eye horizontal asymmetry ratio, or None.
        pitch_ratio:  Vertical landmark ratio, or None.
        yaw_threshold: Configured yaw threshold.
        pitch_threshold: Configured pitch threshold.
    """
    yaw_ratio: Optional[float] = None
    pitch_ratio: Optional[float] = None
    yaw_threshold: float = 0.0
    pitch_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Grouped observation dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersonObservations:
    """Person-related observations grouping.

    Attributes:
        presence:         Person detection observation.
        bounding_box:     Bounding-box validity observation.
        bbox_metrics:     Supplementary bbox geometry (None if no detection).
        interaction_zone: Interaction-zone membership observation.
    """
    presence: Observation[PersonPresenceValue]
    bounding_box: Observation[BboxValidityValue]
    bbox_metrics: Optional[BboxMetrics]
    interaction_zone: Observation[InteractionZoneValue]


@dataclass(frozen=True)
class FaceObservations:
    """Face-related observations grouping.

    Attributes:
        visibility: Face visibility and quality observation.
        metrics:    Supplementary face analysis metrics.
    """
    visibility: Observation[FaceVisibilityValue]
    metrics: Optional[FaceMetrics]


@dataclass(frozen=True)
class OrientationObservations:
    """Orientation-related observations grouping.

    Attributes:
        head_orientation: Head orientation observation.
        metrics:          Supplementary orientation metrics.
    """
    head_orientation: Observation[HeadOrientationValue]
    metrics: Optional[OrientationMetrics]


@dataclass(frozen=True)
class FrameMetadata:
    """Frame-level metadata.

    Attributes:
        timestamp:    Frame timestamp (monotonic seconds).
        image_width:  Frame width in pixels.
        image_height: Frame height in pixels.
    """
    timestamp: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class FrameObservation:
    """Complete frame-level structured observation for M4.

    Aggregates all five observation conditions with metadata.

    Structure:
        FrameObservation
        |
        +-- person
        |   +-- presence
        |   +-- bounding_box + bbox_metrics
        |   +-- interaction_zone
        |
        +-- face
        |   +-- visibility + metrics
        |
        +-- orientation
        |   +-- head_orientation + metrics
        |
        +-- frame_metadata
    """
    person: PersonObservations
    face: FaceObservations
    orientation: OrientationObservations
    frame_metadata: FrameMetadata

