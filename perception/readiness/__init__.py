"""M4 Frame-Level Readiness Perception package.

Provides observation conditions for person presence, bounding-box validity,
face visibility/quality, interaction-zone membership, and head orientation.
"""

from perception.readiness.observation_types import (
    ObservationStatus,
    PersonPresenceValue,
    BboxValidityValue,
    FaceVisibilityValue,
    InteractionZoneValue,
    HeadOrientationValue,
    Observation,
    BboxMetrics,
    FaceMetrics,
    OrientationMetrics,
    PersonObservations,
    FaceObservations,
    OrientationObservations,
    FrameMetadata,
    FrameObservation,
)
from perception.readiness.bbox_validator import BboxValidator
from perception.readiness.interaction_zone import InteractionZoneEvaluator
from perception.readiness.face_analyzer import FaceAnalyzer
from perception.readiness.orientation_estimator import OrientationEstimator
from perception.readiness.frame_observer import FrameObserver
from perception.readiness.readiness_config import (
    CameraConfig,
    DetectorConfig,
    FaceAnalyzerConfig,
    OrientationConfig,
    InteractionZoneConfig,
    ReadinessConfig,
    load_readiness_config,
)

__all__ = [
    "ObservationStatus",
    "PersonPresenceValue",
    "BboxValidityValue",
    "FaceVisibilityValue",
    "InteractionZoneValue",
    "HeadOrientationValue",
    "Observation",
    "BboxMetrics",
    "FaceMetrics",
    "OrientationMetrics",
    "PersonObservations",
    "FaceObservations",
    "OrientationObservations",
    "FrameMetadata",
    "FrameObservation",
    "BboxValidator",
    "InteractionZoneEvaluator",
    "FaceAnalyzer",
    "OrientationEstimator",
    "FrameObserver",
    "CameraConfig",
    "DetectorConfig",
    "FaceAnalyzerConfig",
    "OrientationConfig",
    "InteractionZoneConfig",
    "ReadinessConfig",
    "load_readiness_config",
]

