"""Configuration loader and schema validator for Milestone M4.

Parses and validates M4 parameters from YAML files against expected types,
bounds, and required geometric properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml


@dataclass(frozen=True)
class CameraConfig:
    """Camera capture stream configuration."""
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")


@dataclass(frozen=True)
class DetectorConfig:
    """YOLO person detector configuration."""
    model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = 0.50
    class_id: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_threshold) or not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}")
        if self.class_id != 0:
            raise ValueError(f"class_id must be 0 for person detection, got {self.class_id}")


@dataclass(frozen=True)
class FaceAnalyzerConfig:
    """MediaPipe face analyzer configuration."""
    model_path: str = "models/face_landmarker.task"
    min_face_width: int = 40
    min_face_height: int = 40
    min_detection_confidence: float = 0.50
    min_tracking_confidence: float = 0.50

    def __post_init__(self) -> None:
        if self.min_face_width <= 0:
            raise ValueError(f"min_face_width must be positive, got {self.min_face_width}")
        if self.min_face_height <= 0:
            raise ValueError(f"min_face_height must be positive, got {self.min_face_height}")
        if not math.isfinite(self.min_detection_confidence) or not (0.0 <= self.min_detection_confidence <= 1.0):
            raise ValueError(f"min_detection_confidence must be in [0.0, 1.0], got {self.min_detection_confidence}")
        if not math.isfinite(self.min_tracking_confidence) or not (0.0 <= self.min_tracking_confidence <= 1.0):
            raise ValueError(f"min_tracking_confidence must be in [0.0, 1.0], got {self.min_tracking_confidence}")


@dataclass(frozen=True)
class OrientationConfig:
    """Head orientation estimation configuration."""
    yaw_threshold: float = 0.35
    pitch_threshold: float = 0.60
    uncertain_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.yaw_threshold) or self.yaw_threshold <= 0:
            raise ValueError(f"yaw_threshold must be positive and finite, got {self.yaw_threshold}")
        if not math.isfinite(self.pitch_threshold) or self.pitch_threshold <= 0:
            raise ValueError(f"pitch_threshold must be positive and finite, got {self.pitch_threshold}")
        if not math.isfinite(self.uncertain_multiplier) or self.uncertain_multiplier <= 1.0:
            raise ValueError(f"uncertain_multiplier must be > 1.0 and finite, got {self.uncertain_multiplier}")


@dataclass(frozen=True)
class InteractionZoneConfig:
    """Interaction-zone polygon configuration."""
    enabled: bool = True
    coordinate_system: str = "normalized"
    vertices: tuple[tuple[float, float], ...] = (
        (0.05, 0.05),
        (0.95, 0.05),
        (0.95, 0.95),
        (0.05, 0.95),
    )

    def __post_init__(self) -> None:
        if self.coordinate_system != "normalized":
            raise ValueError(f"Only 'normalized' coordinate_system supported, got '{self.coordinate_system}'")
        if self.enabled:
            from perception.readiness.interaction_zone import validate_polygon_vertices
            validate_polygon_vertices(self.vertices)


@dataclass(frozen=True)
class ReadinessConfig:
    """Top-level Milestone M4 configuration bundle."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    face_analyzer: FaceAnalyzerConfig = field(default_factory=FaceAnalyzerConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    interaction_zone: InteractionZoneConfig = field(default_factory=InteractionZoneConfig)


def load_readiness_config(config_path: str | Path) -> ReadinessConfig:
    """Load, parse, and validate readiness configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated ReadinessConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If config parameters violate schema constraints.
    """
    path = Path(config_path)
    if not path.is_file():
        # Fallback to repo root if path is relative
        repo_root = Path(__file__).resolve().parents[2]
        if (repo_root / config_path).is_file():
            path = repo_root / config_path
        else:
            raise FileNotFoundError(f"Readiness configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary root")

    cam_data = data.get("camera", {})
    cam_cfg = CameraConfig(
        device=str(cam_data.get("device", "/dev/video0")),
        width=int(cam_data.get("width", 1280)),
        height=int(cam_data.get("height", 720)),
        fps=int(cam_data.get("fps", 30)),
        fourcc=str(cam_data.get("fourcc", "MJPG")),
    )

    det_data = data.get("detector", {})
    det_cfg = DetectorConfig(
        model_path=str(det_data.get("model_path", "models/yolov8n.pt")),
        confidence_threshold=float(det_data.get("confidence_threshold", 0.50)),
        class_id=int(det_data.get("class_id", 0)),
    )

    face_data = data.get("face_analyzer", {})
    face_cfg = FaceAnalyzerConfig(
        model_path=str(face_data.get("model_path", "models/face_landmarker.task")),
        min_face_width=int(face_data.get("min_face_width", 40)),
        min_face_height=int(face_data.get("min_face_height", 40)),
        min_detection_confidence=float(face_data.get("min_detection_confidence", 0.50)),
        min_tracking_confidence=float(face_data.get("min_tracking_confidence", 0.50)),
    )

    orient_data = data.get("orientation", {})
    orient_cfg = OrientationConfig(
        yaw_threshold=float(orient_data.get("yaw_threshold", 0.35)),
        pitch_threshold=float(orient_data.get("pitch_threshold", 0.60)),
        uncertain_multiplier=float(orient_data.get("uncertain_multiplier", 1.5)),
    )

    zone_data = data.get("interaction_zone", {})
    raw_vertices = zone_data.get("vertices", [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]])
    tuple_vertices = tuple((float(v[0]), float(v[1])) for v in raw_vertices)
    zone_cfg = InteractionZoneConfig(
        enabled=bool(zone_data.get("enabled", True)),
        coordinate_system=str(zone_data.get("coordinate_system", "normalized")),
        vertices=tuple_vertices,
    )

    return ReadinessConfig(
        camera=cam_cfg,
        detector=det_cfg,
        face_analyzer=face_cfg,
        orientation=orient_cfg,
        interaction_zone=zone_cfg,
    )

