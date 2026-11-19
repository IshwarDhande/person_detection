"""Configuration data model and loader for Milestone M5 Temporal Behaviour Estimation.

Defines validated configuration dataclasses for motion history, stationary estimation,
approach estimation, and stability timing according to Milestone M5 specification Section 7.

Architecture & Parameter Hierarchy:
+-----------------------------------------------------------------------------+
|                            M5 TEMPORAL CONFIG                               |
+-----------------------------------------------------------------------------+
|                                                                             |
|  TemporalConfig                                                             |
|  |                                                                          |
|  +-- history_window_seconds:                 1.0  (Frozen)                  |
|  +-- max_observation_gap_seconds:            0.2  (Frozen)                  |
|  +-- initial_valid_observations:             5    (Frozen)                  |
|  |                                                                          |
|  +-- stationary_velocity_threshold_px_s:     20.0 (Frozen)                  |
|  +-- stationary_displacement_threshold_px:   20.0 (Frozen)                  |
|  +-- stationary_displacement_window_seconds: 1.0  (Frozen)                  |
|  |                                                                          |
|  +-- approach_confirmation_seconds:          0.5  (Frozen)                  |
|  +-- approach_min_direction_cosine:          0.5  (Provisional: cos 60 deg) |
|  +-- approach_min_closing_speed_px_s:        10.0 (Provisional: 10 px/s)    |
|  |                                                                          |
|  +-- stability_duration_seconds:             2.0  (Frozen)                  |
|  +-- stability_violation_tolerance_seconds:  0.2  (Frozen)                  |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class TemporalConfig:
    """Operational parameters for Milestone M5 Temporal Behaviour Estimation.

    Distinguishes frozen design decisions from provisional calibration parameters.
    """

    # History & Continuity (Frozen)
    history_window_seconds: float = 1.0
    max_observation_gap_seconds: float = 0.2
    initial_valid_observations: int = 5

    # Stationary Predicate (Frozen)
    stationary_velocity_threshold_px_s: float = 20.0
    stationary_displacement_threshold_px: float = 20.0
    stationary_displacement_window_seconds: float = 1.0

    # Approach Predicate (Frozen confirmation; Provisional thresholds)
    approach_confirmation_seconds: float = 0.5
    approach_min_direction_cosine: float = 0.5  # Provisional
    approach_min_closing_speed_px_s: float = 10.0  # Provisional

    # Stability Timing (Frozen)
    stability_duration_seconds: float = 2.0
    stability_violation_tolerance_seconds: float = 0.2

    def validate(self) -> None:
        """Validate all parameters according to M5 specification Section 7.1.

        Raises:
            ValueError: If any parameter violates domain or consistency constraints.
        """
        # History checks
        if not math.isfinite(self.history_window_seconds) or self.history_window_seconds <= 0.0:
            raise ValueError(
                f"history_window_seconds must be a positive finite float, got {self.history_window_seconds}"
            )
        if (
            not math.isfinite(self.max_observation_gap_seconds)
            or self.max_observation_gap_seconds <= 0.0
        ):
            raise ValueError(
                f"max_observation_gap_seconds must be a positive finite float, got {self.max_observation_gap_seconds}"
            )
        if self.initial_valid_observations < 1:
            raise ValueError(
                f"initial_valid_observations must be an integer >= 1, got {self.initial_valid_observations}"
            )

        # Stationary checks
        if (
            not math.isfinite(self.stationary_velocity_threshold_px_s)
            or self.stationary_velocity_threshold_px_s < 0.0
        ):
            raise ValueError(
                f"stationary_velocity_threshold_px_s must be non-negative, got {self.stationary_velocity_threshold_px_s}"
            )
        if (
            not math.isfinite(self.stationary_displacement_threshold_px)
            or self.stationary_displacement_threshold_px < 0.0
        ):
            raise ValueError(
                f"stationary_displacement_threshold_px must be non-negative, got {self.stationary_displacement_threshold_px}"
            )
        if (
            not math.isfinite(self.stationary_displacement_window_seconds)
            or self.stationary_displacement_window_seconds <= 0.0
        ):
            raise ValueError(
                f"stationary_displacement_window_seconds must be positive, got {self.stationary_displacement_window_seconds}"
            )

        # Approach checks
        if (
            not math.isfinite(self.approach_confirmation_seconds)
            or self.approach_confirmation_seconds <= 0.0
        ):
            raise ValueError(
                f"approach_confirmation_seconds must be positive, got {self.approach_confirmation_seconds}"
            )
        if not (-1.0 <= self.approach_min_direction_cosine <= 1.0):
            raise ValueError(
                f"approach_min_direction_cosine must be in [-1.0, 1.0], got {self.approach_min_direction_cosine}"
            )
        if (
            not math.isfinite(self.approach_min_closing_speed_px_s)
            or self.approach_min_closing_speed_px_s < 0.0
        ):
            raise ValueError(
                f"approach_min_closing_speed_px_s must be non-negative, got {self.approach_min_closing_speed_px_s}"
            )

        # Stability checks
        if (
            not math.isfinite(self.stability_duration_seconds)
            or self.stability_duration_seconds <= 0.0
        ):
            raise ValueError(
                f"stability_duration_seconds must be positive, got {self.stability_duration_seconds}"
            )
        if (
            not math.isfinite(self.stability_violation_tolerance_seconds)
            or self.stability_violation_tolerance_seconds < 0.0
        ):
            raise ValueError(
                f"stability_violation_tolerance_seconds must be non-negative, got {self.stability_violation_tolerance_seconds}"
            )

        # Consistency checks
        if self.stationary_displacement_window_seconds > self.history_window_seconds:
            raise ValueError(
                f"stationary_displacement_window_seconds ({self.stationary_displacement_window_seconds}) "
                f"cannot exceed history_window_seconds ({self.history_window_seconds})"
            )
        if self.stability_violation_tolerance_seconds >= self.stability_duration_seconds:
            raise ValueError(
                f"stability_violation_tolerance_seconds ({self.stability_violation_tolerance_seconds}) "
                f"must be strictly less than stability_duration_seconds ({self.stability_duration_seconds})"
            )


def load_temporal_config(config_source: Optional[Union[str, Path, dict]] = None) -> TemporalConfig:
    """Load and validate TemporalConfig from a YAML file path or dictionary.

    Args:
        config_source: Path to YAML config file, dict of parameters, or None for defaults.

    Returns:
        Validated TemporalConfig instance.

    Raises:
        ValueError: If file is not found or config values are invalid.
    """
    if config_source is None:
        cfg = TemporalConfig()
        cfg.validate()
        return cfg

    if isinstance(config_source, (str, Path)):
        p = Path(config_source)
        if not p.is_file():
            raise ValueError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif isinstance(config_source, dict):
        data = config_source
    else:
        raise ValueError(f"Unsupported config source type: {type(config_source)}")

    # Support nested 'temporal' section or flat dict
    if "temporal" in data and isinstance(data["temporal"], dict):
        data = data["temporal"]

    fields = {
        "history_window_seconds": data.get("history_window_seconds", 1.0),
        "max_observation_gap_seconds": data.get("max_observation_gap_seconds", 0.2),
        "initial_valid_observations": data.get("initial_valid_observations", 5),
        "stationary_velocity_threshold_px_s": data.get("stationary_velocity_threshold_px_s", 20.0),
        "stationary_displacement_threshold_px": data.get("stationary_displacement_threshold_px", 20.0),
        "stationary_displacement_window_seconds": data.get("stationary_displacement_window_seconds", 1.0),
        "approach_confirmation_seconds": data.get("approach_confirmation_seconds", 0.5),
        "approach_min_direction_cosine": data.get("approach_min_direction_cosine", 0.5),
        "approach_min_closing_speed_px_s": data.get("approach_min_closing_speed_px_s", 10.0),
        "stability_duration_seconds": data.get("stability_duration_seconds", 2.0),
        "stability_violation_tolerance_seconds": data.get("stability_violation_tolerance_seconds", 0.2),
    }

    cfg = TemporalConfig(**fields)
    cfg.validate()
    return cfg

