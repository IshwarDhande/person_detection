"""Configuration management for Milestone M6 First-Contact Readiness.

Loads and validates operational parameters including distance thresholds,
modes, eligibility timing, and tracking-loss grace periods.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_DISTANCE_MAX_M: float = 1.5
DEFAULT_DISTANCE_MODE: str = "mock"
DEFAULT_MOCK_DISTANCE_M: float = 1.2
DEFAULT_ELIGIBILITY_DURATION_SECONDS: float = 2.0
DEFAULT_TRACKING_LOSS_TIMEOUT_SECONDS: float = 5.0
DEFAULT_OBSERVATION_FRESHNESS_SECONDS: float = 0.5
DEFAULT_MAX_INTER_FRAME_GAP_SECONDS: float = 0.5


@dataclass(frozen=True)
class ContactConfig:
    """Configuration parameters for M6 First-Contact Readiness Engine."""

    distance_max_m: float = DEFAULT_DISTANCE_MAX_M
    distance_mode: str = DEFAULT_DISTANCE_MODE
    mock_distance_m: float = DEFAULT_MOCK_DISTANCE_M
    eligibility_duration_seconds: float = DEFAULT_ELIGIBILITY_DURATION_SECONDS
    tracking_loss_timeout_seconds: float = DEFAULT_TRACKING_LOSS_TIMEOUT_SECONDS
    observation_freshness_seconds: float = DEFAULT_OBSERVATION_FRESHNESS_SECONDS
    max_inter_frame_gap_seconds: float = DEFAULT_MAX_INTER_FRAME_GAP_SECONDS
    require_metric_distance_in_production: bool = False

    def validate(self) -> None:
        """Validate configuration parameters against domain requirements."""
        if not math.isfinite(self.distance_max_m) or self.distance_max_m <= 0.0:
            raise ValueError(f"distance_max_m must be a positive finite float, got {self.distance_max_m}")
        mode_clean = self.distance_mode.strip().lower()
        if mode_clean not in ("mock", "metric"):
            raise ValueError(f"distance_mode must be 'mock' or 'metric', got '{self.distance_mode}'")
        if not math.isfinite(self.mock_distance_m) or self.mock_distance_m < 0.0:
            raise ValueError(f"mock_distance_m must be a non-negative finite float, got {self.mock_distance_m}")
        if not math.isfinite(self.eligibility_duration_seconds) or self.eligibility_duration_seconds <= 0.0:
            raise ValueError(
                f"eligibility_duration_seconds must be a positive finite float, got {self.eligibility_duration_seconds}"
            )
        if not math.isfinite(self.tracking_loss_timeout_seconds) or self.tracking_loss_timeout_seconds <= 0.0:
            raise ValueError(
                f"tracking_loss_timeout_seconds must be a positive finite float, got {self.tracking_loss_timeout_seconds}"
            )
        if not math.isfinite(self.observation_freshness_seconds) or self.observation_freshness_seconds <= 0.0:
            raise ValueError(
                f"observation_freshness_seconds must be a positive finite float, got {self.observation_freshness_seconds}"
            )
        if not math.isfinite(self.max_inter_frame_gap_seconds) or self.max_inter_frame_gap_seconds <= 0.0:
            raise ValueError(
                f"max_inter_frame_gap_seconds must be a positive finite float, got {self.max_inter_frame_gap_seconds}"
            )
        if self.require_metric_distance_in_production and mode_clean == "mock":
            raise ValueError(
                "Production mode requires metric distance provider; mock mode is strictly prohibited."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactConfig:
        """Construct ContactConfig from a dictionary."""
        cfg = cls(
            distance_max_m=float(data.get("distance_max_m", DEFAULT_DISTANCE_MAX_M)),
            distance_mode=str(data.get("distance_mode", DEFAULT_DISTANCE_MODE)),
            mock_distance_m=float(data.get("mock_distance_m", DEFAULT_MOCK_DISTANCE_M)),
            eligibility_duration_seconds=float(
                data.get("eligibility_duration_seconds", DEFAULT_ELIGIBILITY_DURATION_SECONDS)
            ),
            tracking_loss_timeout_seconds=float(
                data.get("tracking_loss_timeout_seconds", DEFAULT_TRACKING_LOSS_TIMEOUT_SECONDS)
            ),
            observation_freshness_seconds=float(
                data.get("observation_freshness_seconds", DEFAULT_OBSERVATION_FRESHNESS_SECONDS)
            ),
            max_inter_frame_gap_seconds=float(
                data.get("max_inter_frame_gap_seconds", DEFAULT_MAX_INTER_FRAME_GAP_SECONDS)
            ),
            require_metric_distance_in_production=bool(
                data.get("require_metric_distance_in_production", False)
            ),
        )
        cfg.validate()
        return cfg


def load_contact_config(config_path: Optional[str | Path] = None) -> ContactConfig:
    """Load and validate ContactConfig from a YAML file or defaults."""
    if config_path is None:
        default_yaml = Path("configs/m6/first_contact.yaml")
        if default_yaml.is_file():
            config_path = default_yaml
        else:
            cfg = ContactConfig()
            cfg.validate()
            return cfg

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"M6 configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Configuration file {path} must contain a top-level YAML mapping.")

    # Check for nested 'first_contact' or flat
    data = raw_data.get("first_contact", raw_data)
    return ContactConfig.from_dict(data)

