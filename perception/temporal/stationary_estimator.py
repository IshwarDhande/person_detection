"""Stationary and movement estimation module for Milestone M5.

Combines velocity magnitude and one-second displacement thresholds to classify
a tracked person as STATIONARY or MOVING according to Milestone M5 specification
Section 5.8.

Architecture & Formulation:
+-----------------------------------------------------------------------------+
|                        M5 STATIONARY PREDICATE                              |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Thresholds (Frozen):                                                       |
|    tau_v = 20.0 px/s (velocity threshold)                                  |
|    tau_D = 20.0 px   (displacement threshold over 1.0s window)             |
|                                                                             |
|  Strict Evaluation:                                                         |
|    S_t = (s_t < tau_v) AND (D_t < tau_D)                                    |
|                                                                             |
|  Decision Logic:                                                            |
|    - If speed or displacement UNCERTAIN/INVALID -> UNKNOWN [status]        |
|    - Else if S_t is True                        -> STATIONARY [VALID]       |
|    - Else (s_t >= tau_v OR D_t >= tau_D)        -> MOVING [VALID]           |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

from typing import Optional

from perception.readiness.observation_types import Observation, ObservationStatus
from perception.temporal.observation_types import MovementValue

DEFAULT_STATIONARY_VELOCITY_THRESHOLD_PX_S: float = 20.0
DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_PX: float = 20.0


class StationaryEstimator:
    """Evaluates stationary and movement behaviour from velocity and displacement."""

    def __init__(
        self,
        velocity_threshold_px_s: float = DEFAULT_STATIONARY_VELOCITY_THRESHOLD_PX_S,
        displacement_threshold_px: float = DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_PX,
    ) -> None:
        """Initialize the stationary estimator.

        Args:
            velocity_threshold_px_s: Maximum velocity in px/s for stationary classification.
            displacement_threshold_px: Maximum displacement in px over 1.0s for stationary.
        """
        if velocity_threshold_px_s < 0.0:
            raise ValueError(
                f"velocity_threshold_px_s must be non-negative, got {velocity_threshold_px_s}"
            )
        if displacement_threshold_px < 0.0:
            raise ValueError(
                f"displacement_threshold_px must be non-negative, got {displacement_threshold_px}"
            )

        self.velocity_threshold_px_s = float(velocity_threshold_px_s)
        self.displacement_threshold_px = float(displacement_threshold_px)

    def evaluate(
        self,
        speed_px_s: Optional[float],
        speed_status: ObservationStatus,
        displacement_px: Optional[float],
        displacement_status: ObservationStatus,
    ) -> Observation[MovementValue]:
        """Evaluate stationary and movement predicate.

        Args:
            speed_px_s: Estimated image-space speed in px/s, or None.
            speed_status: Validity status of the speed estimate.
            displacement_px: Estimated net displacement over 1.0s, or None.
            displacement_status: Validity status of displacement (VALID only if 1.0s covered).

        Returns:
            Observation[MovementValue] with value (STATIONARY, MOVING, UNKNOWN) and status.
        """
        # If either input is invalid or uncertain, output must be UNKNOWN
        if speed_status == ObservationStatus.INVALID or displacement_status == ObservationStatus.INVALID:
            return Observation(
                value=MovementValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )

        if speed_status == ObservationStatus.UNCERTAIN or displacement_status == ObservationStatus.UNCERTAIN:
            return Observation(
                value=MovementValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.UNCERTAIN,
            )

        if speed_px_s is None or displacement_px is None:
            return Observation(
                value=MovementValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.UNCERTAIN,
            )

        # Both estimates are VALID: evaluate strict inequality
        is_stationary = (
            speed_px_s < self.velocity_threshold_px_s
            and displacement_px < self.displacement_threshold_px
        )

        if is_stationary:
            return Observation(
                value=MovementValue.STATIONARY,
                confidence=None,
                status=ObservationStatus.VALID,
            )
        else:
            return Observation(
                value=MovementValue.MOVING,
                confidence=None,
                status=ObservationStatus.VALID,
            )

