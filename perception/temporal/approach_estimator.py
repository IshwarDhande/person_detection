"""Approach behaviour estimation module for Milestone M5.

Combines movement direction towards the interaction-zone centroid and distance-trend
reduction to detect whether a person is APPROACHING or NOT_APPROACHING according to
Milestone M5 specification Sections 5.9 - 5.12.

Architecture & Formulation:
+-----------------------------------------------------------------------------+
|                          M5 APPROACH ESTIMATOR                              |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Reference: Interaction-zone centroid z = (x_z, y_z)                        |
|  Person center: c_t = (x_t, y_t), Vector to centroid: r_t = z - c_t         |
|                                                                             |
|  1. Direction Cosine:                                                       |
|       cos(theta_t) = (v_t . r_t) / (||v_t||_2 * ||r_t||_2)                  |
|       Criterion: cos(theta_t) > tau_theta (default: 0.5)                    |
|                                                                             |
|  2. Distance Trend (Closing Speed):                                         |
|       d_i = ||z - c_i||_2 for all observations in history                   |
|       Linear regression: d_i ~ a_d + b_d * (t_i - t_bar)                    |
|       Closing speed: -b_d px/s                                              |
|       Criterion: -b_d > tau_a (b_d < -tau_a, default: 10.0 px/s)            |
|                                                                             |
|  3. 500 ms Confirmation Duration:                                           |
|       Approach evidence must persist for T_a >= 0.5s continuously           |
|       before emitting APPROACHING [VALID].                                  |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import math
from typing import Optional

from perception.readiness.observation_types import Observation, ObservationStatus
from perception.temporal.history import ObservationHistory
from perception.temporal.observation_types import ApproachValue

DEFAULT_APPROACH_MIN_DIRECTION_COSINE: float = 0.5  # Provisional (cos(60 deg) = 0.5)
DEFAULT_APPROACH_MIN_CLOSING_SPEED_PX_S: float = 10.0  # Provisional (10 px/s closing speed)
DEFAULT_APPROACH_CONFIRMATION_SECONDS: float = 0.5  # Frozen (500 ms continuous evidence)


class ApproachEstimator:
    """Estimates approach behaviour towards the interaction-zone centroid."""

    def __init__(
        self,
        min_direction_cosine: float = DEFAULT_APPROACH_MIN_DIRECTION_COSINE,
        min_closing_speed_px_s: float = DEFAULT_APPROACH_MIN_CLOSING_SPEED_PX_S,
        confirmation_seconds: float = DEFAULT_APPROACH_CONFIRMATION_SECONDS,
    ) -> None:
        """Initialize the approach estimator.

        Args:
            min_direction_cosine: Minimum acceptable cosine similarity (tau_theta).
            min_closing_speed_px_s: Minimum distance reduction rate in px/s (tau_a).
            confirmation_seconds: Continuous duration required to confirm approach (T_a).
        """
        if not (-1.0 <= min_direction_cosine <= 1.0):
            raise ValueError(
                f"min_direction_cosine must be in [-1.0, 1.0], got {min_direction_cosine}"
            )
        if min_closing_speed_px_s < 0.0:
            raise ValueError(
                f"min_closing_speed_px_s must be non-negative, got {min_closing_speed_px_s}"
            )
        if confirmation_seconds <= 0.0:
            raise ValueError(
                f"confirmation_seconds must be positive, got {confirmation_seconds}"
            )

        self.min_direction_cosine = float(min_direction_cosine)
        self.min_closing_speed_px_s = float(min_closing_speed_px_s)
        self.confirmation_seconds = float(confirmation_seconds)

        self._approach_start_time: Optional[float] = None

    def reset(self) -> None:
        """Reset confirmation state on tracking loss or discontinuity."""
        self._approach_start_time = None

    def evaluate(
        self,
        history: ObservationHistory,
        zone_centroid: tuple[float, float],
        velocity_xy_px_s: Optional[tuple[float, float]],
        speed_px_s: Optional[float],
        velocity_status: ObservationStatus,
        timestamp: float,
    ) -> tuple[
        Observation[ApproachValue],
        Optional[float],  # direction_cosine
        Optional[float],  # closing_speed_px_s
        Optional[float],  # distance_to_zone_centroid_px
    ]:
        """Evaluate approach predicate towards interaction-zone centroid.

        Args:
            history: Active ObservationHistory instance.
            zone_centroid: (x_z, y_z) centroid of the interaction-zone in pixel coords.
            velocity_xy_px_s: Estimated image-space velocity vector (vx, vy) in px/s.
            speed_px_s: Estimated speed ||v||_2 in px/s.
            velocity_status: ObservationStatus of the velocity estimate.
            timestamp: Monotonic timestamp of the current frame in seconds.

        Returns:
            (approach_obs, direction_cosine, closing_speed_px_s, distance_to_centroid_px)
        """
        if not history.is_ready or history.count < 2 or history.latest_center is None:
            self._approach_start_time = None
            return (
                Observation(
                    value=ApproachValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.UNCERTAIN,
                ),
                None,
                None,
                None,
            )

        cx, cy = history.latest_center
        zx, zy = zone_centroid

        # Vector from person center towards zone centroid
        rx = zx - cx
        ry = zy - cy
        dist = float(math.hypot(rx, ry))

        # Direction cosine calculation
        direction_cosine: Optional[float] = None
        if (
            velocity_status == ObservationStatus.VALID
            and velocity_xy_px_s is not None
            and speed_px_s is not None
            and speed_px_s > 1e-5
            and dist > 1e-5
        ):
            vx, vy = velocity_xy_px_s
            dot_prod = vx * rx + vy * ry
            raw_cosine = dot_prod / (speed_px_s * dist)
            # Defensive clamp to [-1.0, 1.0]
            direction_cosine = float(max(-1.0, min(1.0, raw_cosine)))

        # Distance trend (closing speed) via linear regression over history
        obs = history.observations
        n = len(obs)
        closing_speed: Optional[float] = None

        if n >= 2:
            t_sum = sum(t for t, _ in obs)
            d_list = [math.hypot(zx - c[0], zy - c[1]) for _, c in obs]
            d_sum = sum(d_list)

            t_bar = t_sum / n
            d_bar = d_sum / n

            var_t = 0.0
            cov_td = 0.0
            for (t, _), d in zip(obs, d_list):
                dt = t - t_bar
                var_t += dt * dt
                cov_td += dt * (d - d_bar)

            if var_t >= 1e-9:
                b_d = cov_td / var_t
                closing_speed = float(-b_d)

        # Instantaneous approach evidence check
        has_instant_evidence = False
        if direction_cosine is not None and closing_speed is not None:
            has_direction = direction_cosine > self.min_direction_cosine
            has_speed = closing_speed > self.min_closing_speed_px_s
            has_instant_evidence = has_direction and has_speed

        # Confirmation timer handling
        if has_instant_evidence:
            if self._approach_start_time is None:
                self._approach_start_time = timestamp

            elapsed = timestamp - self._approach_start_time
            if elapsed >= self.confirmation_seconds:
                return (
                    Observation(
                        value=ApproachValue.APPROACHING,
                        confidence=None,
                        status=ObservationStatus.VALID,
                    ),
                    direction_cosine,
                    closing_speed,
                    dist,
                )
            else:
                # Evidence accumulating towards confirmation
                return (
                    Observation(
                        value=ApproachValue.UNKNOWN,
                        confidence=None,
                        status=ObservationStatus.UNCERTAIN,
                    ),
                    direction_cosine,
                    closing_speed,
                    dist,
                )
        else:
            # No instant approach evidence
            self._approach_start_time = None
            if velocity_status == ObservationStatus.VALID:
                # Reliable evidence establishes the person is not approaching
                return (
                    Observation(
                        value=ApproachValue.NOT_APPROACHING,
                        confidence=None,
                        status=ObservationStatus.VALID,
                    ),
                    direction_cosine,
                    closing_speed,
                    dist,
                )
            else:
                return (
                    Observation(
                        value=ApproachValue.UNKNOWN,
                        confidence=None,
                        status=ObservationStatus.UNCERTAIN,
                    ),
                    direction_cosine,
                    closing_speed,
                    dist,
                )

