"""Image-space motion estimation module for Milestone M5.

Computes linear regression velocity vector and net displacement over the configured
history window according to Milestone M5 specification Sections 5.6 and 5.7.

Architecture & Formulation:
+-----------------------------------------------------------------------------+
|                         M5 IMAGE-SPACE MOTION ESTIMATOR                     |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Observations: {(t_i, c_i = (x_i, y_i))}_{i=1}^N in [t - W, t]              |
|                                                                             |
|  1. Timestamp-Aware Linear Regression Velocity:                             |
|       t_bar = (1/N) * sum(t_i),  x_bar = (1/N) * sum(x_i)                   |
|       Var_t = sum((t_i - t_bar)^2)                                          |
|       b_x = sum((t_i - t_bar) * (x_i - x_bar)) / Var_t                      |
|       b_y = sum((t_i - t_bar) * (y_i - y_bar)) / Var_t                      |
|       v_t = [b_x, b_y]^T,  speed = sqrt(b_x^2 + b_y^2) (pixels/second)      |
|                                                                             |
|  2. Net Displacement over 1.0s Window:                                      |
|       D_t = ||c_t - c_start||_2 = sqrt((x_t - x_0)^2 + (y_t - y_0)^2)       |
|       Coverage check: (t_latest - t_earliest) >= window_seconds - eps        |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from perception.readiness.observation_types import ObservationStatus
from perception.temporal.history import ObservationHistory

DEFAULT_MIN_DISPLACEMENT_WINDOW_SECONDS: float = 1.0
DISPLACEMENT_COVERAGE_TOLERANCE_SECONDS: float = 0.05  # 50 ms tolerance for 30 FPS frame quantization


class MotionEstimator:
    """Estimates image-space velocity and net displacement from observation history."""

    def __init__(
        self,
        min_displacement_window_seconds: float = DEFAULT_MIN_DISPLACEMENT_WINDOW_SECONDS,
        displacement_tolerance_seconds: float = DISPLACEMENT_COVERAGE_TOLERANCE_SECONDS,
    ) -> None:
        """Initialize the motion estimator.

        Args:
            min_displacement_window_seconds: Required time span in seconds to validate displacement.
            displacement_tolerance_seconds: Tolerance allowance for discrete frame sampling (e.g. 30 FPS).
        """
        if min_displacement_window_seconds <= 0.0:
            raise ValueError(
                f"min_displacement_window_seconds must be positive, got {min_displacement_window_seconds}"
            )
        self.min_displacement_window_seconds = float(min_displacement_window_seconds)
        self.displacement_tolerance_seconds = float(displacement_tolerance_seconds)

    def estimate_velocity(
        self,
        history: ObservationHistory,
    ) -> tuple[Optional[tuple[float, float]], Optional[float], ObservationStatus]:
        """Estimate 2D velocity vector and speed using timestamp-aware linear regression.

        Args:
            history: ObservationHistory instance with buffered observations.

        Returns:
            ((vx, vy), speed_px_s, status) where:
                - (vx, vy): Estimated velocity vector in px/s, or None if insufficient history.
                - speed_px_s: Speed magnitude ||v||_2 in px/s, or None if insufficient history.
                - status: ObservationStatus (VALID, UNCERTAIN, or INVALID).
        """
        if not history.is_ready or history.count < 2:
            return None, None, ObservationStatus.UNCERTAIN

        obs = history.observations
        n = len(obs)

        # Compute means
        t_sum = sum(t for t, _ in obs)
        x_sum = sum(cx for _, (cx, _) in obs)
        y_sum = sum(cy for _, (_, cy) in obs)

        t_bar = t_sum / n
        x_bar = x_sum / n
        y_bar = y_sum / n

        # Sum of squares
        var_t = 0.0
        cov_tx = 0.0
        cov_ty = 0.0

        for t, (cx, cy) in obs:
            dt = t - t_bar
            var_t += dt * dt
            cov_tx += dt * (cx - x_bar)
            cov_ty += dt * (cy - y_bar)

        if var_t < 1e-9:
            # All timestamps identical or non-distinct
            return None, None, ObservationStatus.INVALID

        vx = float(cov_tx / var_t)
        vy = float(cov_ty / var_t)
        speed = float(math.hypot(vx, vy))

        return (vx, vy), speed, ObservationStatus.VALID

    def estimate_displacement(
        self,
        history: ObservationHistory,
    ) -> tuple[Optional[float], ObservationStatus]:
        """Estimate net Euclidean displacement over the history window.

        Evaluates D_t = ||c_latest - c_earliest||_2 and checks that the time span
        covers approximately 1.0 second.

        Args:
            history: ObservationHistory instance with buffered observations.

        Returns:
            (displacement_px, status) where status is VALID if temporal coverage is
            satisfied; UNCERTAIN if history span is shorter than the required window.
        """
        if not history.is_ready or history.count < 2:
            return None, ObservationStatus.UNCERTAIN

        obs = history.observations
        t_start, c_start = obs[0]
        t_latest, c_latest = obs[-1]

        duration = t_latest - t_start
        dx = c_latest[0] - c_start[0]
        dy = c_latest[1] - c_start[1]
        disp = float(math.hypot(dx, dy))

        # Check temporal coverage
        min_required = self.min_displacement_window_seconds - self.displacement_tolerance_seconds
        if duration < min_required:
            # Insufficient temporal coverage to evaluate stationary displacement
            return disp, ObservationStatus.UNCERTAIN

        return disp, ObservationStatus.VALID

