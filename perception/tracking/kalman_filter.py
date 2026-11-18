"""Constant-Velocity 8-Dimensional Kalman Filter for Milestone M3 Tracking.

This module provides a linear Kalman filter implementing the constant-velocity motion
model for single-person bounding-box tracking as defined in Milestone M3.

State vector (8-D):
    x = [cx, cy, w, h, vx, vy, vw, vh]^T
    where (cx, cy) is bounding-box center, (w, h) are width and height,
    and (vx, vy, vw, vh) are their respective velocities in image space.

Measurement vector (4-D):
    z = [cx, cy, w, h]^T

Architecture & State Transitions:
+-----------------------------------------------------------------------------+
|                8-D CONSTANT-VELOCITY KALMAN FILTER PIPELINE                 |
+-----------------------------------------------------------------------------+
|                                                                             |
|  State: x = [cx, cy, w, h, vx, vy, vw, vh]^T                               |
|                                                                             |
|  1. Time Update (Predict with monotonic elapsed dt):                        |
|       F(dt) = [[I_4, dt * I_4],                                             |
|                [0_4,      I_4]]                                             |
|       x_pred = F(dt) * x                                                    |
|       P_pred = F(dt) * P * F(dt)^T + Q(dt)                                  |
|                                                                             |
|  2. Measurement Gate & Innovation Covariance:                               |
|       H = [I_4, 0_4]                                                        |
|       y = z - H * x_pred                                                    |
|       S = H * P_pred * H^T + R                                              |
|       d_M^2 = y^T * S^(-1) * y  (Mahalanobis distance squared)              |
|                                                                             |
|  3. Measurement Update (Correct with matched detection):                    |
|       K = P_pred * H^T * S^(-1)                                             |
|       x_new = x_pred + K * y                                                |
|       P_new = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T (Joseph form)  |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class ConstantVelocityKalmanFilter8D:
    """Eight-dimensional constant-velocity linear Kalman filter for bounding-box tracking.

    Implements:
        State vector: x = [cx, cy, w, h, vx, vy, vw, vh]^T in R^8
        Measurement vector: z = [cx, cy, w, h]^T in R^4

    Parameters:
        initial_bbox_xyxy: Initial bounding box (x1, y1, x2, y2).
        process_noise_pos: Standard deviation of position acceleration noise (pixels/s^2).
                           Default: 80.0 px/s^2.
        process_noise_size: Standard deviation of size acceleration noise (pixels/s^2).
                            Default: 20.0 px/s^2.
        measurement_noise_pos: Standard deviation of detector center measurement noise (pixels).
                               Default: 10.0 px.
        measurement_noise_size: Standard deviation of detector size measurement noise (pixels).
                                Default: 20.0 px.

    Note on parameters:
        These process and measurement noise values are implementation baseline parameters
        providing stable smoothing and responsive motion prediction on standard video streams.
        They are not claimed to be empirically optimal across all hardware configurations.
    """

    STATE_DIM: int = 8
    MEAS_DIM: int = 4

    def __init__(
        self,
        initial_bbox_xyxy: tuple[float, float, float, float],
        process_noise_pos: float = 80.0,
        process_noise_size: float = 20.0,
        measurement_noise_pos: float = 10.0,
        measurement_noise_size: float = 20.0,
    ) -> None:
        """Initialize the 8-D Kalman filter from an initial bounding box.

        Args:
            initial_bbox_xyxy: Bounding box tuple (x1, y1, x2, y2).
            process_noise_pos: Acceleration noise std for center coordinates (px/s^2).
            process_noise_size: Acceleration noise std for width and height (px/s^2).
            measurement_noise_pos: Measurement noise std for center coordinates (px).
            measurement_noise_size: Measurement noise std for width and height (px).

        Raises:
            ValueError: If initial coordinates or noise parameters are invalid or non-finite.
        """
        self._validate_bbox(initial_bbox_xyxy)
        for name, val in [
            ("process_noise_pos", process_noise_pos),
            ("process_noise_size", process_noise_size),
            ("measurement_noise_pos", measurement_noise_pos),
            ("measurement_noise_size", measurement_noise_size),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a positive finite float, got {val}")

        self.q_pos_std = float(process_noise_pos)
        self.q_size_std = float(process_noise_size)
        self.r_pos_std = float(measurement_noise_pos)
        self.r_size_std = float(measurement_noise_size)

        # Convert initial box to [cx, cy, w, h]
        x1, y1, x2, y2 = initial_bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # Initialize state: initial velocities set to 0.0
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Measurement matrix H (4x8): extracts [cx, cy, w, h]
        self.H = np.zeros((self.MEAS_DIM, self.STATE_DIM), dtype=np.float64)
        for i in range(self.MEAS_DIM):
            self.H[i, i] = 1.0

        # Measurement covariance R (4x4)
        self.R = np.diag([
            self.r_pos_std ** 2,
            self.r_pos_std ** 2,
            self.r_size_std ** 2,
            self.r_size_std ** 2,
        ]).astype(np.float64)

        # Initial state covariance P (8x8):
        # Position/size uncertainty matches measurement noise;
        # Velocity uncertainty is initialized higher (~100 px/s std) until observations refine it.
        init_v_var = 100.0 ** 2
        self.P = np.diag([
            self.r_pos_std ** 2,
            self.r_pos_std ** 2,
            self.r_size_std ** 2,
            self.r_size_std ** 2,
            init_v_var,
            init_v_var,
            init_v_var,
            init_v_var,
        ]).astype(np.float64)

        # Cache predicted state and covariance
        self._x_pred: np.ndarray = self.x.copy()
        self._P_pred: np.ndarray = self.P.copy()

    @staticmethod
    def _validate_bbox(bbox_xyxy: tuple[float, float, float, float]) -> None:
        """Validate bounding box coordinate integrity."""
        if not isinstance(bbox_xyxy, (tuple, list)) or len(bbox_xyxy) != 4:
            raise ValueError(f"bbox_xyxy must be a 4-element tuple, got {bbox_xyxy}")
        for i, val in enumerate(bbox_xyxy):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"bbox_xyxy coordinate at index {i} is not a finite float: {val}")
        x1, y1, x2, y2 = bbox_xyxy
        if x1 > x2 or y1 > y2:
            raise ValueError(f"Inverted bounding box coordinates: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

    @classmethod
    def bbox_to_z(cls, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
        """Convert bounding box (x1, y1, x2, y2) into measurement vector [cx, cy, w, h]."""
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return np.array([cx, cy, w, h], dtype=np.float64)

    @classmethod
    def state_to_bbox(cls, state_vector: np.ndarray) -> tuple[float, float, float, float]:
        """Convert Kalman state vector [cx, cy, w, h, ...] into bounding box (x1, y1, x2, y2)."""
        cx = float(state_vector[0])
        cy = float(state_vector[1])
        w = max(1.0, float(state_vector[2]))
        h = max(1.0, float(state_vector[3]))
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (x1, y1, x2, y2)

    def _build_transition_matrix(self, dt: float) -> np.ndarray:
        """Construct the 8x8 state transition matrix F(dt) for elapsed time dt."""
        F = np.eye(self.STATE_DIM, dtype=np.float64)
        for i in range(self.MEAS_DIM):
            F[i, i + 4] = dt
        return F

    def _build_process_noise(self, dt: float) -> np.ndarray:
        """Construct the 8x8 process noise covariance Q(dt) for elapsed time dt.

        Uses the Continuous White Noise Acceleration (CWNA) formulation:
        For each 1-D coordinate q with acceleration variance sigma_a^2:
            Q_1d = sigma_a^2 * [[dt^3 / 3, dt^2 / 2],
                                [dt^2 / 2, dt      ]]
        """
        dt2 = dt * dt
        dt3 = dt2 * dt

        Q = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)

        # Center coordinates (cx, cy)
        q_pos_var = self.q_pos_std ** 2
        for i in (0, 1):
            Q[i, i] = q_pos_var * (dt3 / 3.0)
            Q[i, i + 4] = q_pos_var * (dt2 / 2.0)
            Q[i + 4, i] = q_pos_var * (dt2 / 2.0)
            Q[i + 4, i + 4] = q_pos_var * dt

        # Size coordinates (w, h)
        q_size_var = self.q_size_std ** 2
        for i in (2, 3):
            Q[i, i] = q_size_var * (dt3 / 3.0)
            Q[i, i + 4] = q_size_var * (dt2 / 2.0)
            Q[i + 4, i] = q_size_var * (dt2 / 2.0)
            Q[i + 4, i + 4] = q_size_var * dt

        return Q

    def predict(self, dt: float) -> tuple[float, float, float, float]:
        """Perform the time-update step using monotonic elapsed time dt.

        Args:
            dt: Monotonic elapsed time in seconds since previous update.

        Returns:
            Predicted bounding box (x1, y1, x2, y2).

        Raises:
            ValueError: If dt is not finite or strictly negative.
        """
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt < 0.0:
            raise ValueError(f"dt must be a non-negative finite float, got {dt}")

        # If dt is negligibly small (e.g. repeated timestamp), preserve state
        if dt < 1e-6:
            self._x_pred = self.x.copy()
            self._P_pred = self.P.copy()
            return self.state_to_bbox(self._x_pred)

        F = self._build_transition_matrix(dt)
        Q = self._build_process_noise(dt)

        # Prior state prediction
        self._x_pred = F @ self.x

        # Defensively ensure predicted width and height remain strictly positive
        if self._x_pred[2] < 1.0:
            self._x_pred[2] = 1.0
        if self._x_pred[3] < 1.0:
            self._x_pred[3] = 1.0

        # Prior covariance prediction
        self._P_pred = F @ self.P @ F.T + Q

        # Defensively symmetrize covariance
        self._P_pred = (self._P_pred + self._P_pred.T) * 0.5

        # In unobserved coasting frames, the state follows the prediction
        self.x = self._x_pred.copy()
        self.P = self._P_pred.copy()

        return self.state_to_bbox(self._x_pred)

    def innovation_covariance(self) -> np.ndarray:
        """Compute the 4x4 innovation covariance matrix S = H * P_pred * H^T + R."""
        S = self.H @ self._P_pred @ self.H.T + self.R
        return (S + S.T) * 0.5

    def mahalanobis_distance_squared(
        self, measurement_bbox_xyxy: tuple[float, float, float, float]
    ) -> float:
        """Calculate the squared Mahalanobis distance d_M^2 for a candidate measurement.

        Formula:
            d_M^2 = (z - H * x_pred)^T * S^(-1) * (z - H * x_pred)

        Args:
            measurement_bbox_xyxy: Candidate detection bounding box (x1, y1, x2, y2).

        Returns:
            Squared Mahalanobis distance (chi-square distributed with 4 degrees of freedom).
        """
        self._validate_bbox(measurement_bbox_xyxy)
        z = self.bbox_to_z(measurement_bbox_xyxy)
        y = z - (self.H @ self._x_pred)  # Innovation vector
        S = self.innovation_covariance()

        try:
            # Solve S * v = y => v = S^(-1) * y
            v = np.linalg.solve(S, y)
            d2 = float(np.dot(y, v))
            return max(0.0, d2)
        except np.linalg.LinAlgError:
            return float("inf")

    def update(
        self, measurement_bbox_xyxy: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Perform the measurement update step with a matched detection.

        Args:
            measurement_bbox_xyxy: Matched detection bounding box (x1, y1, x2, y2).

        Returns:
            Posterior updated bounding box (x1, y1, x2, y2).

        Raises:
            ValueError: If bounding box coordinates are invalid or non-finite.
        """
        self._validate_bbox(measurement_bbox_xyxy)
        z = self.bbox_to_z(measurement_bbox_xyxy)

        # Innovation
        y = z - (self.H @ self._x_pred)
        S = self.innovation_covariance()

        # Kalman gain: K = P_pred * H^T * S^(-1)
        # Solve S^T * K^T = H * P_pred^T
        try:
            K = np.linalg.solve(S, self.H @ self._P_pred).T
        except np.linalg.LinAlgError:
            # Fallback using pseudo-inverse if singular
            K = self._P_pred @ self.H.T @ np.linalg.pinv(S)

        # State update
        self.x = self._x_pred + (K @ y)

        # Defensively clamp width and height to positive values
        if self.x[2] < 1.0:
            self.x[2] = 1.0
        if self.x[3] < 1.0:
            self.x[3] = 1.0

        # Joseph form covariance update for numerical stability and guaranteed symmetry:
        # P = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
        I = np.eye(self.STATE_DIM, dtype=np.float64)
        I_KH = I - (K @ self.H)
        self.P = (I_KH @ self._P_pred @ I_KH.T) + (K @ self.R @ K.T)
        self.P = (self.P + self.P.T) * 0.5

        # Update cached prediction to match posterior
        self._x_pred = self.x.copy()
        self._P_pred = self.P.copy()

        return self.state_to_bbox(self.x)

    @property
    def current_bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Current bounding box coordinates (x1, y1, x2, y2)."""
        return self.state_to_bbox(self.x)

    @property
    def predicted_bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Most recent predicted bounding box coordinates (x1, y1, x2, y2)."""
        return self.state_to_bbox(self._x_pred)

    @property
    def center_xy(self) -> tuple[float, float]:
        """Current bounding box center (cx, cy)."""
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity_xywh(self) -> tuple[float, float, float, float]:
        """Estimated image-space velocity vector (vx, vy, vw, vh) in pixels/sec."""
        return (
            float(self.x[4]),
            float(self.x[5]),
            float(self.x[6]),
            float(self.x[7]),
        )

    @property
    def position_uncertainty_std(self) -> float:
        """Root-mean-square spatial uncertainty (std dev in pixels) of the center position."""
        var_cx = float(self.P[0, 0])
        var_cy = float(self.P[1, 1])
        return math.sqrt(max(0.0, var_cx + var_cy))

