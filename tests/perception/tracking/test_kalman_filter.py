"""Unit tests for 8-Dimensional Constant-Velocity Kalman Filter (Milestone M3)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.tracking.kalman_filter import ConstantVelocityKalmanFilter8D


class TestKalmanFilterInit(unittest.TestCase):
    """Test initialization and parameter validation of 8-D Kalman filter."""

    def test_state_vector_initialization(self) -> None:
        """Verify initial 8-D state vector: [cx, cy, w, h, 0, 0, 0, 0]."""
        # Box from (100, 200) to (300, 600) => cx=200, cy=400, w=200, h=400
        kf = ConstantVelocityKalmanFilter8D(
            initial_bbox_xyxy=(100.0, 200.0, 300.0, 600.0),
            process_noise_pos=80.0,
            process_noise_size=20.0,
            measurement_noise_pos=10.0,
            measurement_noise_size=20.0,
        )

        self.assertEqual(kf.x.shape, (8,))
        self.assertAlmostEqual(kf.x[0], 200.0)
        self.assertAlmostEqual(kf.x[1], 400.0)
        self.assertAlmostEqual(kf.x[2], 200.0)
        self.assertAlmostEqual(kf.x[3], 400.0)
        self.assertAlmostEqual(kf.x[4], 0.0)
        self.assertAlmostEqual(kf.x[5], 0.0)
        self.assertAlmostEqual(kf.x[6], 0.0)
        self.assertAlmostEqual(kf.x[7], 0.0)

        # Covariance P must be 8x8 symmetric positive-definite
        self.assertEqual(kf.P.shape, (8, 8))
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(kf.P)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_reject_invalid_bounding_boxes(self) -> None:
        """Verify invalid or inverted bounding boxes are rejected."""
        # Inverted coordinates: x1 > x2
        with self.assertRaises(ValueError):
            ConstantVelocityKalmanFilter8D((300.0, 200.0, 100.0, 600.0))

        # Inverted coordinates: y1 > y2
        with self.assertRaises(ValueError):
            ConstantVelocityKalmanFilter8D((100.0, 600.0, 300.0, 200.0))

        # Non-finite coordinates
        with self.assertRaises(ValueError):
            ConstantVelocityKalmanFilter8D((float("nan"), 200.0, 300.0, 600.0))

        # Wrong length
        with self.assertRaises(ValueError):
            ConstantVelocityKalmanFilter8D((100.0, 200.0, 300.0))  # type: ignore

    def test_reject_invalid_noise_parameters(self) -> None:
        """Verify non-positive or non-finite noise parameters raise ValueError."""
        with self.assertRaises(ValueError):
            ConstantVelocityKalmanFilter8D((10, 10, 50, 50), process_noise_pos=0.0)

        with self.assertRaises(ValueError):
            ConstantVelocityKalmanFilter8D((10, 10, 50, 50), measurement_noise_pos=-5.0)

        with self.assertRaises(ValueError):
            ConstantVelocityKalmanFilter8D((10, 10, 50, 50), process_noise_size=float("inf"))


class TestKalmanFilterPrediction(unittest.TestCase):
    """Test time-update prediction with monotonic elapsed time dt."""

    def test_constant_velocity_prediction(self) -> None:
        """Verify state advances according to constant-velocity model: q_t = q_(t-1) + dt * v_q."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 300.0))
        # Inject velocity manually for test: vx=100 px/s, vy=-50 px/s
        kf.x[4] = 100.0
        kf.x[5] = -50.0

        # Predict with dt = 0.5s => cx should move +50px, cy should move -25px
        initial_cx = kf.x[0]
        initial_cy = kf.x[1]

        pred_box = kf.predict(dt=0.5)
        pred_cx = (pred_box[0] + pred_box[2]) / 2.0
        pred_cy = (pred_box[1] + pred_box[3]) / 2.0

        self.assertAlmostEqual(pred_cx, initial_cx + 50.0, delta=1e-3)
        self.assertAlmostEqual(pred_cy, initial_cy - 25.0, delta=1e-3)

        # Covariance grows after prediction
        init_pos_std = math.sqrt(kf.r_pos_std ** 2)
        self.assertGreater(kf.position_uncertainty_std, init_pos_std)

    def test_zero_elapsed_time_prediction(self) -> None:
        """Verify dt=0 preserves state without error."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 300.0))
        initial_box = kf.current_bbox_xyxy

        pred_box = kf.predict(dt=0.0)
        self.assertAlmostEqual(pred_box[0], initial_box[0])
        self.assertAlmostEqual(pred_box[1], initial_box[1])
        self.assertAlmostEqual(pred_box[2], initial_box[2])
        self.assertAlmostEqual(pred_box[3], initial_box[3])

    def test_reject_negative_or_non_finite_dt(self) -> None:
        """Verify negative or non-finite dt raises ValueError."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 300.0))
        with self.assertRaises(ValueError):
            kf.predict(dt=-0.1)

        with self.assertRaises(ValueError):
            kf.predict(dt=float("nan"))


class TestKalmanFilterUpdate(unittest.TestCase):
    """Test measurement correction and covariance reduction."""

    def test_measurement_update_pulls_state(self) -> None:
        """Verify matched measurement pulls posterior state towards observation."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 200.0))
        # Initial center: cx=150, cy=150, w=100, h=100
        kf.predict(dt=0.1)

        # Measurement offset: box at (120, 120, 220, 220) => cx=170, cy=170
        meas_box = (120.0, 120.0, 220.0, 220.0)
        post_box = kf.update(meas_box)

        post_cx = (post_box[0] + post_box[2]) / 2.0
        post_cy = (post_box[1] + post_box[3]) / 2.0

        # State should be updated between prior prediction (150) and measurement (170)
        self.assertGreater(post_cx, 150.0)
        self.assertLess(post_cx, 175.0)
        self.assertGreater(post_cy, 150.0)
        self.assertLess(post_cy, 175.0)

        # Velocity should become positive in both x and y
        self.assertGreater(kf.velocity_xywh[0], 0.0)
        self.assertGreater(kf.velocity_xywh[1], 0.0)

        # Joseph form guarantees symmetry
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-10)

    def test_mahalanobis_distance_squared(self) -> None:
        """Verify Mahalanobis distance is zero for exact predicted match and increases with error."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 200.0))
        kf.predict(dt=0.1)

        pred_box = kf.predicted_bbox_xyxy
        d2_exact = kf.mahalanobis_distance_squared(pred_box)
        self.assertAlmostEqual(d2_exact, 0.0, delta=1e-5)

        # Small offset (10px)
        offset_box = (pred_box[0] + 10.0, pred_box[1] + 10.0, pred_box[2] + 10.0, pred_box[3] + 10.0)
        d2_small = kf.mahalanobis_distance_squared(offset_box)
        self.assertGreater(d2_small, 0.0)

        # Large offset (300px)
        far_box = (pred_box[0] + 300.0, pred_box[1] + 300.0, pred_box[2] + 300.0, pred_box[3] + 300.0)
        d2_large = kf.mahalanobis_distance_squared(far_box)
        self.assertGreater(d2_large, d2_small)


class TestKalmanFilterMathematicalStructure(unittest.TestCase):
    """Detailed mathematical verification of matrices, models, and numerical validity."""

    def test_state_transition_matrix_structure(self) -> None:
        """Verify F(dt) structure: I_4 top-left, dt*I_4 top-right, 0 bottom-left, I_4 bottom-right."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 200.0))
        dt = 0.05
        F = kf._build_transition_matrix(dt)

        self.assertEqual(F.shape, (8, 8))
        # Top-left and bottom-right 4x4 blocks must be identity
        np.testing.assert_allclose(F[:4, :4], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(F[4:, 4:], np.eye(4), atol=1e-12)
        # Bottom-left 4x4 block must be zeros
        np.testing.assert_allclose(F[4:, :4], np.zeros((4, 4)), atol=1e-12)
        # Top-right 4x4 block must be dt * I_4
        np.testing.assert_allclose(F[:4, 4:], dt * np.eye(4), atol=1e-12)

        # Explicit state pair verification: (pos, vel)
        # cx <-> vx at (0, 4)
        self.assertAlmostEqual(F[0, 4], dt)
        # cy <-> vy at (1, 5)
        self.assertAlmostEqual(F[1, 5], dt)
        # w <-> vw at (2, 6)
        self.assertAlmostEqual(F[2, 6], dt)
        # h <-> vh at (3, 7)
        self.assertAlmostEqual(F[3, 7], dt)

    def test_process_noise_matrix_structure_and_cwna_blocks(self) -> None:
        """Verify Q(dt) CWNA blocks match state pairs and matrix is symmetric positive-definite."""
        kf = ConstantVelocityKalmanFilter8D(
            (100.0, 100.0, 200.0, 200.0),
            process_noise_pos=80.0,
            process_noise_size=20.0,
        )
        dt = 0.1
        Q = kf._build_process_noise(dt)

        self.assertEqual(Q.shape, (8, 8))
        # Must be symmetric
        np.testing.assert_allclose(Q, Q.T, atol=1e-12)

        # CWNA block values: dt^3/3, dt^2/2, dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        q_pos_var = 80.0 ** 2
        q_size_var = 20.0 ** 2

        # cx <-> vx at (0, 4)
        self.assertAlmostEqual(Q[0, 0], q_pos_var * (dt3 / 3.0))
        self.assertAlmostEqual(Q[0, 4], q_pos_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[4, 0], q_pos_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[4, 4], q_pos_var * dt)

        # cy <-> vy at (1, 5)
        self.assertAlmostEqual(Q[1, 1], q_pos_var * (dt3 / 3.0))
        self.assertAlmostEqual(Q[1, 5], q_pos_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[5, 1], q_pos_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[5, 5], q_pos_var * dt)

        # w <-> vw at (2, 6)
        self.assertAlmostEqual(Q[2, 2], q_size_var * (dt3 / 3.0))
        self.assertAlmostEqual(Q[2, 6], q_size_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[6, 2], q_size_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[6, 6], q_size_var * dt)

        # h <-> vh at (3, 7)
        self.assertAlmostEqual(Q[3, 3], q_size_var * (dt3 / 3.0))
        self.assertAlmostEqual(Q[3, 7], q_size_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[7, 3], q_size_var * (dt2 / 2.0))
        self.assertAlmostEqual(Q[7, 7], q_size_var * dt)

        # Eigenvalues must be non-negative
        eigvals = np.linalg.eigvalsh(Q)
        self.assertTrue(np.all(eigvals >= -1e-12))

    def test_measurement_matrix_and_innovation_covariance(self) -> None:
        """Verify H matrix extracts [cx, cy, w, h] and S = H P^- H^T + R."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 200.0))
        kf.predict(dt=0.033)

        # H is 4x8: [I_4, 0_4]
        self.assertEqual(kf.H.shape, (4, 8))
        np.testing.assert_allclose(kf.H[:, :4], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(kf.H[:, 4:], np.zeros((4, 4)), atol=1e-12)

        # S = H P^- H^T + R
        S_expected = kf.H @ kf._P_pred @ kf.H.T + kf.R
        S_actual = kf.innovation_covariance()
        np.testing.assert_allclose(S_actual, S_expected, atol=1e-12)

    def test_mahalanobis_gate_uses_same_innovation_covariance(self) -> None:
        """Verify mahalanobis_distance_squared strictly computes y^T S^(-1) y with S = innovation_covariance()."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 200.0))
        kf.predict(dt=0.033)

        cand_box = (110.0, 105.0, 210.0, 205.0)
        d_m_sq = kf.mahalanobis_distance_squared(cand_box)

        # Manual computation
        z = kf.bbox_to_z(cand_box)
        y = z - (kf.H @ kf._x_pred)
        S = kf.innovation_covariance()
        d_m_sq_manual = float(y.T @ np.linalg.inv(S) @ y)

        self.assertAlmostEqual(d_m_sq, d_m_sq_manual, places=7)

    def test_covariance_numerical_stability_across_multiple_steps(self) -> None:
        """Verify covariance P remains symmetric and positive-definite across 100 predict/update steps."""
        kf = ConstantVelocityKalmanFilter8D((100.0, 100.0, 200.0, 200.0))
        for step in range(100):
            kf.predict(dt=0.033)
            # Simulated measurement with small drift
            meas_box = (
                100.0 + step * 0.5,
                100.0 + step * 0.3,
                200.0 + step * 0.5,
                200.0 + step * 0.3,
            )
            kf.update(meas_box)

            # Symmetry check
            np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-9)
            # Positive-definiteness check
            min_eig = float(np.min(np.linalg.eigvalsh(kf.P)))
            self.assertGreater(min_eig, 0.0)


if __name__ == "__main__":
    unittest.main()

