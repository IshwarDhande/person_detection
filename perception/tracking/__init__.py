"""Tracking Subsystem for Milestone M3 Single-Person Tracking.

Exports:
    - ConstantVelocityKalmanFilter8D: 8-D linear Kalman filter with constant-velocity motion model.
    - TrackState: Enum for track lifecycle (TENTATIVE, CONFIRMED, LOST, DELETED).
    - TrackObservation: Compact observation record in bounded history.
    - Track: Minimal single-person tracking state representation.
    - SinglePersonTracker: Single-person motion-based tracker with IoU and uncertainty gating.
    - compute_iou: Box intersection-over-union utility function.
"""

from __future__ import annotations

from perception.tracking.kalman_filter import ConstantVelocityKalmanFilter8D
from perception.tracking.single_person_tracker import (
    SinglePersonTracker,
    Track,
    TrackObservation,
    TrackState,
    compute_iou,
)

__all__ = [
    "ConstantVelocityKalmanFilter8D",
    "TrackState",
    "TrackObservation",
    "Track",
    "SinglePersonTracker",
    "compute_iou",
]

