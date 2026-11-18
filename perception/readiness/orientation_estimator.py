"""Head orientation estimator for M4 frame-level readiness perception.

Implements Condition 5 — Robot-Facing Head Orientation using facial
landmarks from MediaPipe Face Landmarker.

Orientation estimation approach:
    Uses geometric ratios of key facial landmarks to classify head pose
    as FACING_ROBOT, NOT_FACING_ROBOT, or UNKNOWN.

    Yaw estimation (horizontal turn):
        Uses nose tip, left eye outer corner, right eye outer corner.
        Computes asymmetry ratio between nose-to-left-eye and
        nose-to-right-eye horizontal distances.

    Pitch estimation (vertical tilt):
        Uses nose tip, forehead (between eyebrows), and chin.
        Computes ratio of nose-to-forehead vs nose-to-chin distances.

    Key MediaPipe Face Landmarker indices:
        1   = nose tip
        33  = left eye outer corner
        263 = right eye outer corner
        10  = forehead (top of face)
        152 = chin
        234 = left cheek
        454 = right cheek

Decision boundaries:
    +----------------------------------+
    |          FACING_ROBOT            |
    |   |yaw_ratio| < yaw_threshold   |
    |   |pitch_ratio| < pitch_thresh   |
    +----------------------------------+
    |          UNCERTAIN               |
    |   threshold < ratio < 1.5*thresh |
    +----------------------------------+
    |       NOT_FACING_ROBOT           |
    |   ratio >= 1.5 * threshold       |
    +----------------------------------+
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from perception.readiness.observation_types import (
    HeadOrientationValue,
    Observation,
    ObservationStatus,
    OrientationMetrics,
)


# MediaPipe Face Landmarker canonical landmark indices
_NOSE_TIP = 1
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_FOREHEAD = 10
_CHIN = 152
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


class OrientationEstimator:
    """Estimates approximate head orientation from facial landmarks.

    Classifies head pose as FACING_ROBOT, NOT_FACING_ROBOT, or UNKNOWN
    based on geometric ratios of key landmark positions.

    Does NOT provide precise head-pose angles — this is an approximate
    orientation indicator per M4 specification.
    """

    # Default thresholds (experimentally tunable)
    DEFAULT_YAW_THRESHOLD = 0.35
    DEFAULT_PITCH_THRESHOLD = 0.60
    DEFAULT_UNCERTAIN_MULTIPLIER = 1.5

    def __init__(
        self,
        yaw_threshold: float = DEFAULT_YAW_THRESHOLD,
        pitch_threshold: float = DEFAULT_PITCH_THRESHOLD,
        uncertain_multiplier: float = DEFAULT_UNCERTAIN_MULTIPLIER,
    ) -> None:
        """Initialize the orientation estimator.

        Args:
            yaw_threshold:         Max yaw ratio for FACING_ROBOT.
            pitch_threshold:       Max pitch ratio for FACING_ROBOT.
            uncertain_multiplier:  Factor applied to thresholds for the
                                   UNCERTAIN boundary (between FACING and
                                   NOT_FACING).

        Raises:
            ValueError: If thresholds are non-positive or non-finite.
        """
        if not (math.isfinite(yaw_threshold) and yaw_threshold > 0):
            raise ValueError(f"yaw_threshold must be positive and finite, got {yaw_threshold}")
        if not (math.isfinite(pitch_threshold) and pitch_threshold > 0):
            raise ValueError(f"pitch_threshold must be positive and finite, got {pitch_threshold}")
        if not (math.isfinite(uncertain_multiplier) and uncertain_multiplier > 1.0):
            raise ValueError(
                f"uncertain_multiplier must be > 1.0 and finite, got {uncertain_multiplier}"
            )

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.uncertain_multiplier = uncertain_multiplier

    def estimate(
        self,
        landmarks: Sequence[tuple[float, float, float]],
    ) -> tuple[Observation[HeadOrientationValue], Optional[OrientationMetrics]]:
        """Estimate head orientation from facial landmarks.

        Args:
            landmarks: List of (x, y, z) landmark coordinates in pixel space.
                       Must have at least max(required indices) + 1 elements.

        Returns:
            Tuple of (observation, orientation_metrics).
        """
        if landmarks is None or len(landmarks) == 0:
            return (
                Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        # Check we have enough landmarks
        required_max = max(_NOSE_TIP, _LEFT_EYE_OUTER, _RIGHT_EYE_OUTER,
                          _FOREHEAD, _CHIN)
        if len(landmarks) <= required_max:
            return (
                Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        try:
            nose = landmarks[_NOSE_TIP]
            left_eye = landmarks[_LEFT_EYE_OUTER]
            right_eye = landmarks[_RIGHT_EYE_OUTER]
            forehead = landmarks[_FOREHEAD]
            chin = landmarks[_CHIN]
        except (IndexError, TypeError):
            return (
                Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        # Validate that all coords are finite
        for point in (nose, left_eye, right_eye, forehead, chin):
            if not all(math.isfinite(v) for v in point[:2]):
                return (
                    Observation(
                        value=HeadOrientationValue.UNKNOWN,
                        confidence=None,
                        status=ObservationStatus.INVALID,
                    ),
                    None,
                )

        # --- Yaw estimation ---
        # Distance from nose to left eye vs nose to right eye (horizontal)
        d_nose_left = abs(nose[0] - left_eye[0])
        d_nose_right = abs(nose[0] - right_eye[0])

        # Denominator validation: horizontal distance sum cannot be degenerate
        d_sum = d_nose_left + d_nose_right
        if d_sum < 1e-5 or not math.isfinite(d_sum):
            return (
                Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        # Yaw ratio: asymmetry between left and right nose-to-eye distances
        # 0 = perfectly frontal, 1 = nose overlaps one eye (extreme turn)
        yaw_ratio = abs(d_nose_left - d_nose_right) / d_sum

        # --- Pitch estimation ---
        # Distance from nose to forehead vs nose to chin (vertical)
        d_nose_forehead = abs(nose[1] - forehead[1])
        d_nose_chin = abs(nose[1] - chin[1])

        # Denominator validation: vertical distance sum cannot be degenerate
        d_vert_sum = d_nose_forehead + d_nose_chin
        if d_vert_sum < 1e-5 or not math.isfinite(d_vert_sum):
            return (
                Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        pitch_ratio = abs(d_nose_forehead - d_nose_chin) / d_vert_sum

        # Defensive check: ensure computed ratios are finite numbers
        if not (math.isfinite(yaw_ratio) and math.isfinite(pitch_ratio)):
            return (
                Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        metrics = OrientationMetrics(
            yaw_ratio=yaw_ratio,
            pitch_ratio=pitch_ratio,
            yaw_threshold=self.yaw_threshold,
            pitch_threshold=self.pitch_threshold,
        )

        # --- Classification ---
        yaw_uncertain_bound = self.yaw_threshold * self.uncertain_multiplier
        pitch_uncertain_bound = self.pitch_threshold * self.uncertain_multiplier

        # Clear turn detected
        if yaw_ratio >= yaw_uncertain_bound or pitch_ratio >= pitch_uncertain_bound:
            return (
                Observation(
                    value=HeadOrientationValue.NOT_FACING_ROBOT,
                    confidence=None,
                    status=ObservationStatus.VALID,
                ),
                metrics,
            )

        # Ambiguous zone
        if yaw_ratio >= self.yaw_threshold or pitch_ratio >= self.pitch_threshold:
            return (
                Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.UNCERTAIN,
                ),
                metrics,
            )

        # Approximately frontal
        return (
            Observation(
                value=HeadOrientationValue.FACING_ROBOT,
                confidence=None,
                status=ObservationStatus.VALID,
            ),
            metrics,
        )

