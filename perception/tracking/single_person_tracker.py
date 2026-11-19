"""Single-Person Tracker Module for Milestone M3.

This module provides the `SinglePersonTracker` class, `Track` data model, `TrackState`
lifecycle enumeration, and IoU association logic according to Milestone M3 specification.

Architecture & State Transitions:
+-----------------------------------------------------------------------------+
|                      M3 SINGLE-PERSON TRACKER PIPELINE                      |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Input: List[PersonDetection], Timestamp t_current                          |
|       |                                                                     |
|       v                                                                     |
|  [Active Track Exists?]                                                     |
|       |                                                                     |
|       +-- NO ---> Valid detection present?                                  |
|       |                 |                                                   |
|       |                 +-- YES -> Initialize new track (TENTATIVE, ID=N)    |
|       |                 +-- NO  -> Return None                              |
|       |                                                                     |
|       +-- YES --> 1. Kalman Predict(dt = t_current - t_prev)                |
|                   2. Association Check:                                     |
|                      - IoU(B_pred, B_det) >= iou_threshold (default: 0.3)   |
|                      - Mahalanobis d_M^2 <= gate (default: 16.0)            |
|                      - Select candidate with highest IoU                    |
|                                                                             |
|                   +-- MATCH FOUND:                                          |
|                   |   - Update Kalman state with measurement                |
|                   |   - Increment consecutive hit_count; miss_count = 0     |
|                   |   - observed = True, last_seen_timestamp = t_current    |
|                   |   - If hit_count >= 3 -> state = CONFIRMED              |
|                   |                                                         |
|                   +-- NO MATCH (missed detection):                          |
|                       - observed = False, hit_count = 0                     |
|                       - miss_count += 1, state = LOST / COASTING            |
|                       - Retain predicted box                                |
|                       - If (t_current - last_seen) > 2.0s -> DELETED        |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import enum
import math
from typing import Optional, Sequence

from perception.detection.person_detector import PersonDetection
from perception.tracking.kalman_filter import ConstantVelocityKalmanFilter8D


class TrackState(enum.Enum):
    """Lifecycle states for a single-person track."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DELETED = "DELETED"


@dataclass(frozen=True)
class TrackObservation:
    """Compact record of a single track observation in the bounded history.

    Stores strictly lightweight numerical geometry and observation status;
    does not store raw image crops, face images, or embeddings.
    """

    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]
    observed: bool


@dataclass
class Track:
    """Minimal single-person tracking state representation for Milestone M3.

    Attributes:
        track_id: Temporary numeric identifier assigned upon track initialization.
        bbox_xyxy: Current observed or predicted bounding box (x1, y1, x2, y2).
        center: Current center coordinate (cx, cy).
        velocity: Estimated image-space velocity vector (vx, vy, vw, vh) in px/s.
        timestamp: Monotonic timestamp of the most recent frame update.
        creation_timestamp: Monotonic timestamp when the track was initialized.
        last_seen_timestamp: Monotonic timestamp of the most recent successful match.
        hit_count: Number of consecutive successful detection matches.
        miss_count: Number of consecutive frames without a detection match.
        state: Current track lifecycle state (TENTATIVE, CONFIRMED, LOST, DELETED).
        observed: True if matched to a real YOLO detection in this frame; False if coasting/predicted.
        history: Bounded observation history (maximum 30 compact observations).
    """

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]
    velocity: tuple[float, float, float, float]
    timestamp: float
    creation_timestamp: float
    last_seen_timestamp: float
    hit_count: int
    miss_count: int
    state: TrackState
    observed: bool
    history: deque[TrackObservation] = field(default_factory=lambda: deque(maxlen=30))

    @property
    def age(self) -> float:
        """Total duration in seconds since track creation."""
        return max(0.0, self.timestamp - self.creation_timestamp)

    @property
    def miss_duration(self) -> float:
        """Duration in seconds since the last successful detection match."""
        return max(0.0, self.timestamp - self.last_seen_timestamp)

    def to_observation(self) -> TrackObservation:
        """Create a compact TrackObservation from the current track state."""
        return TrackObservation(
            timestamp=self.timestamp,
            bbox_xyxy=self.bbox_xyxy,
            center=self.center,
            observed=self.observed,
        )


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Calculate Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        box_a: First box (x1, y1, x2, y2).
        box_b: Second box (x1, y1, x2, y2).

    Returns:
        IoU value in [0.0, 1.0]. Returns 0.0 if boxes do not intersect or are degenerate.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Defensive coordinate order
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection_area = iw * ih

    if intersection_area <= 0.0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return float(intersection_area / union_area)


class SinglePersonTracker:
    """Single-person tracker maintaining temporary identity, motion estimation, and track lifecycle.

    Conforms to Milestone M3:
        - 8-D constant-velocity Kalman filter motion estimation.
        - IoU >= 0.3 association with Mahalanobis prediction-uncertainty gate.
        - 3 consecutive matches for confirmation.
        - Temporary track retention for at most 2.0 seconds during detection loss.
        - Reacquisition within 2.0 seconds preserves track_id.
        - Track expiration removes track state; subsequent detection gets a new ID.
        - Bounded history of at most 30 observations.
        - No face embeddings, appearance models, or multi-person tracking.
    """

    DEFAULT_IOU_THRESHOLD: float = 0.3
    DEFAULT_CONFIRMATION_HITS: int = 3
    DEFAULT_RETENTION_TIMEOUT: float = 2.0
    DEFAULT_MAX_MAHALANOBIS_SQ: float = 16.0  # Chi-square with 4 DoF (~99.7% confidence)
    DEFAULT_HISTORY_MAX_LEN: int = 30

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        confirmation_hits: int = DEFAULT_CONFIRMATION_HITS,
        retention_timeout_seconds: float = DEFAULT_RETENTION_TIMEOUT,
        max_mahalanobis_distance_squared: float = DEFAULT_MAX_MAHALANOBIS_SQ,
        history_max_len: int = DEFAULT_HISTORY_MAX_LEN,
    ) -> None:
        """Initialize the single-person tracker.

        Args:
            iou_threshold: Minimum IoU required for detection association (default: 0.3).
            confirmation_hits: Consecutive hits required for CONFIRMED state (default: 3).
            retention_timeout_seconds: Maximum duration in seconds to retain lost track (default: 2.0).
            max_mahalanobis_distance_squared: Maximum innovation Mahalanobis distance squared (default: 16.0).
            history_max_len: Maximum number of observations stored in track history (default: 30).

        Raises:
            ValueError: If parameters violate domain constraints.
        """
        if not (0.0 < iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be in (0.0, 1.0], got {iou_threshold}")
        if confirmation_hits < 1:
            raise ValueError(f"confirmation_hits must be >= 1, got {confirmation_hits}")
        if retention_timeout_seconds <= 0.0:
            raise ValueError(f"retention_timeout_seconds must be positive, got {retention_timeout_seconds}")
        if max_mahalanobis_distance_squared <= 0.0:
            raise ValueError(
                f"max_mahalanobis_distance_squared must be positive, got {max_mahalanobis_distance_squared}"
            )
        if history_max_len < 1:
            raise ValueError(f"history_max_len must be >= 1, got {history_max_len}")

        self.iou_threshold = float(iou_threshold)
        self.confirmation_hits = int(confirmation_hits)
        self.retention_timeout_seconds = float(retention_timeout_seconds)
        self.max_mahalanobis_distance_squared = float(max_mahalanobis_distance_squared)
        self.history_max_len = int(history_max_len)

        # Internal state
        self._track: Optional[Track] = None
        self._kalman: Optional[ConstantVelocityKalmanFilter8D] = None
        self._next_track_id: int = 1
        self._prev_timestamp: Optional[float] = None
        self._was_confirmed: bool = False
        self._last_matched_detection: Optional[PersonDetection] = None

    @property
    def active_track(self) -> Optional[Track]:
        """Currently active track or None if no track is maintained."""
        return self._track

    @property
    def last_matched_detection(self) -> Optional[PersonDetection]:
        """The detection associated with the active track in the current frame, or None."""
        return self._last_matched_detection

    @property
    def next_track_id(self) -> int:
        """Track ID that will be assigned to the next initialized track."""
        return self._next_track_id

    def _clear_track(self) -> None:
        """Clear active track and Kalman state without resetting the monotonic frame clock."""
        if self._track is not None:
            self._track.state = TrackState.DELETED
        self._track = None
        self._kalman = None
        self._was_confirmed = False
        self._last_matched_detection = None

    def reset(self) -> None:
        """Reset entire tracker state, releasing any active track, Kalman filter, and monotonic clock."""
        self._clear_track()
        self._prev_timestamp = None

    def _init_new_track(self, detection: PersonDetection, timestamp: float) -> Track:
        """Initialize a new tentative track from an initial valid detection."""
        bbox = detection.effective_bbox_xyxy
        track_id = self._next_track_id
        self._next_track_id += 1

        self._kalman = ConstantVelocityKalmanFilter8D(bbox)
        self._was_confirmed = False
        self._last_matched_detection = detection

        track = Track(
            track_id=track_id,
            bbox_xyxy=bbox,
            center=self._kalman.center_xy,
            velocity=self._kalman.velocity_xywh,
            timestamp=timestamp,
            creation_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            hit_count=1,
            miss_count=0,
            state=TrackState.TENTATIVE,
            observed=True,
            history=deque(maxlen=self.history_max_len),
        )
        track.history.append(track.to_observation())
        self._track = track
        return track

    def update(
        self,
        detections: Sequence[PersonDetection],
        timestamp: float,
    ) -> Optional[Track]:
        """Update tracker with current frame detections and monotonic timestamp.

        Args:
            detections: Sequence of PersonDetection objects detected in current frame.
            timestamp: Monotonic timestamp in seconds of the current frame.

        Returns:
            Updated Track object, or None if no track is active or track expired.

        Raises:
            ValueError: If timestamp is non-finite or backwards in time.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp}")

        if self._prev_timestamp is not None and timestamp < self._prev_timestamp:
            raise ValueError(
                f"Monotonic timestamp decreased: prev={self._prev_timestamp}, current={timestamp}"
            )

        # Elapsed time since previous update
        dt = 0.0 if self._prev_timestamp is None else (timestamp - self._prev_timestamp)
        self._prev_timestamp = timestamp

        # Filter detections defensively to valid person class
        valid_detections = [d for d in detections if d.class_id == 0]

        # Case 1: No active track
        if self._track is None:
            if len(valid_detections) == 0:
                return None
            # Initialize track from first valid detection
            return self._init_new_track(valid_detections[0], timestamp)

        # Case 2: Active track exists
        assert self._kalman is not None

        # Predict Kalman state to current timestamp
        predicted_bbox = self._kalman.predict(dt)

        # Association search among candidate detections
        best_candidate: Optional[PersonDetection] = None
        best_iou: float = -1.0

        for det in valid_detections:
            candidate_box = det.effective_bbox_xyxy
            iou = compute_iou(predicted_bbox, candidate_box)
            if iou < self.iou_threshold:
                continue

            d_m_sq = self._kalman.mahalanobis_distance_squared(candidate_box)
            if d_m_sq > self.max_mahalanobis_distance_squared:
                continue

            # Candidate passes both IoU and uncertainty gates
            if iou > best_iou:
                best_iou = iou
                best_candidate = det

        # Subcase 2A: Successful Association Match
        if best_candidate is not None:
            self._last_matched_detection = best_candidate
            matched_bbox = best_candidate.effective_bbox_xyxy
            updated_bbox = self._kalman.update(matched_bbox)

            self._track.bbox_xyxy = updated_bbox
            self._track.center = self._kalman.center_xy
            self._track.velocity = self._kalman.velocity_xywh
            self._track.timestamp = timestamp
            self._track.last_seen_timestamp = timestamp
            self._track.hit_count += 1
            self._track.miss_count = 0
            self._track.observed = True

            # State transition evaluation
            if self._track.state == TrackState.TENTATIVE:
                if self._track.hit_count >= self.confirmation_hits:
                    self._track.state = TrackState.CONFIRMED
                    self._was_confirmed = True
                else:
                    self._track.state = TrackState.TENTATIVE
            elif self._track.state == TrackState.LOST:
                # Reacquisition within timeout preserves track_id
                if self._was_confirmed:
                    self._track.state = TrackState.CONFIRMED
                else:
                    self._track.state = TrackState.TENTATIVE
                    # Tentative track hit_count is now 1 (must re-earn 3 consecutive hits)
                    self._track.hit_count = 1
            elif self._track.state == TrackState.CONFIRMED:
                self._track.state = TrackState.CONFIRMED

            self._track.history.append(self._track.to_observation())
            return self._track

        # Subcase 2B: No Match Found (unobserved / coasting)
        self._last_matched_detection = None
        self._track.bbox_xyxy = predicted_bbox
        self._track.center = self._kalman.center_xy
        self._track.velocity = self._kalman.velocity_xywh
        self._track.timestamp = timestamp
        self._track.hit_count = 0  # Miss breaks consecutive hit sequence
        self._track.miss_count += 1
        self._track.observed = False
        self._track.state = TrackState.LOST

        miss_duration = timestamp - self._track.last_seen_timestamp

        # Check retention timeout (2.0 seconds)
        if miss_duration > self.retention_timeout_seconds:
            # Track expired: delete track state
            self._clear_track()

            # If an unassociated detection is present in this frame, start a new track with new ID
            if len(valid_detections) > 0:
                return self._init_new_track(valid_detections[0], timestamp)

            return None

        # Within retention timeout: retain coasting track
        self._track.history.append(self._track.to_observation())
        return self._track

