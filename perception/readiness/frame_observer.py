"""Frame-level observation aggregator for M4 readiness perception.

Combines all five observation conditions into a single FrameObservation.

Pipeline:
    RGB Frame + YOLO Detections
        |
        v
    [Condition 1: Person Presence]
        |
        v
    [Condition 2: BBox Validation] --- invalid? --> skip face/orient/zone
        |
        v
    [Condition 4: Interaction Zone] (independent of face analysis)
        |
        v
    [Person Crop Extraction]
        |
        v
    [Condition 3: Face Visibility] --- MediaPipe Face Landmarker
        |
        v
    [Condition 5: Head Orientation] --- from face landmarks
        |
        v
    FrameObservation (structured output)

Independence rules (per M4 spec §7):
    Failures in one observation should not automatically invalidate
    unrelated observations when those can still be evaluated independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ObserverTimings:
    """Detailed component execution latencies for a single frame observation."""
    bbox_ms: float
    zone_ms: float
    face_ms: Optional[float]
    orient_ms: Optional[float]
    total_observer_ms: float

from perception.detection.person_detector import PersonDetection
from perception.readiness.bbox_validator import BboxValidator
from perception.readiness.face_analyzer import FaceAnalyzer
from perception.readiness.interaction_zone import InteractionZoneEvaluator
from perception.readiness.observation_types import (
    BboxMetrics,
    BboxValidityValue,
    FaceMetrics,
    FaceObservations,
    FaceVisibilityValue,
    FrameMetadata,
    FrameObservation,
    HeadOrientationValue,
    InteractionZoneValue,
    Observation,
    ObservationStatus,
    OrientationMetrics,
    OrientationObservations,
    PersonObservations,
    PersonPresenceValue,
)
from perception.readiness.orientation_estimator import OrientationEstimator


class FrameObserver:
    """Aggregates all M4 observation conditions into a FrameObservation.

    Runs the five-condition pipeline on each frame:
        1. Person presence
        2. Bounding-box validation
        3. Face visibility and quality
        4. Interaction-zone membership
        5. Head orientation

    Components:
        +-----------------------------------------------------------+
        |                    FrameObserver                          |
        |                                                           |
        |  +--------------+  +------------------+  +-----------+   |
        |  | BboxValidator|  | FaceAnalyzer     |  | Interact. |   |
        |  |              |  | (MediaPipe)      |  | Zone Eval |   |
        |  +--------------+  +------------------+  +-----------+   |
        |                                                           |
        |  +------------------+                                     |
        |  | OrientEstimator  |                                     |
        |  +------------------+                                     |
        +-----------------------------------------------------------+
    """

    def __init__(
        self,
        face_analyzer: Optional[FaceAnalyzer] = None,
        zone_evaluator: Optional[InteractionZoneEvaluator] = None,
        orientation_estimator: Optional[OrientationEstimator] = None,
    ) -> None:
        """Initialize the frame observer.

        Args:
            face_analyzer:         Initialized FaceAnalyzer (or None to skip face).
            zone_evaluator:        Initialized InteractionZoneEvaluator (or None).
            orientation_estimator: Initialized OrientationEstimator (or None).
        """
        self._face_analyzer = face_analyzer
        self._zone_evaluator = zone_evaluator
        self._orientation_estimator = orientation_estimator
        self._last_timings: Optional[ObserverTimings] = None

    @property
    def last_timings(self) -> Optional[ObserverTimings]:
        """Return the detailed latency timings from the most recent observation."""
        return self._last_timings

    def observe(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        timestamp: Optional[float] = None,
    ) -> FrameObservation:
        """Produce a complete frame-level observation.

        Args:
            frame:      Current BGR frame as numpy array.
            detections: Current-frame YOLO person detections.
            timestamp:  Monotonic timestamp (seconds). Defaults to time.monotonic().

        Returns:
            FrameObservation with all five conditions evaluated.
        """
        t_obs_start = time.perf_counter()
        if timestamp is None:
            timestamp = time.monotonic()

        h_img, w_img = frame.shape[:2]

        frame_meta = FrameMetadata(
            timestamp=timestamp,
            image_width=w_img,
            image_height=h_img,
        )

        bbox_ms = 0.0
        zone_ms = 0.0
        face_ms: Optional[float] = None
        orient_ms: Optional[float] = None

        # --- Condition 1: Person Presence ---
        if detections is None:
            # Detector unavailable
            presence_obs = Observation(
                value=PersonPresenceValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )
            self._last_timings = ObserverTimings(
                bbox_ms=0.0,
                zone_ms=0.0,
                face_ms=None,
                orient_ms=None,
                total_observer_ms=(time.perf_counter() - t_obs_start) * 1000.0,
            )
            return self._build_no_person_observation(presence_obs, frame_meta)

        if len(detections) == 0:
            # Detection ran successfully, no person found
            presence_obs = Observation(
                value=PersonPresenceValue.NOT_DETECTED,
                confidence=None,
                status=ObservationStatus.VALID,
            )
            self._last_timings = ObserverTimings(
                bbox_ms=0.0,
                zone_ms=0.0,
                face_ms=None,
                orient_ms=None,
                total_observer_ms=(time.perf_counter() - t_obs_start) * 1000.0,
            )
            return self._build_no_person_observation(presence_obs, frame_meta)

        # At least one person detected — use highest-confidence detection (Option A)
        best_det = max(detections, key=lambda d: d.confidence)
        presence_obs = Observation(
            value=PersonPresenceValue.DETECTED,
            confidence=best_det.confidence,
            status=ObservationStatus.VALID,
        )

        # --- Condition 2: BBox Validation ---
        t_bbox_0 = time.perf_counter()
        bbox_obs, bbox_metrics = BboxValidator.validate(
            bbox_xyxy=best_det.bbox_xyxy,
            image_width=w_img,
            image_height=h_img,
        )
        bbox_ms = (time.perf_counter() - t_bbox_0) * 1000.0

        # --- Condition 4: Interaction Zone ---
        t_zone_0 = time.perf_counter()
        if bbox_obs.value == BboxValidityValue.VALID and self._zone_evaluator is not None:
            zone_obs = self._zone_evaluator.evaluate(
                bbox_xyxy=best_det.bbox_xyxy,
                image_width=w_img,
                image_height=h_img,
            )
        else:
            zone_obs = Observation(
                value=InteractionZoneValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )
        zone_ms = (time.perf_counter() - t_zone_0) * 1000.0

        person_obs = PersonObservations(
            presence=presence_obs,
            bounding_box=bbox_obs,
            bbox_metrics=bbox_metrics,
            interaction_zone=zone_obs,
        )

        # --- Condition 3 & 5: Face + Orientation ---
        if bbox_obs.value != BboxValidityValue.VALID or bbox_metrics is None:
            # Cannot evaluate face or orientation without valid bbox
            face_obs = FaceObservations(
                visibility=Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                metrics=None,
            )
            orient_obs = OrientationObservations(
                head_orientation=Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                metrics=None,
            )
            self._last_timings = ObserverTimings(
                bbox_ms=bbox_ms,
                zone_ms=zone_ms,
                face_ms=None,
                orient_ms=None,
                total_observer_ms=(time.perf_counter() - t_obs_start) * 1000.0,
            )
            return FrameObservation(
                person=person_obs,
                face=face_obs,
                orientation=orient_obs,
                frame_metadata=frame_meta,
            )

        # Extract person crop from clipped bbox
        cx1, cy1, cx2, cy2 = bbox_metrics.clipped_bbox_xyxy
        ix1 = max(0, int(cx1))
        iy1 = max(0, int(cy1))
        ix2 = min(w_img, int(cx2))
        iy2 = min(h_img, int(cy2))

        if ix2 <= ix1 or iy2 <= iy1:
            face_obs = FaceObservations(
                visibility=Observation(
                    value=FaceVisibilityValue.INSUFFICIENT,
                    confidence=None,
                    status=ObservationStatus.VALID,
                ),
                metrics=None,
            )
            orient_obs = OrientationObservations(
                head_orientation=Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                metrics=None,
            )
            self._last_timings = ObserverTimings(
                bbox_ms=bbox_ms,
                zone_ms=zone_ms,
                face_ms=None,
                orient_ms=None,
                total_observer_ms=(time.perf_counter() - t_obs_start) * 1000.0,
            )
            return FrameObservation(
                person=person_obs,
                face=face_obs,
                orientation=orient_obs,
                frame_metadata=frame_meta,
            )

        person_crop = frame[iy1:iy2, ix1:ix2]

        # --- Condition 3: Face Visibility ---
        landmarks = None
        if self._face_analyzer is not None:
            t_face_0 = time.perf_counter()
            face_visibility_obs, face_metrics, landmarks = self._face_analyzer.analyze(
                person_crop=person_crop,
                crop_offset_x=float(ix1),
                crop_offset_y=float(iy1),
            )
            face_ms = (time.perf_counter() - t_face_0) * 1000.0
        else:
            face_visibility_obs = Observation(
                value=FaceVisibilityValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )
            face_metrics = None

        face_obs = FaceObservations(
            visibility=face_visibility_obs,
            metrics=face_metrics,
        )

        # --- Condition 5: Head Orientation ---
        if (
            landmarks is not None
            and face_visibility_obs.value == FaceVisibilityValue.SUFFICIENT
            and self._orientation_estimator is not None
        ):
            t_orient_0 = time.perf_counter()
            orient_result, orient_metrics = self._orientation_estimator.estimate(landmarks)
            orient_ms = (time.perf_counter() - t_orient_0) * 1000.0
        else:
            if landmarks is not None and face_visibility_obs.value == FaceVisibilityValue.INSUFFICIENT:
                orient_result = Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                )
            elif face_visibility_obs.status == ObservationStatus.UNCERTAIN:
                orient_result = Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.UNCERTAIN,
                )
            else:
                orient_result = Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                )
            orient_metrics = None

        orient_obs = OrientationObservations(
            head_orientation=orient_result,
            metrics=orient_metrics,
        )

        self._last_timings = ObserverTimings(
            bbox_ms=bbox_ms,
            zone_ms=zone_ms,
            face_ms=face_ms,
            orient_ms=orient_ms,
            total_observer_ms=(time.perf_counter() - t_obs_start) * 1000.0,
        )

        return FrameObservation(
            person=person_obs,
            face=face_obs,
            orientation=orient_obs,
            frame_metadata=frame_meta,
        )

    def _build_no_person_observation(
        self,
        presence_obs: Observation[PersonPresenceValue],
        frame_meta: FrameMetadata,
    ) -> FrameObservation:
        """Build a FrameObservation when no person is detected."""
        return FrameObservation(
            person=PersonObservations(
                presence=presence_obs,
                bounding_box=Observation(
                    value=BboxValidityValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                bbox_metrics=None,
                interaction_zone=Observation(
                    value=InteractionZoneValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
            ),
            face=FaceObservations(
                visibility=Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                metrics=None,
            ),
            orientation=OrientationObservations(
                head_orientation=Observation(
                    value=HeadOrientationValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                metrics=None,
            ),
            frame_metadata=frame_meta,
        )

    def close(self) -> None:
        """Release resources held by sub-components."""
        if self._face_analyzer is not None:
            self._face_analyzer.close()

