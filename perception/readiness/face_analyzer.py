"""Face analyzer for M4 frame-level readiness perception.

Implements Condition 3 — Face Visibility and Quality using MediaPipe
Face Landmarker.

Pipeline:
    Person crop (from clipped bbox)
        |
        v
    [MediaPipe Face Landmarker]
        |
        v
    [Face detected?] -- No --> INSUFFICIENT / VALID
        |
        Yes
        v
    [Face >= 40x40 px?] -- No --> INSUFFICIENT / VALID
        |
        Yes
        v
    [Landmarks available + quality?] -- No --> UNKNOWN / UNCERTAIN
        |
        Yes
        v
    SUFFICIENT / VALID + FaceMetrics + landmarks

Note: Runs on EVERY eligible frame. No periodic skip optimization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from perception.readiness.observation_types import (
    FaceMetrics,
    FaceVisibilityValue,
    Observation,
    ObservationStatus,
)


# MediaPipe Face Landmarker canonical landmark indices used by orientation and quality checks
_NOSE_TIP = 1
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_FOREHEAD = 10
_CHIN = 152
REQUIRED_LANDMARK_INDICES = (_NOSE_TIP, _LEFT_EYE_OUTER, _RIGHT_EYE_OUTER, _FOREHEAD, _CHIN)


class FaceAnalyzer:
    """Face visibility and quality analyzer using MediaPipe Face Landmarker.

    Provides face detection, landmark extraction, and quality assessment
    for the person crop from each eligible frame.

    Required minimum face dimensions: 40 x 40 pixels (configurable).
    """

    # MediaPipe Face Landmarker landmark count for the full mesh
    FULL_MESH_LANDMARK_COUNT = 478

    # Minimum expected landmarks for quality assessment
    MIN_QUALITY_LANDMARKS = 10

    # Default minimum physical distance between eyes in pixels for non-degenerate geometry
    DEFAULT_MIN_EYE_DISTANCE = 4.0

    # Default minimum physical distance from forehead to chin in pixels
    DEFAULT_MIN_VERTICAL_SPAN = 8.0

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        min_face_width: int = 40,
        min_face_height: int = 40,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_eye_distance: float = DEFAULT_MIN_EYE_DISTANCE,
        min_vertical_span: float = DEFAULT_MIN_VERTICAL_SPAN,
    ) -> None:
        """Initialize the face analyzer with MediaPipe Face Landmarker.

        Args:
            model_path:              Path to face_landmarker.task model file.
            min_face_width:          Minimum face width in pixels.
            min_face_height:         Minimum face height in pixels.
            min_detection_confidence: MediaPipe detection confidence threshold.
            min_tracking_confidence:  MediaPipe tracking confidence threshold.
            min_eye_distance:        Minimum horizontal eye-to-eye pixel distance.
            min_vertical_span:       Minimum vertical forehead-to-chin pixel distance.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError:      If MediaPipe initialization fails.
        """
        self.min_face_width = min_face_width
        self.min_face_height = min_face_height
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_eye_distance = min_eye_distance
        self.min_vertical_span = min_vertical_span
        self._landmarker: Any = None
        self._model_path = model_path

        resolved = Path(model_path)
        if not resolved.is_file():
            # Try relative to repo root
            repo_root = Path(__file__).resolve().parents[2]
            if (repo_root / model_path).is_file():
                resolved = repo_root / model_path
            else:
                raise FileNotFoundError(
                    f"Face Landmarker model not found: {model_path}"
                )

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(resolved)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )

            self._landmarker = FaceLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Face Landmarker: {e}") from e

    def analyze(
        self,
        person_crop: np.ndarray,
        crop_offset_x: float = 0.0,
        crop_offset_y: float = 0.0,
    ) -> tuple[
        Observation[FaceVisibilityValue],
        Optional[FaceMetrics],
        Optional[list[tuple[float, float, float]]],
    ]:
        """Analyze face visibility and quality in a person crop.

        Args:
            person_crop:    BGR image crop of the detected person.
            crop_offset_x:  X offset of crop in the full frame (pixels).
            crop_offset_y:  Y offset of crop in the full frame (pixels).

        Returns:
            Tuple of:
                - Observation with SUFFICIENT/INSUFFICIENT/UNKNOWN.
                - FaceMetrics or None.
                - List of (x, y, z) landmark coordinates in full-frame
                  pixel space, or None.
        """
        if self._landmarker is None:
            return (
                Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
                None,
            )

        if person_crop is None or person_crop.size == 0:
            return (
                Observation(
                    value=FaceVisibilityValue.INSUFFICIENT,
                    confidence=None,
                    status=ObservationStatus.VALID,
                ),
                None,
                None,
            )

        crop_h, crop_w = person_crop.shape[:2]
        if crop_h < 1 or crop_w < 1:
            return (
                Observation(
                    value=FaceVisibilityValue.INSUFFICIENT,
                    confidence=None,
                    status=ObservationStatus.VALID,
                ),
                None,
                None,
            )

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_crop = person_crop[:, :, :3]
            if person_crop.shape[2] >= 3:
                rgb_crop = rgb_crop[:, :, ::-1].copy()

            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_crop,
            )

            result = self._landmarker.detect(mp_image)
        except Exception:
            return (
                Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
                None,
            )

        # No face detected
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return (
                Observation(
                    value=FaceVisibilityValue.INSUFFICIENT,
                    confidence=None,
                    status=ObservationStatus.VALID,
                ),
                None,
                None,
            )

        # Extract first (and only, since num_faces=1) face landmarks
        landmarks = result.face_landmarks[0]

        # Convert normalized landmarks to pixel coordinates in crop space
        lm_pixels = []
        for lm in landmarks:
            lx = lm.x * crop_w + crop_offset_x
            ly = lm.y * crop_h + crop_offset_y
            lz = lm.z * crop_w  # z is relative to width
            lm_pixels.append((lx, ly, lz))

        # Compute face bounding box from landmarks
        xs = [p[0] for p in lm_pixels]
        ys = [p[1] for p in lm_pixels]
        face_x1 = min(xs)
        face_y1 = min(ys)
        face_x2 = max(xs)
        face_y2 = max(ys)
        face_w = face_x2 - face_x1
        face_h = face_y2 - face_y1

        # Extract actual model-provided detection confidence ONLY if exposed by the MediaPipe API.
        # In MediaPipe Tasks FaceLandmarkerResult, the API provides face_landmarks without exposing
        # a dedicated detection confidence score. If no genuine model confidence is exposed,
        # the observation confidence is strictly set to None.
        # Under no circumstance is the configured detection threshold used as detection confidence.
        det_confidence: Optional[float] = None
        face_detections = getattr(result, "face_detections", None)
        if face_detections and isinstance(face_detections, (list, tuple)) and len(face_detections) > 0:
            first_det = face_detections[0]
            categories = getattr(first_det, "categories", None)
            if categories and isinstance(categories, (list, tuple)) and len(categories) > 0:
                first_cat = categories[0]
                if hasattr(first_cat, "score"):
                    try:
                        raw_score = float(first_cat.score)
                        if math.isfinite(raw_score) and 0.0 <= raw_score <= 1.0:
                            det_confidence = raw_score
                    except (TypeError, ValueError):
                        pass

        face_metrics = FaceMetrics(
            face_bbox_xyxy=(face_x1, face_y1, face_x2, face_y2),
            face_width=face_w,
            face_height=face_h,
            detection_confidence=det_confidence,
            landmark_count=len(landmarks),
        )

        # Check minimum face dimensions (Condition 3 visibility threshold)
        if face_w < self.min_face_width or face_h < self.min_face_height:
            return (
                Observation(
                    value=FaceVisibilityValue.INSUFFICIENT,
                    confidence=det_confidence,
                    status=ObservationStatus.VALID,
                ),
                face_metrics,
                lm_pixels,
            )

        # Check landmark count
        if len(landmarks) < self.MIN_QUALITY_LANDMARKS:
            return (
                Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=det_confidence,
                    status=ObservationStatus.UNCERTAIN,
                ),
                face_metrics,
                lm_pixels,
            )

        # Check that required landmarks used by orientation estimation exist
        max_required = max(REQUIRED_LANDMARK_INDICES)
        if len(lm_pixels) <= max_required:
            return (
                Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=det_confidence,
                    status=ObservationStatus.INVALID,
                ),
                face_metrics,
                lm_pixels,
            )

        # Check that all required landmark coordinates are finite
        for req_idx in REQUIRED_LANDMARK_INDICES:
            pt = lm_pixels[req_idx]
            if not (math.isfinite(pt[0]) and math.isfinite(pt[1]) and math.isfinite(pt[2])):
                return (
                    Observation(
                        value=FaceVisibilityValue.UNKNOWN,
                        confidence=det_confidence,
                        status=ObservationStatus.INVALID,
                    ),
                    face_metrics,
                    lm_pixels,
                )

        # Check geometric non-degeneracy of key facial features
        left_eye_pt = lm_pixels[_LEFT_EYE_OUTER]
        right_eye_pt = lm_pixels[_RIGHT_EYE_OUTER]
        forehead_pt = lm_pixels[_FOREHEAD]
        chin_pt = lm_pixels[_CHIN]

        eye_distance = abs(left_eye_pt[0] - right_eye_pt[0])
        vertical_span = abs(forehead_pt[1] - chin_pt[1])

        if eye_distance < self.min_eye_distance or vertical_span < self.min_vertical_span:
            # Degenerate or collapsed geometry produces UNCERTAIN status
            return (
                Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=det_confidence,
                    status=ObservationStatus.UNCERTAIN,
                ),
                face_metrics,
                lm_pixels,
            )

        # Check geometric consistency: face should have reasonable aspect ratio
        if face_w <= 0 or face_h <= 0:
            return (
                Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=det_confidence,
                    status=ObservationStatus.UNCERTAIN,
                ),
                face_metrics,
                lm_pixels,
            )

        aspect_ratio = face_w / face_h
        # Extreme aspect ratios indicate poor landmark quality
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return (
                Observation(
                    value=FaceVisibilityValue.UNKNOWN,
                    confidence=det_confidence,
                    status=ObservationStatus.UNCERTAIN,
                ),
                face_metrics,
                lm_pixels,
            )

        # Face detected with adequate quality
        return (
            Observation(
                value=FaceVisibilityValue.SUFFICIENT,
                confidence=det_confidence,
                status=ObservationStatus.VALID,
            ),
            face_metrics,
            lm_pixels,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

