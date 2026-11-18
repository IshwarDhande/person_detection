"""Person Detection Module for Vision-Based First Contact.

This module provides the reusable `PersonDetector` class, `SinglePersonSmoother` (EMA),
and `PersonDetection` data representation conforming to Milestone M1 and M2 specifications.

Architecture:
+-----------------------------------------------------------------------------+
|                     PERSON DETECTOR WITH EMA SMOOTHING (M2)                 |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Input Frame (RGB/BGR np.ndarray)                                           |
|       |                                                                     |
|       v                                                                     |
|  [Ultralytics YOLOv8n]                                                      |
|       |                                                                     |
|       v Raw Candidate Detections (xyxy, conf, cls)                          |
|  +-----------------------------------------------------------------------+  |
|  | Post-Processing & Filtering:                                          |  |
|  |   1. Class Filter: cls == 0 (COCO person only; discard cars, pets)    |  |
|  |   2. Confidence Filter: conf >= confidence_threshold (fixed at 0.50)  |  |
|  |   3. Validation: Coordinates & confidence finite, valid intervals     |  |
|  +-----------------------------------------------------------------------+  |
|       |                                                                     |
|       v Raw Detections: list[PersonDetection]                               |
|  +-----------------------------------------------------------------------+  |
|  | SinglePersonSmoother (EMA):                                            |  |
|  |   * Count == 0: Reset smoother state -> Return []                       |  |
|  |   * Count == 1: Apply EMA: b_hat_t = a*b_t + (1-a)*b_hat_(t-1)         |  |
|  |   * Count > 1:  Bypass smoothing, Reset state -> Return raw detections  |  |
|  +-----------------------------------------------------------------------+  |
|       |                                                                     |
|       v                                                                     |
|  List[PersonDetection]                                                      |
|    - bbox_xyxy: raw (x1, y1, x2, y2)                                        |
|    - confidence: float in [0.0, 1.0]                                        |
|    - class_id: 0                                                            |
|    - smoothed_bbox_xyxy: (sx1, sy1, sx2, sy2) or None                       |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PersonDetection:
    """Frame-level detection representation for a person candidate.

    This representation is strictly per-frame: it holds no persistent identity,
    tracking ID, trajectory history, or 3D spatial attributes.

    Attributes:
        bbox_xyxy: Raw bounding box coordinates (x1, y1, x2, y2) in pixel space.
        confidence: Model confidence score for the person detection in [0.0, 1.0].
        class_id: COCO class ID; must strictly be 0 for person detections.
        smoothed_bbox_xyxy: Optional EMA-smoothed bounding box coordinates (x1, y1, x2, y2).
                            Populated only in unambiguous single-person scenarios.
    """

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    smoothed_bbox_xyxy: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """Validate bounding box coordinates, confidence, and class identifier."""
        if self.class_id != 0:
            raise ValueError(f"PersonDetection must have class_id=0 (COCO person), got {self.class_id}")

        if not math.isfinite(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be a finite float in [0.0, 1.0], got {self.confidence}")

        if len(self.bbox_xyxy) != 4:
            raise ValueError(
                f"bbox_xyxy must contain exactly 4 coordinates (x1, y1, x2, y2), got {len(self.bbox_xyxy)}"
            )

        for i, val in enumerate(self.bbox_xyxy):
            if not math.isfinite(val):
                raise ValueError(f"bbox_xyxy coordinate at index {i} is not finite: {val}")

        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 > x2 or y1 > y2:
            raise ValueError(
                f"Invalid bounding box: inverted coordinates (x1={x1}, y1={y1}, x2={x2}, y2={y2})"
            )

        if self.smoothed_bbox_xyxy is not None:
            if len(self.smoothed_bbox_xyxy) != 4:
                raise ValueError(
                    f"smoothed_bbox_xyxy must contain exactly 4 coordinates, got {len(self.smoothed_bbox_xyxy)}"
                )
            for i, val in enumerate(self.smoothed_bbox_xyxy):
                if not math.isfinite(val):
                    raise ValueError(f"smoothed_bbox_xyxy coordinate at index {i} is not finite: {val}")
            sx1, sy1, sx2, sy2 = self.smoothed_bbox_xyxy
            if sx1 > sx2 or sy1 > sy2:
                raise ValueError(
                    f"Invalid smoothed bounding box: inverted coordinates (sx1={sx1}, sy1={sy1}, sx2={sx2}, sy2={sy2})"
                )

    @property
    def width(self) -> float:
        """Width of the raw bounding box in pixels."""
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        """Height of the raw bounding box in pixels."""
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def center_xy(self) -> tuple[float, float]:
        """Center coordinate (cx, cy) of the raw bounding box in pixels."""
        return (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0,
        )

    @property
    def is_smoothed(self) -> bool:
        """Whether an EMA-smoothed bounding box is present."""
        return self.smoothed_bbox_xyxy is not None

    @property
    def effective_bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Return smoothed bounding box if available, otherwise raw bounding box."""
        return self.smoothed_bbox_xyxy if self.smoothed_bbox_xyxy is not None else self.bbox_xyxy


class SinglePersonSmoother:
    """Lightweight Exponential Moving Average (EMA) smoother for single-person bounding boxes.

    Formula:
        b_hat_t = alpha * b_t + (1 - alpha) * b_hat_(t-1)
    where:
        - b_t is the raw bounding box (x1, y1, x2, y2)
        - b_hat_(t-1) is the previous smoothed bounding box
        - alpha is the smoothing factor in (0.0, 1.0]

    Behaviour across detection counts:
        - 0 detections: Immediately reports no detection and resets internal smoothing state.
        - 1 detection: Applies EMA smoothing, initializing from the first detection.
        - >1 detections: Bypasses smoothing, preserves raw detections, and resets smoothing state.
    """

    DEFAULT_ALPHA: float = 0.65

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        """Initialize the EMA smoother.

        Args:
            alpha: Smoothing coefficient in (0.0, 1.0]. Defaults to 0.65.

        Raises:
            ValueError: If alpha is not a finite float in (0.0, 1.0].
        """
        if (
            not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or not (0.0 < alpha <= 1.0)
        ):
            raise ValueError(f"alpha must be a finite float in (0.0, 1.0], got {alpha}")

        self.alpha = float(alpha)
        self._smoothed_box: Optional[tuple[float, float, float, float]] = None

    @property
    def is_initialized(self) -> bool:
        """Whether the smoother has an active smoothed bounding box."""
        return self._smoothed_box is not None

    @property
    def current_smoothed_box(self) -> Optional[tuple[float, float, float, float]]:
        """Current smoothed box coordinates or None if uninitialized/reset."""
        return self._smoothed_box

    def reset(self) -> None:
        """Reset internal smoothing state."""
        self._smoothed_box = None

    def smooth(self, raw_detections: list[PersonDetection]) -> list[PersonDetection]:
        """Apply single-person EMA smoothing to raw detections.

        Args:
            raw_detections: List of PersonDetection objects from current frame inference.

        Returns:
            List of PersonDetection objects:
            - [] if raw_detections is empty (smoothing state reset).
            - [PersonDetection with smoothed_bbox_xyxy] if len(raw_detections) == 1.
            - raw_detections as-is (with smoothed_bbox_xyxy=None) if len(raw_detections) > 1 (smoothing state reset).
        """
        if len(raw_detections) == 0:
            self.reset()
            return []

        if len(raw_detections) > 1:
            # Multi-person scenario: bypass smoothing and reset state
            self.reset()
            return list(raw_detections)

        # Unambiguous single-person detection
        raw_det = raw_detections[0]
        x1, y1, x2, y2 = raw_det.bbox_xyxy

        if self._smoothed_box is None:
            # First detection initializes the smoothed box directly
            sx1, sy1, sx2, sy2 = float(x1), float(y1), float(x2), float(y2)
        else:
            prev_x1, prev_y1, prev_x2, prev_y2 = self._smoothed_box
            a = self.alpha
            sx1 = a * x1 + (1.0 - a) * prev_x1
            sy1 = a * y1 + (1.0 - a) * prev_y1
            sx2 = a * x2 + (1.0 - a) * prev_x2
            sy2 = a * y2 + (1.0 - a) * prev_y2

        # Defensively ensure finite and correctly ordered
        if not (math.isfinite(sx1) and math.isfinite(sy1) and math.isfinite(sx2) and math.isfinite(sy2)):
            self.reset()
            return [raw_det]

        if sx1 > sx2:
            sx1, sx2 = sx2, sx1
        if sy1 > sy2:
            sy1, sy2 = sy2, sy1

        smoothed_box = (float(sx1), float(sy1), float(sx2), float(sy2))
        self._smoothed_box = smoothed_box

        smoothed_det = PersonDetection(
            bbox_xyxy=raw_det.bbox_xyxy,
            confidence=raw_det.confidence,
            class_id=raw_det.class_id,
            smoothed_bbox_xyxy=smoothed_box,
        )
        return [smoothed_det]


class PersonDetector:
    """Reusable person detector wrapping Ultralytics YOLOv8 for Milestone M1 and M2.

    Encapsulates model initialization, frame inference, person-class filtering (class ID 0),
    confidence thresholding, and optional single-person EMA smoothing.

    ```text
    +------------------+         +--------------------+         +-----------------------+
    |  OpenCV Frame    | ------> | PersonDetector     | ------> | list[PersonDetection] |
    |  (720x1280 BGR)  |         | (YOLOv8 Inference) |         | (class_id=0, conf>=T) |
    +------------------+         +--------------------+         +-----------------------+
    ```
    """

    COCO_PERSON_CLASS_ID: int = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        smoother: Optional[SinglePersonSmoother] = None,
    ) -> None:
        """Initialize the person detector.

        Args:
            model_path: Path to the YOLOv8 weights file (defaults to 'yolov8n.pt').
            confidence_threshold: Minimum detection confidence in [0.0, 1.0] (fixed at 0.5 for M1/M2).
            device: Optional PyTorch device string (e.g. 'cpu', 'cuda:0').
            smoother: Optional SinglePersonSmoother instance for EMA bounding-box stabilization.

        Raises:
            FileNotFoundError: If the specified model_path does not exist.
            ValueError: If confidence_threshold is outside [0.0, 1.0].
            RuntimeError: If the YOLO model fails to initialize.
        """
        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            # Fallback to checking relative to the repository root
            repo_root = Path(__file__).resolve().parents[2]
            if (repo_root / model_path).is_file():
                resolved_path = repo_root / model_path
            else:
                raise FileNotFoundError(f"Model weights file not found: {model_path}")

        if (
            not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(confidence_threshold)
            or not (0.0 <= confidence_threshold <= 1.0)
        ):
            raise ValueError(
                f"confidence_threshold must be a finite float in [0.0, 1.0], got {confidence_threshold}"
            )

        self.model_path = str(resolved_path)
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.smoother = smoother
        self.model: Any = None

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLO model from '{model_path}': {e}") from e

    def detect_raw(self, frame: np.ndarray) -> list[PersonDetection]:
        """Perform frame-level person detection without applying any temporal smoothing.

        Args:
            frame: Input image as a non-empty numpy.ndarray.

        Returns:
            List of raw PersonDetection objects meeting COCO class ID 0 and confidence threshold.
        """
        if self.model is None:
            raise RuntimeError("PersonDetector has been closed or model is not initialized.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: expected a non-empty numpy.ndarray")

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=[self.COCO_PERSON_CLASS_ID],
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {e}") from e

        return self._parse_results(results)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Perform frame-level person detection, applying smoothing if configured.

        If self.smoother is configured, single-person EMA smoothing is applied.
        Otherwise, returns raw detections directly.

        Args:
            frame: Input image as a non-empty numpy.ndarray (typically BGR or RGB).

        Returns:
            List of PersonDetection objects matching COCO class ID 0 and meeting
            the configured confidence threshold. Returns [] if no person is detected.

        Raises:
            RuntimeError: If detector has been closed or if model inference encounters an unrecoverable error.
            ValueError: If input frame is None, not an ndarray, or empty.
        """
        raw_detections = self.detect_raw(frame)
        if self.smoother is not None:
            return self.smoother.smooth(raw_detections)
        return raw_detections

    def _parse_results(self, results: Any) -> list[PersonDetection]:
        """Parse raw model inference results into validated PersonDetection objects.

        Defensively verifies that only class 0 detections exceeding the confidence
        threshold are retained, and converts coordinates into float pixel values.

        Args:
            results: Ultralytics Results list or object.

        Returns:
            List of validated PersonDetection instances.
        """
        if not results:
            return []

        first_result = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(first_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[PersonDetection] = []

        try:
            xyxy_raw = (
                boxes.xyxy.cpu().numpy()
                if hasattr(boxes.xyxy, "cpu")
                else np.asarray(boxes.xyxy)
            )
            conf_raw = (
                boxes.conf.cpu().numpy()
                if hasattr(boxes.conf, "cpu")
                else np.asarray(boxes.conf)
            )
            cls_raw = (
                boxes.cls.cpu().numpy()
                if hasattr(boxes.cls, "cpu")
                else np.asarray(boxes.cls)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract bounding box tensors from inference output: {e}") from e

        for xyxy, conf, cls_id in zip(xyxy_raw, conf_raw, cls_raw):
            try:
                raw_cls_float = float(cls_id)
            except (TypeError, ValueError):
                continue

            # Reject non-finite class IDs (NaN, +inf, -inf) safely
            if not math.isfinite(raw_cls_float):
                continue

            # Reject non-integer class IDs (e.g. 0.5, 0.49). Never round into class 0.
            if not raw_cls_float.is_integer():
                continue

            int_cls = int(raw_cls_float)

            # Defensive verification: retain ONLY exact COCO class 0 ('person')
            if int_cls != self.COCO_PERSON_CLASS_ID:
                continue

            float_conf = float(conf)

            # Defensive verification: retain ONLY detections >= confidence threshold
            if float_conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = (
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3]),
            )

            detections.append(
                PersonDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float_conf,
                    class_id=int_cls,
                )
            )

        return detections

    def close(self) -> None:
        """Release underlying model resources and reset smoother state."""
        if self.smoother is not None:
            self.smoother.reset()
        self.model = None
