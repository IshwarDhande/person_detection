"""Person Detection Module for Vision-Based First Contact.

This module provides the reusable `PersonDetector` class and `PersonDetection`
data representation conforming to Milestone M1 specifications.

Architecture:
+-----------------------------------------------------------------------------+
|                             PERSON DETECTOR (M1)                            |
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
|  |   2. Confidence Filter: conf >= confidence_threshold (default 0.50)   |  |
|  |   3. Validation: Coordinates & confidence finite, valid intervals     |  |
|  +-----------------------------------------------------------------------+  |
|       |                                                                     |
|       v                                                                     |
|  List[PersonDetection]                                                      |
|    - bbox_xyxy: (x1, y1, x2, y2)                                            |
|    - confidence: float in [0.0, 1.0]                                        |
|    - class_id: 0                                                            |
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
        bbox_xyxy: Bounding box coordinates (x1, y1, x2, y2) in pixel space.
        confidence: Model confidence score for the person detection in [0.0, 1.0].
        class_id: COCO class ID; must strictly be 0 for person detections.
    """

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0

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

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def center_xy(self) -> tuple[float, float]:
        """Center coordinate (cx, cy) of the bounding box in pixels."""
        return (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0,
        )


class PersonDetector:
    """Reusable person detector wrapping Ultralytics YOLOv8 for Milestone M1.

    Encapsulates model initialization, frame inference, person-class filtering (class ID 0),
    confidence thresholding, and conversion to PersonDetection records.

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
    ) -> None:
        """Initialize the person detector.

        Args:
            model_path: Path to the YOLOv8 weights file (defaults to 'yolov8n.pt').
            confidence_threshold: Minimum detection confidence in [0.0, 1.0].
            device: Optional PyTorch device string (e.g. 'cpu', 'cuda:0').

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
        self.model: Any = None

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLO model from '{model_path}': {e}") from e

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Perform frame-level person detection on an input image.

        Args:
            frame: Input image as a non-empty numpy.ndarray (typically BGR or RGB).

        Returns:
            List of PersonDetection objects matching COCO class ID 0 and meeting
            the configured confidence threshold. Returns [] if no person is detected.

        Raises:
            RuntimeError: If detector has been closed or if model inference encounters an unrecoverable error.
            ValueError: If input frame is None, not an ndarray, or empty.
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
            int_cls = int(round(float(cls_id)))
            float_conf = float(conf)

            # Defensive verification: retain ONLY COCO class 0 ('person')
            if int_cls != self.COCO_PERSON_CLASS_ID:
                continue

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
        """Release underlying model resources."""
        self.model = None

