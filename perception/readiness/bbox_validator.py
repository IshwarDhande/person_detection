"""Bounding-box validator for M4 frame-level readiness perception.

Implements Condition 2 — Person Bounding-Box Validity.

Validation pipeline:
    Raw bbox (x1, y1, x2, y2) from YOLO
        |
        v
    [Finite-coordinate check]  -- all 4 coords finite?
        |
        v
    [Positive-dimension check] -- x2 > x1 and y2 > y1?
        |
        v
    [Image-intersection check] -- bbox intersects [0, W) x [0, H)?
        |
        v
    [Clip to image bounds]     -- produce clipped_bbox
        |
        v
    [Area-ratio calculation]   -- A_b / (W_img * H_img)
        |
        v
    Observation(BboxValidityValue, confidence, status) + BboxMetrics
"""

from __future__ import annotations

import math
from typing import Optional

from perception.readiness.observation_types import (
    BboxMetrics,
    BboxValidityValue,
    Observation,
    ObservationStatus,
)


class BboxValidator:
    """Validates person bounding boxes geometrically.

    Validates:
        1. All coordinates are finite.
        2. x2 > x1 (positive width).
        3. y2 > y1 (positive height).
        4. Bounding box intersects the image frame.

    Produces clipped bounding box, area, and area ratio.
    Does NOT enforce minimum/maximum person-box size thresholds (per M4 spec).
    """

    @staticmethod
    def validate(
        bbox_xyxy: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[Observation[BboxValidityValue], Optional[BboxMetrics]]:
        """Validate a bounding box geometrically.

        Args:
            bbox_xyxy:    Raw bounding box (x1, y1, x2, y2) in pixel coords.
            image_width:  Frame width in pixels.
            image_height: Frame height in pixels.

        Returns:
            Tuple of (observation, bbox_metrics).
            bbox_metrics is None when the bbox is invalid.
        """
        x1, y1, x2, y2 = bbox_xyxy

        # Check 1: All coordinates must be finite
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return (
                Observation(
                    value=BboxValidityValue.INVALID,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        # Check 2: Positive dimensions (strict inequality per spec)
        if x2 <= x1 or y2 <= y1:
            return (
                Observation(
                    value=BboxValidityValue.INVALID,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        # Check 3: Bounding box must intersect the image frame
        # Image frame is [0, image_width) x [0, image_height)
        if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
            return (
                Observation(
                    value=BboxValidityValue.INVALID,
                    confidence=None,
                    status=ObservationStatus.INVALID,
                ),
                None,
            )

        # Clip bounding box to image bounds
        cx1 = max(0.0, x1)
        cy1 = max(0.0, y1)
        cx2 = min(float(image_width), x2)
        cy2 = min(float(image_height), y2)

        # Compute metrics from original (unclipped) bbox
        w_b = x2 - x1
        h_b = y2 - y1
        a_b = w_b * h_b
        image_area = float(image_width * image_height)
        a_ratio = a_b / image_area if image_area > 0 else 0.0

        metrics = BboxMetrics(
            original_bbox_xyxy=bbox_xyxy,
            clipped_bbox_xyxy=(cx1, cy1, cx2, cy2),
            width=w_b,
            height=h_b,
            area=a_b,
            area_ratio=a_ratio,
            image_width=image_width,
            image_height=image_height,
        )

        return (
            Observation(
                value=BboxValidityValue.VALID,
                confidence=None,  # Deterministic geometric check — no model confidence
                status=ObservationStatus.VALID,
            ),
            metrics,
        )

