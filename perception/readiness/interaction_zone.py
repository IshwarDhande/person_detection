"""Interaction-zone evaluator for M4 frame-level readiness perception.

Implements Condition 4 — Image-Space Interaction-Zone Membership.

Membership rule (per M4 spec):
    A person is INSIDE only when the ENTIRE bounding box lies inside
    the configured polygon.  B ⊆ P => INSIDE; B ⊄ P => OUTSIDE.

    All 4 corners of the bounding box must be inside the polygon.

Polygon coordinates are stored in normalized [0,1] space in YAML config
and converted to pixel coordinates for evaluation.

Pipeline:
    Normalized polygon vertices (from YAML)
        |
        v
    [Convert to pixel coords] -- x_px = x_n * W, y_px = y_n * H
        |
        v
    [Point-in-polygon for all 4 bbox corners]
        |
        v
    INSIDE / OUTSIDE / UNKNOWN
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from perception.readiness.observation_types import (
    InteractionZoneValue,
    Observation,
    ObservationStatus,
)


DEFAULT_BOUNDARY_TOLERANCE: float = 1e-5


def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    """Counter-clockwise turn check for three points."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross_strictly(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Return True if segment p1-p2 strictly crosses segment p3-p4."""
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def validate_polygon_vertices(
    vertices: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Validate normalized polygon vertices for geometric consistency.

    Checks:
        1. At least 3 vertices.
        2. Finite coordinates in normalized range [0.0, 1.0].
        3. No consecutive duplicate vertices.
        4. Non-degenerate area (Shoelace formula >= 1e-6).
        5. No self-intersecting non-adjacent edges.

    Returns:
        List of validated float coordinate tuples.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if len(vertices) < 3:
        raise ValueError(
            f"Interaction zone polygon requires >= 3 vertices, got {len(vertices)}"
        )

    validated: list[tuple[float, float]] = []
    for i, pt in enumerate(vertices):
        if len(pt) != 2:
            raise ValueError(f"Vertex {i} must have exactly 2 coordinates, got {len(pt)}")
        x, y = float(pt[0]), float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Vertex {i} has non-finite coordinates: ({x}, {y})")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Vertex {i} out of normalized range [0.0, 1.0]: ({x}, {y})")
        validated.append((x, y))

    n = len(validated)
    # Check for consecutive duplicate vertices
    for i in range(n):
        j = (i + 1) % n
        if abs(validated[i][0] - validated[j][0]) < 1e-9 and abs(validated[i][1] - validated[j][1]) < 1e-9:
            raise ValueError(f"Polygon has duplicate consecutive vertices at indices {i} and {j}")

    # Area calculation via Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += validated[i][0] * validated[j][1] - validated[j][0] * validated[i][1]
    area = abs(area) * 0.5
    if area < 1e-6:
        raise ValueError(f"Interaction zone polygon has degenerate area ({area:.2e} < 1e-6)")

    # Self-intersection check: non-adjacent edges must not cross
    for i in range(n):
        i_next = (i + 1) % n
        for k in range(i + 1, n):
            k_next = (k + 1) % n
            # Skip adjacent edges sharing vertices
            if i == k or i_next == k or i == k_next or (i == 0 and k_next == 0):
                continue
            if _segments_cross_strictly(validated[i], validated[i_next], validated[k], validated[k_next]):
                raise ValueError(
                    f"Interaction zone polygon is self-intersecting between edges {i} and {k}"
                )

    return validated


class InteractionZoneEvaluator:
    """Evaluates whether a bounding box is entirely inside a polygon zone.

    The polygon is defined using normalized coordinates [0, 1] and
    converted to pixel coordinates at evaluation time.
    """

    def __init__(
        self,
        vertices_normalized: Sequence[tuple[float, float]],
        enabled: bool = True,
        boundary_tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> None:
        """Initialize with normalized polygon vertices.

        Args:
            vertices_normalized: List of (x_norm, y_norm) vertices in [0, 1].
                                 Must have >= 3 vertices to form a polygon.
            enabled:             Whether zone evaluation is active.
            boundary_tolerance:  Tolerance for inclusive boundary tests (pixels).

        Raises:
            ValueError: If polygon has fewer than 3 vertices, non-finite or degenerate coords.
        """
        self.enabled = enabled
        self.boundary_tolerance = boundary_tolerance
        self._vertices_normalized: list[tuple[float, float]] = []

        if not enabled:
            return

        self._vertices_normalized = validate_polygon_vertices(vertices_normalized)

    @property
    def vertices_normalized(self) -> list[tuple[float, float]]:
        """Return the normalized polygon vertices."""
        return list(self._vertices_normalized)

    def to_pixel_coords(
        self,
        image_width: int,
        image_height: int,
    ) -> list[tuple[float, float]]:
        """Convert normalized vertices to pixel coordinates.

        Args:
            image_width:  Frame width in pixels.
            image_height: Frame height in pixels.

        Returns:
            List of (x_px, y_px) polygon vertices in pixel coordinates.
        """
        return [
            (x_n * image_width, y_n * image_height)
            for x_n, y_n in self._vertices_normalized
        ]

    @staticmethod
    def _point_on_segment(
        px: float,
        py: float,
        p1: tuple[float, float],
        p2: tuple[float, float],
        tol: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> bool:
        """Test if point (px, py) lies on line segment p1-p2 within tolerance."""
        x1, y1 = p1
        x2, y2 = p2

        # Bounding box check with tolerance
        if not (min(x1, x2) - tol <= px <= max(x1, x2) + tol):
            return False
        if not (min(y1, y2) - tol <= py <= max(y1, y2) + tol):
            return False

        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < tol * tol:
            dist_sq = (px - x1) ** 2 + (py - y1) ** 2
            return dist_sq <= tol * tol

        # Project point onto segment
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
        t_clamped = max(0.0, min(1.0, t))
        proj_x = x1 + t_clamped * dx
        proj_y = y1 + t_clamped * dy
        dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        return dist_sq <= tol * tol

    @staticmethod
    def _point_in_polygon(
        px: float,
        py: float,
        polygon: list[tuple[float, float]],
        tol: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> bool:
        """Test if a point is inside or on the boundary of a polygon.

        Explicitly checks whether the point lies on any polygon boundary segment
        within a numerical tolerance before executing ray-casting.

        Args:
            px: Point x coordinate.
            py: Point y coordinate.
            polygon: List of (x, y) polygon vertices in pixel space.
            tol: Tolerance in pixels for boundary segment checks.

        Returns:
            True if point is inside or on the polygon boundary.
        """
        n = len(polygon)
        if n < 3:
            return False

        # 1. Explicit boundary check: boundary points are classified as inside
        for i in range(n):
            j = (i + 1) % n
            if InteractionZoneEvaluator._point_on_segment(px, py, polygon[i], polygon[j], tol):
                return True

        # 2. Interior check via ray-casting
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            # Ray casting: count crossings of horizontal ray from (px, py)
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i

        return inside

    def evaluate(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> Observation[InteractionZoneValue]:
        """Evaluate whether the entire bounding box is inside the zone.

        Tests all 4 corners of the bounding box against the polygon.
        All corners must be inside for INSIDE; otherwise OUTSIDE.

        Args:
            bbox_xyxy:    Bounding box (x1, y1, x2, y2) in pixel coordinates.
            image_width:  Frame width in pixels.
            image_height: Frame height in pixels.

        Returns:
            Observation with INSIDE, OUTSIDE, or UNKNOWN value.
        """
        if not self.enabled:
            return Observation(
                value=InteractionZoneValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )

        if len(self._vertices_normalized) < 3:
            return Observation(
                value=InteractionZoneValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )

        # Convert polygon to pixel coordinates
        polygon_px = self.to_pixel_coords(image_width, image_height)

        x1, y1, x2, y2 = bbox_xyxy

        # Verify bbox coordinates are finite
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return Observation(
                value=InteractionZoneValue.UNKNOWN,
                confidence=None,
                status=ObservationStatus.INVALID,
            )

        # Test all 4 corners: TL, TR, BR, BL
        corners = [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x2, y2),  # bottom-right
            (x1, y2),  # bottom-left
        ]

        all_inside = all(
            self._point_in_polygon(cx, cy, polygon_px, self.boundary_tolerance)
            for cx, cy in corners
        )

        if all_inside:
            return Observation(
                value=InteractionZoneValue.INSIDE,
                confidence=None,  # Deterministic geometric check
                status=ObservationStatus.VALID,
            )
        else:
            return Observation(
                value=InteractionZoneValue.OUTSIDE,
                confidence=None,
                status=ObservationStatus.VALID,
            )

