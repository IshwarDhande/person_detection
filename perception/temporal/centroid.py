"""Polygon centroid calculation utility for Milestone M5.

Computes the 2D geometric centroid of a simple polygon using Green's theorem /
the Shoelace formula according to Milestone M5 specification Section 5.2.

Mathematical Formulation:
+-----------------------------------------------------------------------------+
|                          POLYGON CENTROID GEOMETRY                          |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Signed Area:                                                               |
|    A = 0.5 * sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i)                |
|                                                                             |
|  Centroid Coordinates:                                                      |
|    x_z = 1 / (6A) * sum_{i=0}^{n-1} (x_i + x_{i+1}) * (x_i*y_{i+1} - x_{i+1}*y_i)
|    y_z = 1 / (6A) * sum_{i=0}^{n-1} (y_i + y_{i+1}) * (x_i*y_{i+1} - x_{i+1}*y_i)
|                                                                             |
|  Indices wrap around: (x_n, y_n) = (x_0, y_0).                             |
|  Degenerate check: |A| < 1e-6 raises ValueError.                            |
|                                                                             |
+-----------------------------------------------------------------------------+
"""

from __future__ import annotations

import math
from typing import Sequence


def compute_polygon_centroid(
    vertices: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Compute the 2D geometric centroid of a non-self-intersecting polygon.

    Args:
        vertices: Sequence of (x, y) coordinates defining polygon vertices in
                  consistent cyclic order (clockwise or counter-clockwise).

    Returns:
        (x_centroid, y_centroid) as a tuple of floats.

    Raises:
        ValueError: If vertices has fewer than 3 points, contains non-finite
                    coordinates, or defines a degenerate polygon with area < 1e-6.
    """
    if len(vertices) < 3:
        raise ValueError(
            f"Polygon must have at least 3 vertices to compute centroid, got {len(vertices)}"
        )

    pts: list[tuple[float, float]] = []
    for i, pt in enumerate(vertices):
        if len(pt) != 2:
            raise ValueError(f"Vertex {i} must have exactly 2 coordinates, got {len(pt)}")
        x, y = float(pt[0]), float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Vertex {i} contains non-finite coordinates: ({x}, {y})")
        pts.append((x, y))

    n = len(pts)
    signed_area = 0.0
    cx_accum = 0.0
    cy_accum = 0.0

    for i in range(n):
        j = (i + 1) % n
        xi, yi = pts[i]
        xj, yj = pts[j]
        cross = xi * yj - xj * yi
        signed_area += cross
        cx_accum += (xi + xj) * cross
        cy_accum += (yi + yj) * cross

    signed_area *= 0.5

    if abs(signed_area) < 1e-6:
        raise ValueError(
            f"Polygon is degenerate with near-zero signed area ({signed_area:.2e} < 1e-6); "
            "centroid is undefined."
        )

    # 6 * A in denominator
    factor = 1.0 / (6.0 * signed_area)
    cx = cx_accum * factor
    cy = cy_accum * factor

    return float(cx), float(cy)

