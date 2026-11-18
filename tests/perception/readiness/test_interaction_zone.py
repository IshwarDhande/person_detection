"""Unit tests for Condition 4 — Image-Space Interaction-Zone Membership.

Tests entire-bounding-box containment (all 4 corners), boundary crossing,
outside cases, invalid polygon handling, and normalized coordinate conversion.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.readiness.interaction_zone import InteractionZoneEvaluator
from perception.readiness.observation_types import (
    InteractionZoneValue,
    ObservationStatus,
)


class TestInteractionZone(unittest.TestCase):
    """Test suite for InteractionZoneEvaluator."""

    def setUp(self) -> None:
        # Standard trapezoidal interaction zone in normalized coordinates
        # [0.2, 0.2] to [0.8, 0.2] at top, [0.1, 0.9] to [0.9, 0.9] at bottom
        self.vertices_norm = [
            (0.20, 0.20),
            (0.80, 0.20),
            (0.90, 0.90),
            (0.10, 0.90),
        ]
        self.evaluator = InteractionZoneEvaluator(self.vertices_norm, enabled=True)
        self.img_w = 1000
        self.img_h = 1000

    def test_entire_bounding_box_inside(self) -> None:
        """A box strictly inside the polygon must evaluate to INSIDE with VALID status."""
        # In 1000x1000 image: polygon is (200,200), (800,200), (900,900), (100,900)
        # Place box at (400, 400, 600, 600)
        bbox = (400.0, 400.0, 600.0, 600.0)
        obs = self.evaluator.evaluate(bbox, self.img_w, self.img_h)

        self.assertEqual(obs.value, InteractionZoneValue.INSIDE)
        self.assertEqual(obs.status, ObservationStatus.VALID)
        self.assertIsNone(obs.confidence)

    def test_entire_bounding_box_outside(self) -> None:
        """A box entirely outside the polygon must evaluate to OUTSIDE with VALID status."""
        # Box placed in upper-left corner outside polygon
        bbox = (10.0, 10.0, 100.0, 100.0)
        obs = self.evaluator.evaluate(bbox, self.img_w, self.img_h)

        self.assertEqual(obs.value, InteractionZoneValue.OUTSIDE)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_box_crossing_polygon_boundary_is_outside(self) -> None:
        """Per spec: B ⊆ P required. If any corner is outside, result must be OUTSIDE."""
        # Center of box is inside polygon, but right edge crosses boundary (x2 = 850 at y=300)
        bbox = (700.0, 250.0, 850.0, 400.0)
        obs = self.evaluator.evaluate(bbox, self.img_w, self.img_h)

        self.assertEqual(obs.value, InteractionZoneValue.OUTSIDE)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_box_partially_inside_is_outside(self) -> None:
        """Box with 3 corners inside and 1 outside must be OUTSIDE."""
        # Top-left corner extends past the left slanted boundary
        # At y=300, left boundary is approx x = 200 - (100/700)*100 = ~185.
        # Box at (150, 300, 300, 500) has left corners outside.
        bbox = (150.0, 300.0, 300.0, 500.0)
        obs = self.evaluator.evaluate(bbox, self.img_w, self.img_h)

        self.assertEqual(obs.value, InteractionZoneValue.OUTSIDE)
        self.assertEqual(obs.status, ObservationStatus.VALID)

    def test_disabled_evaluator_returns_unknown_invalid(self) -> None:
        """Disabled evaluator must return UNKNOWN with INVALID status."""
        disabled_eval = InteractionZoneEvaluator(self.vertices_norm, enabled=False)
        obs = disabled_eval.evaluate((400.0, 400.0, 600.0, 600.0), self.img_w, self.img_h)

        self.assertEqual(obs.value, InteractionZoneValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)

    def test_invalid_polygon_initialization(self) -> None:
        """Fewer than 3 vertices or non-finite coords must raise ValueError on init."""
        with self.assertRaises(ValueError):
            InteractionZoneEvaluator([(0.1, 0.1), (0.2, 0.2)])  # only 2 vertices

        with self.assertRaises(ValueError):
            InteractionZoneEvaluator([(0.1, 0.1), (float("nan"), 0.2), (0.3, 0.3)])

        with self.assertRaises(ValueError):
            InteractionZoneEvaluator([(0.1, 0.1), (1.5, 0.2), (0.3, 0.3)])  # out of [0, 1]

    def test_non_finite_bbox_returns_unknown_invalid(self) -> None:
        """Non-finite bbox coordinates must evaluate to UNKNOWN / INVALID."""
        obs = self.evaluator.evaluate((100.0, float("nan"), 200.0, 300.0), self.img_w, self.img_h)
        self.assertEqual(obs.value, InteractionZoneValue.UNKNOWN)
        self.assertEqual(obs.status, ObservationStatus.INVALID)

    def test_normalized_to_pixel_conversion(self) -> None:
        """Coordinates must scale properly across different image resolutions."""
        # At 1280x720: (0.2, 0.2) -> (256, 144)
        px_coords = self.evaluator.to_pixel_coords(1280, 720)
        self.assertAlmostEqual(px_coords[0][0], 256.0)
        self.assertAlmostEqual(px_coords[0][1], 144.0)
        self.assertAlmostEqual(px_coords[1][0], 1024.0)
        self.assertAlmostEqual(px_coords[1][1], 144.0)

    def test_points_on_all_four_polygon_edges_are_inside(self) -> None:
        """Points exactly on each of the 4 polygon segments must evaluate as inside."""
        poly_px = self.evaluator.to_pixel_coords(1000, 1000)
        # Vertices are: (200,200), (800,200), (900,900), (100,900)
        edge_points = [
            (500.0, 200.0),  # Top edge midpoint
            (850.0, 550.0),  # Right slanted edge midpoint
            (500.0, 900.0),  # Bottom edge midpoint
            (150.0, 550.0),  # Left slanted edge midpoint
        ]
        for pt in edge_points:
            self.assertTrue(
                InteractionZoneEvaluator._point_in_polygon(pt[0], pt[1], poly_px),
                f"Point on boundary {pt} failed to evaluate as inside",
            )

    def test_all_polygon_vertices_are_inside(self) -> None:
        """All polygon vertices must evaluate as inside."""
        poly_px = self.evaluator.to_pixel_coords(1000, 1000)
        for vertex in poly_px:
            self.assertTrue(
                InteractionZoneEvaluator._point_in_polygon(vertex[0], vertex[1], poly_px),
                f"Vertex {vertex} failed to evaluate as inside",
            )

    def test_interior_and_exterior_points(self) -> None:
        """Interior points evaluate as inside; exterior points as outside."""
        poly_px = self.evaluator.to_pixel_coords(1000, 1000)
        self.assertTrue(InteractionZoneEvaluator._point_in_polygon(500.0, 500.0, poly_px))
        self.assertFalse(InteractionZoneEvaluator._point_in_polygon(50.0, 50.0, poly_px))
        self.assertFalse(InteractionZoneEvaluator._point_in_polygon(950.0, 500.0, poly_px))

    def test_bounding_boxes_touching_each_polygon_edge(self) -> None:
        """Boxes touching each polygon edge (with all 4 corners inside/on boundary) evaluate to INSIDE."""
        # Rectangular zone [200, 800] x [200, 800] in a 1000x1000 frame
        rect_eval = InteractionZoneEvaluator(
            [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            enabled=True,
        )
        # Box touching top edge (y1 = 200)
        obs_top = rect_eval.evaluate((300.0, 200.0, 700.0, 500.0), 1000, 1000)
        self.assertEqual(obs_top.value, InteractionZoneValue.INSIDE)

        # Box touching right edge (x2 = 800)
        obs_right = rect_eval.evaluate((500.0, 300.0, 800.0, 700.0), 1000, 1000)
        self.assertEqual(obs_right.value, InteractionZoneValue.INSIDE)

        # Box touching bottom edge (y2 = 800)
        obs_bottom = rect_eval.evaluate((300.0, 500.0, 700.0, 800.0), 1000, 1000)
        self.assertEqual(obs_bottom.value, InteractionZoneValue.INSIDE)

        # Box touching left edge (x1 = 200)
        obs_left = rect_eval.evaluate((200.0, 300.0, 500.0, 700.0), 1000, 1000)
        self.assertEqual(obs_left.value, InteractionZoneValue.INSIDE)

        # Box touching corner vertex (x1=200, y1=200)
        obs_corner = rect_eval.evaluate((200.0, 200.0, 500.0, 500.0), 1000, 1000)
        self.assertEqual(obs_corner.value, InteractionZoneValue.INSIDE)

    def test_bounding_boxes_crossing_each_polygon_edge_are_outside(self) -> None:
        """Boxes slightly crossing any boundary segment must evaluate to OUTSIDE."""
        rect_eval = InteractionZoneEvaluator(
            [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            enabled=True,
        )
        # Box crosses top edge
        self.assertEqual(rect_eval.evaluate((300.0, 199.0, 700.0, 500.0), 1000, 1000).value, InteractionZoneValue.OUTSIDE)
        # Box crosses right edge
        self.assertEqual(rect_eval.evaluate((500.0, 300.0, 801.0, 700.0), 1000, 1000).value, InteractionZoneValue.OUTSIDE)
        # Box crosses bottom edge
        self.assertEqual(rect_eval.evaluate((300.0, 500.0, 700.0, 801.0), 1000, 1000).value, InteractionZoneValue.OUTSIDE)
        # Box crosses left edge
        self.assertEqual(rect_eval.evaluate((199.0, 300.0, 500.0, 700.0), 1000, 1000).value, InteractionZoneValue.OUTSIDE)


if __name__ == "__main__":
    unittest.main()

