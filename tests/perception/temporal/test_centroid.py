"""Unit tests for 2D polygon centroid calculation."""

import math
import unittest

from perception.temporal.centroid import compute_polygon_centroid


class TestPolygonCentroid(unittest.TestCase):
    """Test Green's theorem / Shoelace centroid computation."""

    def test_square_centroid(self) -> None:
        """Verify centroid of a 100x100 square is at (50, 50)."""
        square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        cx, cy = compute_polygon_centroid(square)
        self.assertAlmostEqual(cx, 50.0, places=4)
        self.assertAlmostEqual(cy, 50.0, places=4)

    def test_rectangle_offset(self) -> None:
        """Verify centroid of an offset rectangle."""
        rect = [(200.0, 100.0), (600.0, 100.0), (600.0, 500.0), (200.0, 500.0)]
        cx, cy = compute_polygon_centroid(rect)
        self.assertAlmostEqual(cx, 400.0, places=4)
        self.assertAlmostEqual(cy, 300.0, places=4)

    def test_triangle_centroid(self) -> None:
        """Verify centroid of a triangle is at the average of its vertices."""
        tri = [(0.0, 0.0), (60.0, 0.0), (30.0, 90.0)]
        cx, cy = compute_polygon_centroid(tri)
        self.assertAlmostEqual(cx, 30.0, places=4)
        self.assertAlmostEqual(cy, 30.0, places=4)

    def test_clockwise_and_counterclockwise_give_same_centroid(self) -> None:
        """Verify cyclic vertex orientation does not change centroid location."""
        ccw = [(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)]
        cw = list(reversed(ccw))
        cx_ccw, cy_ccw = compute_polygon_centroid(ccw)
        cx_cw, cy_cw = compute_polygon_centroid(cw)
        self.assertAlmostEqual(cx_ccw, cx_cw, places=5)
        self.assertAlmostEqual(cy_ccw, cy_cw, places=5)

    def test_l_shaped_polygon(self) -> None:
        """Verify centroid of an L-shaped polygon."""
        l_shape = [
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, 20.0),
            (20.0, 20.0),
            (20.0, 60.0),
            (0.0, 60.0),
        ]
        cx, cy = compute_polygon_centroid(l_shape)
        # Total area = 40*20 + 20*40 = 800 + 800 = 1600 (or 20*60 + 20*20 = 1600)
        # Block 1 (0..20, 0..60): area=1200, centroid=(10, 30)
        # Block 2 (20..40, 0..20): area=400, centroid=(30, 10)
        # Combined cx = (1200*10 + 400*30)/1600 = 24000/1600 = 15.0
        # Combined cy = (1200*30 + 400*10)/1600 = 40000/1600 = 25.0
        self.assertAlmostEqual(cx, 15.0, places=3)
        self.assertAlmostEqual(cy, 25.0, places=3)

    def test_fewer_than_three_vertices_raises_error(self) -> None:
        """Verify < 3 vertices raises ValueError."""
        with self.assertRaises(ValueError):
            compute_polygon_centroid([(0.0, 0.0), (1.0, 1.0)])

    def test_collinear_or_zero_area_polygon_raises_error(self) -> None:
        """Verify collinear / degenerate polygon with near-zero area raises ValueError."""
        collinear = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
        with self.assertRaises(ValueError):
            compute_polygon_centroid(collinear)

    def test_non_finite_coordinates_raises_error(self) -> None:
        """Verify non-finite coordinates raise ValueError."""
        with self.assertRaises(ValueError):
            compute_polygon_centroid([(0.0, 0.0), (float("nan"), 10.0), (10.0, 10.0)])


if __name__ == "__main__":
    unittest.main()

