import pytest
from core.utils.geometry import polygon_area, polygon_bbox


class TestPolygonArea:
    def test_unit_square(self):
        """Verify that polygon_area returns 1.0 for a unit square.

        A square with corners at (0,0), (1,0), (1,1), (0,1) has area 1. Success
        means the result equals 1.0 within floating-point tolerance.
        """
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert polygon_area(pts) == pytest.approx(1.0)

    def test_right_triangle(self):
        """Verify that polygon_area returns the correct area for a right triangle.

        A right triangle with legs of length 2 has area = 0.5 * 2 * 2 = 2.0.
        Success means the result equals 2.0 within floating-point tolerance.
        """
        pts = [(0, 0), (2, 0), (0, 2)]
        assert polygon_area(pts) == pytest.approx(2.0)

    def test_fewer_than_3_points_returns_zero(self):
        """Verify that polygon_area returns 0 for degenerate inputs with fewer than 3 vertices.

        An empty list, a single point, or two points cannot form a polygon, so the
        area should be 0.0. Success means all three degenerate cases return exactly 0.0.
        """
        assert polygon_area([]) == 0.0
        assert polygon_area([(0, 0)]) == 0.0
        assert polygon_area([(0, 0), (1, 0)]) == 0.0

    def test_order_independent(self):
        """Verify that polygon_area returns the same value regardless of vertex winding order.

        Area should be the same for both clockwise and counter-clockwise vertex orderings.
        Success means the area of the original polygon equals the area of the reversed
        polygon within floating-point tolerance.
        """
        cw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        ccw = list(reversed(cw))
        assert polygon_area(cw) == pytest.approx(polygon_area(ccw))


class TestPolygonBbox:
    def test_unit_square(self):
        """Verify that polygon_bbox returns [min_x, min_y, width, height] for a unit square.

        A unit square anchored at the origin should have bbox [0, 0, 1, 1]. Success
        means all four values match within floating-point tolerance.
        """
        # Returns [min_x, min_y, width, height]
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert polygon_bbox(pts) == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_empty_returns_zeros(self):
        """Verify that polygon_bbox returns [0, 0, 0, 0] for an empty polygon.

        With no vertices there is no bounding box, so the function should return a
        safe default of all zeros. Success means the result is [0.0, 0.0, 0.0, 0.0].
        """
        assert polygon_bbox([]) == [0.0, 0.0, 0.0, 0.0]

    def test_non_origin_shape(self):
        """Verify that polygon_bbox correctly computes the bounding box for a non-origin rectangle.

        A rectangle from (2,3) to (5,7) has min_x=2, min_y=3, width=3, height=4.
        Success means all four bbox values match within floating-point tolerance.
        """
        pts = [(2, 3), (5, 3), (5, 7), (2, 7)]
        result = polygon_bbox(pts)
        assert result == pytest.approx([2.0, 3.0, 3.0, 4.0])
