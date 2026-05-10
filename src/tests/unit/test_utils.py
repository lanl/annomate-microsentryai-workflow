import pytest
from core.utils.geometry import polygon_area, polygon_bbox


class TestPolygonArea:
    def test_unit_square(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert polygon_area(pts) == pytest.approx(1.0)

    def test_right_triangle(self):
        pts = [(0, 0), (2, 0), (0, 2)]
        assert polygon_area(pts) == pytest.approx(2.0)

    def test_fewer_than_3_points_returns_zero(self):
        assert polygon_area([]) == 0.0
        assert polygon_area([(0, 0)]) == 0.0
        assert polygon_area([(0, 0), (1, 0)]) == 0.0

    def test_order_independent(self):
        cw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        ccw = list(reversed(cw))
        assert polygon_area(cw) == pytest.approx(polygon_area(ccw))


class TestPolygonBbox:
    def test_unit_square(self):
        # Returns [min_x, min_y, width, height]
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert polygon_bbox(pts) == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_empty_returns_zeros(self):
        assert polygon_bbox([]) == [0.0, 0.0, 0.0, 0.0]

    def test_non_origin_shape(self):
        pts = [(2, 3), (5, 3), (5, 7), (2, 7)]
        result = polygon_bbox(pts)
        assert result == pytest.approx([2.0, 3.0, 3.0, 4.0])
