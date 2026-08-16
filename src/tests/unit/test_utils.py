import pytest
from core.utils.geometry import nearest_point_on_edge, polygon_area, polygon_bbox
from core.utils.image_scan import scan_images


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


class TestNearestPointOnEdge:
    def test_point_above_top_edge_snaps_to_edge_midpoint(self):
        """A query point centered above the top edge of a unit square snaps to that edge's midpoint.

        Square corners are (0,0),(1,0),(1,1),(0,1); the top edge runs from
        vertex 0 to vertex 1. Success means the returned edge_start_idx is 0
        and the projected point is the edge midpoint (0.5, 0.0).
        """
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = nearest_point_on_edge(square, (0.5, -1.0))
        assert result is not None
        edge_idx, point = result
        assert edge_idx == 0
        assert point == pytest.approx((0.5, 0.0))

    def test_wraps_from_last_vertex_to_first(self):
        """The closing edge (last vertex back to the first) is checked too.

        For the unit square, edge 3 runs from (0,1) back to (0,0). A query
        point to the left of that edge should snap onto it.
        """
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = nearest_point_on_edge(square, (-1.0, 0.5))
        assert result is not None
        edge_idx, point = result
        assert edge_idx == 3
        assert point == pytest.approx((0.0, 0.5))

    def test_projection_clamps_to_segment_endpoints(self):
        """A query point beyond a segment's end projects onto the nearest endpoint.

        Querying far past vertex 1 along the top edge's line should clamp to
        vertex 1 itself rather than extrapolating past the segment.
        """
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = nearest_point_on_edge(square, (5.0, 0.0))
        assert result is not None
        _, point = result
        assert point == pytest.approx((1.0, 0.0))

    def test_beyond_threshold_returns_none(self):
        """A query point farther than *threshold* from every edge returns None."""
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert nearest_point_on_edge(square, (0.5, -10.0), threshold=1.0) is None

    def test_fewer_than_2_points_returns_none(self):
        """Fewer than 2 vertices cannot define an edge, so the result is None."""
        assert nearest_point_on_edge([], (0, 0)) is None
        assert nearest_point_on_edge([(0, 0)], (0, 0)) is None


class TestScanImages:
    def test_finds_nested_images_as_relative_posix_paths(self, tmp_path):
        """Recursively finds images at any depth, keyed by path relative to root.

        Builds root.png, nest1/nest1.png, nest1/nest2/nest2.png and confirms all
        three come back as POSIX-separated paths relative to tmp_path.
        """
        (tmp_path / "root.png").touch()
        (tmp_path / "nest1").mkdir()
        (tmp_path / "nest1" / "nest1.png").touch()
        (tmp_path / "nest1" / "nest2").mkdir()
        (tmp_path / "nest1" / "nest2" / "nest2.png").touch()

        result = scan_images(str(tmp_path))

        assert result == ["nest1/nest1.png", "nest1/nest2/nest2.png", "root.png"]

    def test_same_basename_in_different_folders_are_distinct(self, tmp_path):
        """Two images sharing a basename in different subfolders resolve distinctly."""
        (tmp_path / "nest1").mkdir()
        (tmp_path / "nest1" / "dup.png").touch()
        (tmp_path / "nest3").mkdir()
        (tmp_path / "nest3" / "dup.png").touch()

        result = scan_images(str(tmp_path))

        assert result == ["nest1/dup.png", "nest3/dup.png"]

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "report.txt").touch()
        (tmp_path / "photo.jpg").touch()

        assert scan_images(str(tmp_path)) == ["photo.jpg"]

    def test_skips_hidden_directories(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "img.png").touch()
        (tmp_path / "visible.png").touch()

        assert scan_images(str(tmp_path)) == ["visible.png"]

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert scan_images(str(tmp_path)) == []
