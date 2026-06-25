"""Unit tests for anomaly constraint core logic.

All tests are pure Python — no QApplication required.
"""

import pytest

from core.logic.anomaly_constraints import (
    check_area_violations,
    check_distance_violations,
    min_vertex_distance,
    polygon_centroid,
)
from core.states.anomaly_constraint_state import AnomalyConstraintState


# ---------------------------------------------------------------------------
# polygon_centroid
# ---------------------------------------------------------------------------


class TestPolygonCentroid:
    def test_triangle(self):
        """Centroid of a right triangle with vertices at (0,0), (6,0), (0,6).

        The arithmetic mean centroid is (2, 2).
        """
        pts = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)]
        cx, cy = polygon_centroid(pts)
        assert cx == pytest.approx(2.0)
        assert cy == pytest.approx(2.0)

    def test_square(self):
        """Centroid of a unit square is (0.5, 0.5)."""
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        cx, cy = polygon_centroid(pts)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)

    def test_empty_returns_origin(self):
        """Empty polygon returns (0, 0)."""
        assert polygon_centroid([]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# min_vertex_distance
# ---------------------------------------------------------------------------


class TestMinVertexDistance:
    def test_adjacent_squares(self):
        """Two unit squares placed 1 unit apart.

        Square A: (0,0)-(1,1), Square B: (2,0)-(3,1). Nearest vertices are
        (1,0)↔(2,0) and (1,1)↔(2,1), distance = 1.0.
        """
        a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        b = [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]
        assert min_vertex_distance(a, b) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        """Any empty polygon produces 0.0."""
        assert min_vertex_distance([], [(0.0, 0.0)]) == 0.0
        assert min_vertex_distance([(0.0, 0.0)], []) == 0.0


# ---------------------------------------------------------------------------
# check_area_violations
# ---------------------------------------------------------------------------


def _square_ann(size: float, offset=(0.0, 0.0)) -> dict:
    x0, y0 = offset
    return {
        "polygon": [
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
        ]
    }


class TestCheckAreaViolations:
    def test_no_violations(self):
        """All polygons are below the threshold — returns an empty set.

        Two 10×10 px squares (area=100 px²) with threshold=200. Neither violates.
        """
        anns = [_square_ann(10.0), _square_ann(10.0, offset=(20.0, 0.0))]
        result = check_area_violations(anns, threshold_sq_world=200.0, scale=None)
        assert result == set()

    def test_some_violations(self):
        """Only the large polygon exceeds the threshold.

        One 5×5 px square (area=25) and one 20×20 px square (area=400) with
        threshold=100. Only index 1 violates.
        """
        anns = [_square_ann(5.0), _square_ann(20.0, offset=(30.0, 0.0))]
        result = check_area_violations(anns, threshold_sq_world=100.0, scale=None)
        assert result == {1}

    def test_all_violations(self):
        """All polygons violate when threshold is very small."""
        anns = [_square_ann(10.0), _square_ann(10.0)]
        result = check_area_violations(anns, threshold_sq_world=1.0, scale=None)
        assert result == {0, 1}

    def test_zero_threshold_returns_empty(self):
        """Threshold of 0.0 disables the check — always empty."""
        anns = [_square_ann(100.0)]
        assert check_area_violations(anns, threshold_sq_world=0.0, scale=None) == set()

    def test_with_scale(self):
        """Scale factor converts px² to world-unit² before comparing.

        A 10×10 px square has area=100 px². With scale=0.1 (mm/px), area in
        world units = 100 * 0.01 = 1.0 mm². Threshold=0.5 mm² → violation.
        Threshold=2.0 mm² → no violation.
        """
        ann = [_square_ann(10.0)]
        scale = 0.1
        assert check_area_violations(ann, threshold_sq_world=0.5, scale=scale) == {0}
        assert check_area_violations(ann, threshold_sq_world=2.0, scale=scale) == set()

    def test_empty_annotations(self):
        """Empty annotation list returns empty set without error."""
        assert check_area_violations([], threshold_sq_world=10.0, scale=None) == set()


# ---------------------------------------------------------------------------
# check_distance_violations
# ---------------------------------------------------------------------------


def _centroid_ann(cx: float, cy: float, size: float = 1.0) -> dict:
    """Tiny square annotation centered at (cx, cy) for predictable centroid."""
    h = size / 2.0
    return {
        "polygon": [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy + h),
            (cx - h, cy + h),
        ]
    }


class TestCheckDistanceViolations:
    def test_no_violations_centroid(self):
        """Two annotations far apart do not violate the distance threshold.

        Centroids 100 px apart with threshold=50 → no violation.
        """
        anns = [_centroid_ann(0.0, 0.0), _centroid_ann(100.0, 0.0)]
        cache: dict = {}
        result = check_distance_violations(anns, 50.0, None, "centroid", cache)
        assert result == set()

    def test_pair_too_close_centroid(self):
        """Two annotations 10 px apart violate a 50 px threshold (centroid).

        Success: frozenset {0, 1} is in the returned set.
        """
        anns = [_centroid_ann(0.0, 0.0), _centroid_ann(10.0, 0.0)]
        cache: dict = {}
        result = check_distance_violations(anns, 50.0, None, "centroid", cache)
        assert frozenset({0, 1}) in result

    def test_pair_too_close_edge(self):
        """Edge method: two squares 1 px apart violate a 5 px threshold."""
        a = _centroid_ann(0.0, 0.0, size=2.0)
        b = _centroid_ann(3.0, 0.0, size=2.0)
        cache: dict = {}
        result = check_distance_violations([a, b], 5.0, None, "edge", cache)
        assert frozenset({0, 1}) in result

    def test_zero_threshold_returns_empty(self):
        """Threshold of 0.0 always returns empty — check disabled."""
        anns = [_centroid_ann(0.0, 0.0), _centroid_ann(1.0, 0.0)]
        cache: dict = {}
        assert check_distance_violations(anns, 0.0, None, "centroid", cache) == set()

    def test_single_annotation_returns_empty(self):
        """Cannot have a distance violation with fewer than two annotations."""
        cache: dict = {}
        result = check_distance_violations(
            [_centroid_ann(0.0, 0.0)], 100.0, None, "centroid", cache
        )
        assert result == set()

    def test_cache_populated_after_call(self):
        """Distance cache is filled after check_distance_violations returns.

        Two annotations produce exactly one cache entry for key (0, 1).
        """
        anns = [_centroid_ann(0.0, 0.0), _centroid_ann(10.0, 0.0)]
        cache: dict = {}
        check_distance_violations(anns, 50.0, None, "centroid", cache)
        assert (0, 1) in cache
        assert cache[(0, 1)] == pytest.approx(10.0)

    def test_cache_hit_not_recomputed(self):
        """Manually inserting a large cache value prevents the real distance
        from being recomputed.

        Insert (0, 1) → 999 into cache before calling. The function uses the
        cached value (999) for comparison rather than the actual ~10 px. With
        threshold=50, 999 >= 50, so no violation is reported even though the
        true distance is 10.
        """
        anns = [_centroid_ann(0.0, 0.0), _centroid_ann(10.0, 0.0)]
        cache = {(0, 1): 999.0}
        result = check_distance_violations(anns, 50.0, None, "centroid", cache)
        assert frozenset({0, 1}) not in result

    def test_with_scale(self):
        """Scale converts world threshold to px before comparison.

        Centroids 10 px apart. scale=0.1 mm/px. threshold=0.5 mm →
        threshold_px = 0.5/0.1 = 5 px. 10 px > 5 px → no violation.
        threshold=2.0 mm → threshold_px = 20 px. 10 px < 20 px → violation.
        """
        anns = [_centroid_ann(0.0, 0.0), _centroid_ann(10.0, 0.0)]
        cache: dict = {}
        assert check_distance_violations(anns, 0.5, 0.1, "centroid", cache) == set()
        cache.clear()
        result = check_distance_violations(anns, 2.0, 0.1, "centroid", cache)
        assert frozenset({0, 1}) in result


# ---------------------------------------------------------------------------
# AnomalyConstraintState serialisation
# ---------------------------------------------------------------------------


class TestAnomalyConstraintStateSerialization:
    def test_round_trip_defaults(self):
        """to_dict / from_dict with default values produces an identical state."""
        s = AnomalyConstraintState()
        loaded = AnomalyConstraintState.from_dict(s.to_dict())
        assert loaded.enabled == s.enabled
        assert loaded.area_check_enabled == s.area_check_enabled
        assert loaded.area_threshold == s.area_threshold
        assert loaded.area_color == s.area_color
        assert loaded.distance_check_enabled == s.distance_check_enabled
        assert loaded.distance_threshold == s.distance_threshold
        assert loaded.distance_method == s.distance_method
        assert loaded.distance_color == s.distance_color

    def test_round_trip_custom(self):
        """Custom values survive serialisation."""
        s = AnomalyConstraintState(
            enabled=True,
            area_threshold=250.5,
            area_color=(100, 200, 50),
            distance_threshold=30.0,
            distance_method="edge",
            distance_color=(10, 20, 200),
        )
        loaded = AnomalyConstraintState.from_dict(s.to_dict())
        assert loaded.enabled is True
        assert loaded.area_threshold == pytest.approx(250.5)
        assert loaded.area_color == (100, 200, 50)
        assert loaded.distance_threshold == pytest.approx(30.0)
        assert loaded.distance_method == "edge"
        assert loaded.distance_color == (10, 20, 200)

    def test_from_dict_missing_keys(self):
        """Missing keys fall back to defaults without raising."""
        loaded = AnomalyConstraintState.from_dict({})
        assert loaded.enabled is False
        assert loaded.area_threshold == 0.0
        assert loaded.distance_method == "centroid"
        assert loaded.area_color == (255, 165, 0)
        assert loaded.distance_color == (220, 50, 50)
