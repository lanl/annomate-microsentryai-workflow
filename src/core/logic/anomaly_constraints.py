from __future__ import annotations

import math
from typing import List, Tuple

from core.utils.geometry import polygon_area


def polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return the arithmetic-mean centroid of a polygon's vertices."""
    if not points:
        return (0.0, 0.0)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return (cx, cy)


def min_vertex_distance(
    pts_a: List[Tuple[float, float]],
    pts_b: List[Tuple[float, float]],
) -> float:
    """Return the minimum vertex-to-vertex distance between two polygons.

    Used as a practical approximation of minimum edge-to-edge distance.
    Returns 0.0 if either polygon is empty.
    """
    if not pts_a or not pts_b:
        return 0.0
    best = math.inf
    for ax, ay in pts_a:
        for bx, by in pts_b:
            dx, dy = ax - bx, ay - by
            d = math.sqrt(dx * dx + dy * dy)
            if d < best:
                best = d
    return best


def check_area_violations(
    annotations: list[dict],
    threshold_sq_world: float,
    scale: float | None,
) -> set[int]:
    """Return the indices of annotations whose area exceeds *threshold_sq_world*.

    Args:
        annotations: List of annotation dicts (each has a ``"polygon"`` key).
        threshold_sq_world: Area threshold in world-unit² (px² if *scale* is None).
            A value of 0.0 returns an empty set immediately.
        scale: World units per original-image pixel, or None for pixel mode.

    Returns:
        Set of annotation indices (0-based) that exceed the threshold.
    """
    if threshold_sq_world <= 0.0:
        return set()

    violations: set[int] = set()
    scale_sq = (scale * scale) if scale is not None else 1.0
    for i, ann in enumerate(annotations):
        pts = ann.get("polygon", [])
        area_px = polygon_area(pts)
        area_world = area_px * scale_sq
        if area_world > threshold_sq_world:
            violations.add(i)
    return violations


def check_distance_violations(
    annotations: list[dict],
    threshold_world: float,
    scale: float | None,
    method: str,
    distance_cache: dict[tuple[int, int], float],
) -> set[frozenset[int]]:
    """Return pairs of annotations whose distance is below *threshold_world*.

    Distances are computed lazily and cached in *distance_cache* (mutated in-place).
    Cache keys are ``(min_index, max_index)`` tuples; values are distances in pixels.

    Args:
        annotations: List of annotation dicts.
        threshold_world: Minimum acceptable distance in world-units (px if *scale* is None).
            A value of 0.0 returns an empty set immediately.
        scale: World units per original-image pixel, or None for pixel mode.
        method: ``"centroid"`` for centroid-to-centroid, ``"edge"`` for min vertex distance.
        distance_cache: Mutable dict; updated in-place with newly computed pairs.

    Returns:
        Set of frozensets ``{i, j}`` for pairs that are too close.
    """
    if threshold_world <= 0.0 or len(annotations) < 2:
        return set()

    scale_factor = scale if scale is not None else 1.0
    threshold_px = threshold_world / scale_factor

    violations: set[frozenset[int]] = set()
    n = len(annotations)

    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            if key not in distance_cache:
                pts_i = annotations[i].get("polygon", [])
                pts_j = annotations[j].get("polygon", [])
                if method == "edge":
                    dist_px = min_vertex_distance(pts_i, pts_j)
                else:
                    ci = polygon_centroid(pts_i)
                    cj = polygon_centroid(pts_j)
                    dx, dy = ci[0] - cj[0], ci[1] - cj[1]
                    dist_px = math.sqrt(dx * dx + dy * dy)
                distance_cache[key] = dist_px

            if distance_cache[key] < threshold_px:
                violations.add(frozenset({i, j}))

    return violations
