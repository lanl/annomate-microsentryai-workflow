from typing import List, Optional, Tuple
import numpy as np
import cv2


def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Compute the area of a polygon using the Shoelace formula.

    Args:
        points (List[Tuple[float, float]]): Ordered sequence of (x, y)
            vertices defining the polygon boundary.

    Returns:
        float: Polygon area in square pixels. Returns ``0.0`` when fewer
            than three vertices are supplied.
    """
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bbox(points: List[Tuple[float, float]]) -> List[float]:
    """Compute the axis-aligned bounding box of a polygon.

    Args:
        points (List[Tuple[float, float]]): Ordered sequence of (x, y)
            vertices.

    Returns:
        List[float]: Bounding box as ``[x_min, y_min, width, height]``.
            Returns ``[0.0, 0.0, 0.0, 0.0]`` when *points* is empty.
    """
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, min_y = float(min(xs)), float(min(ys))
    return [min_x, min_y, float(max(xs) - min_x), float(max(ys) - min_y)]


def simplify_polygon(
    pts: List[Tuple[float, float]], epsilon: float
) -> List[Tuple[float, float]]:
    """Simplify a polygon using the Douglas-Peucker algorithm via OpenCV.

    Returns the original list object unchanged when simplification cannot
    produce at least 3 points, so callers can use identity check (``is``) to
    detect a no-op.

    Args:
        pts (List[Tuple[float, float]]): Input polygon vertices as (x, y) pairs.
        epsilon (float): Maximum allowable deviation from the original curve.
            Larger values produce coarser simplification.

    Returns:
        List[Tuple[float, float]]: Simplified vertex list, or the original
            *pts* object if simplification yields fewer than 3 points or the
            input itself has fewer than 3 points.
    """
    if len(pts) < 3:
        return pts
    cnt = np.array([[[p[0], p[1]]] for p in pts], dtype=np.float32)
    approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
    if approx is None or len(approx) < 3:
        return pts
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def nearest_point_on_edge(
    points: List[Tuple[float, float]],
    query: Tuple[float, float],
    threshold: Optional[float] = None,
) -> Optional[Tuple[int, Tuple[float, float]]]:
    """Find the closest point lying on a closed polygon's boundary to *query*.

    Args:
        points (List[Tuple[float, float]]): Ordered (x, y) vertices of a
            closed polygon; edges wrap from the last point back to the first.
        query (Tuple[float, float]): The (x, y) point to test against each edge.
        threshold (Optional[float]): If given, ignore edges whose closest
            point is farther than this distance from *query*.

    Returns:
        Optional[Tuple[int, Tuple[float, float]]]: ``(edge_start_idx, point)``
            for the nearest edge, where *point* is the closest point on that
            edge and a new vertex would be inserted at ``edge_start_idx + 1``.
            ``None`` if *points* has fewer than 2 vertices or every edge is
            farther than *threshold*.
    """
    n = len(points)
    if n < 2:
        return None
    qx, qy = query
    best: Optional[Tuple[int, Tuple[float, float]]] = None
    best_dist = threshold if threshold is not None else float("inf")
    for i in range(n):
        ax, ay = points[i]
        bx, by = points[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((qx - ax) * dx + (qy - ay) * dy) / seg_len_sq))
        px, py = ax + t * dx, ay + t * dy
        dist = ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = (i, (px, py))
    return best


def scale_polygon_about_center(
    pts: List[Tuple[float, float]], factor: float
) -> List[Tuple[float, float]]:
    """Scale polygon points about their centroid by *factor*.

    Args:
        pts (List[Tuple[float, float]]): Input polygon vertices as (x, y) pairs.
        factor (float): Scaling factor relative to the centroid. Values
            greater than ``1.0`` expand the polygon; values between ``0.0``
            and ``1.0`` shrink it.

    Returns:
        List[Tuple[float, float]]: New vertex list scaled about the centroid.
            Returns *pts* unchanged when empty.
    """
    if not pts:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [(cx + (p[0] - cx) * factor, cy + (p[1] - cy) * factor) for p in pts]
