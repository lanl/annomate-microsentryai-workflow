"""
Utility Functions for Geometric Operations.

This module provides helper functions for calculating properties of polygons,
such as area and bounding boxes, used throughout the annotation tool.
"""

from typing import List, Tuple
import numpy as np


def polygon_area(points: List[Tuple[float, float]]) -> float:
    """
    Calculates the area of a polygon using the Shoelace formula (Surveyor's formula).

    Args:
        points (List[Tuple[float, float]]): A list of (x, y) coordinates defining the polygon.

    Returns:
        float: The absolute area of the polygon. Returns 0.0 if fewer than 3 points.
    """
    if len(points) < 3:
        return 0.0

    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bbox(points: List[Tuple[float, float]]) -> List[float]:
    """
    Computes the bounding box of a polygon.

    Args:
        points (List[Tuple[float, float]]): A list of (x, y) coordinates.

    Returns:
        List[float]: A list containing [min_x, min_y, width, height].
                     Returns [0.0, 0.0, 0.0, 0.0] if the input list is empty.
    """
    if not points:
        return [0.0, 0.0, 0.0, 0.0]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x = float(min(xs))
    min_y = float(min(ys))
    width = float(max(xs) - min_x)
    height = float(max(ys) - min_y)

    return [min_x, min_y, width, height]


def merge_polygons(polys: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """
    Flattens a list of polygons into a single list of points.

    Note: This does not perform a geometric union (boolean operation);
    it simply concatenates the vertex lists.

    Args:
        polys (List[List[Tuple[float, float]]]): A list of polygons, where each
                                                 polygon is a list of (x, y) tuples.

    Returns:
        List[Tuple[float, float]]: A single list containing all vertices from all polygons.
    """
    merged: List[Tuple[float, float]] = []
    for poly in polys:
        merged.extend(poly)
    return merged