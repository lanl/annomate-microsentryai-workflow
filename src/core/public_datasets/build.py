"""
build_dataset_from_category — the one place that turns a PublicDatasetAdapter's
normalized records into the plain dict shape DatasetState already expects.
Adapters never construct annotations themselves; this function is shared by
every registered format so mask-to-polygon conversion and class-registry
building are written once.
"""

import logging

import cv2
import numpy as np

from core.public_datasets.interface import PublicDatasetAdapter, RegionSource
from core.utils.constants import DEFAULT_CLASS_COLORS
from core.utils.geometry import simplify_polygon

logger = logging.getLogger("AnnoMate.PublicDatasetImport")

# Matches the thresholds ai_strategies/sam_strategy.py already uses for its
# own mask -> polygon conversion, for consistency across the app.
_MIN_CONTOUR_AREA = 10.0
_SIMPLIFY_EPSILON = 2.0


def _mask_to_polygons(binary: np.ndarray) -> list:
    """Convert a binary (0/1) mask into one polygon per foreground region."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < _MIN_CONTOUR_AREA:
            continue
        points = [(float(p[0][0]), float(p[0][1])) for p in contour]
        if len(points) < 3:
            continue
        polygons.append(simplify_polygon(points, _SIMPLIFY_EPSILON))
    return polygons


def _region_to_polygons(region: RegionSource) -> list:
    """Convert one RegionSource into zero or more (x, y) polygons."""
    if region.kind == "mask_file":
        mask = cv2.imread(region.path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("Could not read mask file: %s", region.path)
            return []
        return _mask_to_polygons((mask > 0).astype(np.uint8))
    if region.kind == "mask_array":
        return _mask_to_polygons((region.array > 0).astype(np.uint8))
    if region.kind == "polygon":
        return [region.points]
    if region.kind == "bbox":
        x0, y0, x1, y1 = region.xyxy
        return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    raise ValueError(f"Unknown region kind: {region.kind!r}")


def build_dataset_from_category(
    adapter: PublicDatasetAdapter, root: str, category: str
) -> dict:
    """Build a DatasetState-ready dict from one category of a public dataset.

    Args:
        adapter: Adapter for the source format.
        root: Absolute path to the dataset root passed to the adapter.
        category: Category name, as returned by ``adapter.detect_categories()``.

    Returns:
        dict: Keys ``image_dir``, ``image_files``, ``annotations``,
        ``class_names``, ``class_colors``, ``review_decisions``, ``notes``,
        ``image_classes``, ``annotation_mode`` — the same shape
        ``ProjectController.new_project_from_import()`` applies directly to
        DatasetState. No ``.annoproj`` file is written here.
    """
    image_dir = adapter.category_root(root, category)
    image_files = []
    annotations = {}
    review_decisions = {}
    notes = {}
    image_classes = {}
    class_names = []
    class_colors = {}

    def _register_class(name: str) -> None:
        if name not in class_names:
            class_colors[name] = DEFAULT_CLASS_COLORS[
                len(class_names) % len(DEFAULT_CLASS_COLORS)
            ]
            class_names.append(name)

    for record in adapter.iter_records(root, category):
        image_files.append(record.rel_path)

        if record.is_normal is True:
            review_decisions[record.rel_path] = "accept"
        elif record.is_normal is False:
            review_decisions[record.rel_path] = "reject"
        # is_normal is None (no ground truth available) -> left undecided

        if record.note:
            notes[record.rel_path] = record.note

        if record.image_classes:
            tags = list(dict.fromkeys(name.lower() for name in record.image_classes))
            for tag in tags:
                _register_class(tag)
            image_classes[record.rel_path] = tags

        recs = []
        for region in record.regions:
            class_name = region.class_name.lower()
            _register_class(class_name)
            for polygon in _region_to_polygons(region):
                recs.append(
                    {
                        "category_name": class_name,
                        "polygon": polygon,
                        "thickness": 2.0,
                        "visible": True,
                    }
                )
        if recs:
            annotations[record.rel_path] = recs

    return {
        "image_dir": image_dir,
        "image_files": sorted(image_files),
        "annotations": annotations,
        "class_names": class_names,
        "class_colors": class_colors,
        "review_decisions": review_decisions,
        "notes": notes,
        "image_classes": image_classes,
        "annotation_mode": "pixel",
    }
