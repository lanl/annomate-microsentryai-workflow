"""
VisAAdapter — adapter for the VisA benchmark's CSV-manifest layout.

Unlike MVTec's directory-convention formats, VisA's ground truth lives in a
per-category `image_anno.csv` manifest (columns: image, label, mask), not in
folder naming. Paths inside the manifest are relative to the dataset root
and self-prefixed with the category name (e.g.
"candle/Data/Images/Anomaly/000.JPG"), not relative to the category root.

Reading the manifest instead of walking the directory tree also sidesteps
a real hazard confirmed in this dataset's actual downloaded layout: it
carries thousands of stray "*.JPG:Zone.Identifier" sibling files (a
Windows/WSL download artifact) alongside every real image, which a naive
directory scan would need to filter but the manifest never mentions.

Each category's `label` column is a small, fixed, controlled vocabulary of
defect tags (4-8 per category), not free text — and it's multi-label: a
comma-separated value (e.g. "bubble,discolor") means the image genuinely
exhibits more than one defect type at once, which happens for anywhere from
7% to 70% of a category's defective images. VisA ships one combined mask
per image, not one mask per defect type, and mask contour counts were
confirmed (empirically, against this data) not to correlate with tag counts
— so per-region attribution across multiple tags isn't recoverable. Only
the first-listed tag becomes the pixel-level annotation's class; the full
tag list is preserved as image-level class tags instead, so no tag is
silently dropped even though pixel precision is only ever claimed for one
of them.
"""

import csv
from pathlib import Path
from typing import Iterator

from core.public_datasets.interface import (
    CategoryStats,
    DatasetRecord,
    PublicDatasetAdapter,
    RegionSource,
)

_FALLBACK_CLASS = "anomaly"
_MANIFEST_NAME = "image_anno.csv"

# Shortened class names for VisA's own (often verbose) tag vocabulary,
# keyed by category then raw tag. Only overrides where the raw tag carries
# redundant context (e.g. "on candle", "of packaging") are listed — already
# short tags (bubble, scratch, melt, ...) are left as-is. Where a category
# distinguishes "similar/same colour" from "different colour" as two
# genuinely separate defect types, the qualifier is kept rather than
# collapsing both to "color spot".
_CLASS_NAME_OVERRIDES = {
    "candle": {
        "damaged corner of packaging": "damaged corner",
        "different colour spot": "color spot",
        "foreign particals on candle": "foreign particle",
        "chunk of wax missing": "wax missing",
        "wax melded out of the candle": "wax melded",
        "weird candle wick": "bad wick",
        "extra wax in candle": "extra wax",
    },
    "cashew": {
        "corner or edge breakage": "edge breakage",
        "small scratches": "scratches",
        "same colour spot": "same-color spot",
        "different colour spot": "diff-color spot",
    },
    "chewinggum": {
        "chunk of gum missing": "gum missing",
        "small cracks": "cracks",
        "similar colour spot": "color spot",
    },
    "fryum": {
        "fryum stuck together": "stuck together",
        "corner or edge breakage": "edge breakage",
        "small scratches": "scratches",
        "similar colour spot": "similar-color spot",
        "different colour spot": "diff-color spot",
    },
    "macaroni1": {
        "chip around edge and corner": "edge chip",
        "small scratches": "scratches",
        "small cracks": "cracks",
        "similar colour spot": "similar-color spot",
        "different colour spot": "diff-color spot",
    },
    "macaroni2": {
        "small chip around edge": "edge chip",
        "breakage down the middle": "middle breakage",
        "small cracks": "cracks",
        "color spot similar to the object": "similar-color spot",
        "different color spot": "diff-color spot",
    },
    "pipe_fryum": {
        "corner and edge breakage": "edge breakage",
        "small scratches": "scratches",
        "small cracks": "cracks",
        "similar colour spot": "similar-color spot",
        "different colour spot": "diff-color spot",
    },
}


def _read_manifest(cat_dir: Path) -> list:
    manifest = cat_dir / _MANIFEST_NAME
    if not manifest.is_file():
        return []
    with open(manifest, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _short_tags(category: str, label: str) -> list:
    """Split a manifest label into its individual tags, shortened where known."""
    overrides = _CLASS_NAME_OVERRIDES.get(category, {})
    tags = [t.strip() for t in label.split(",") if t.strip()]
    return [overrides.get(tag, tag) for tag in tags]


class VisAAdapter(PublicDatasetAdapter):
    """Adapter for VisA's per-category image_anno.csv manifest."""

    format_name = "VisA"

    def detect_categories(self, root: str) -> list:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        categories = []
        for entry in sorted(root_path.iterdir()):
            if entry.is_dir() and (entry / _MANIFEST_NAME).is_file():
                categories.append(entry.name)
        return categories

    def scan_category(self, root: str, category: str) -> CategoryStats:
        rows = _read_manifest(Path(root) / category)
        normal_count = 0
        defect_counts = {}
        for row in rows:
            label = row.get("label", "").strip()
            if label.lower() == "normal":
                normal_count += 1
                continue
            tags = _short_tags(category, label)
            primary = tags[0] if tags else _FALLBACK_CLASS
            defect_counts[primary] = defect_counts.get(primary, 0) + 1

        return CategoryStats(
            category_name=category,
            total_images=normal_count + sum(defect_counts.values()),
            normal_count=normal_count,
            defect_counts=defect_counts,
        )

    def iter_records(self, root: str, category: str) -> Iterator[DatasetRecord]:
        cat_dir = Path(root) / category
        for row in _read_manifest(cat_dir):
            # Manifest paths are relative to root and self-prefixed with the
            # category name; category_root() (the default) points image_dir
            # at root/category, so strip that prefix to get a path relative
            # to it.
            rel_path = Path(row["image"].strip()).relative_to(category).as_posix()
            label = row.get("label", "").strip()
            is_normal = label.lower() == "normal"

            regions = []
            image_classes = []
            if not is_normal:
                tags = _short_tags(category, label)
                primary = tags[0] if tags else _FALLBACK_CLASS
                image_classes = tags
                mask_rel = row.get("mask", "").strip()
                if mask_rel:
                    mask_path = Path(root) / Path(mask_rel)
                    regions.append(
                        RegionSource(
                            kind="mask_file",
                            class_name=primary,
                            path=str(mask_path),
                        )
                    )

            yield DatasetRecord(
                rel_path=rel_path,
                is_normal=is_normal,
                regions=regions,
                # Kept as the manifest's original, unshortened text — full
                # human-readable context belongs in a free-text note, unlike
                # the class name which needs to stay short and consistent.
                note=label if not is_normal else "",
                image_classes=image_classes,
            )
