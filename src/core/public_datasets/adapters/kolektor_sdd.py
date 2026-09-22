"""
KolektorSDDAdapter — adapter for the original (unrestructured) KolektorSDD
directory layout: one subfolder per physical part (e.g. "kos01"), each
containing several "PartN.jpg" images with a paired "PartN_label.bmp"
mask. Unlike MVTec's benchmarks, the raw layout carries no train/test split
and no category concept — it's a single flat pool of one object type, and
whether an image is normal or defective is only knowable by checking
whether its label mask has any nonzero pixels.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from core.public_datasets.interface import (
    CategoryStats,
    DatasetRecord,
    PublicDatasetAdapter,
    RegionSource,
)

_CATEGORY_NAME = "KolektorSDD"
_DEFECT_CLASS = "defect"
_PART_DIR_PREFIX = "kos"


def _is_part_dir(entry: Path) -> bool:
    return entry.is_dir() and entry.name.lower().startswith(_PART_DIR_PREFIX)


def _part_images(part_dir: Path) -> list:
    """Return sorted (image_path, label_path) pairs directly inside *part_dir*."""
    pairs = []
    for jpg in sorted(part_dir.glob("*.jpg")):
        label_path = part_dir / f"{jpg.stem}_label.bmp"
        pairs.append((jpg, label_path))
    return pairs


def _has_defect(label_path: Path) -> bool:
    if not label_path.is_file():
        return False
    return bool(np.array(Image.open(label_path)).any())


class KolektorSDDAdapter(PublicDatasetAdapter):
    """Adapter for the original KolektorSDD layout (kosNN part folders).

    There is exactly one pseudo-"category" — the whole dataset — since the
    raw layout has no per-object-type grouping the way MVTec's benchmarks do.
    """

    format_name = "KolektorSDD"

    def detect_categories(self, root: str) -> list:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        if any(_is_part_dir(entry) for entry in root_path.iterdir()):
            return [_CATEGORY_NAME]
        return []

    def category_root(self, root: str, category: str) -> str:
        # The pseudo-category IS the root — there's no per-category subfolder
        # to descend into, unlike MVTec's benchmarks.
        return str(Path(root).resolve())

    def scan_category(self, root: str, category: str) -> CategoryStats:
        root_path = Path(root)
        normal_count = 0
        defect_count = 0
        for part_dir in sorted(p for p in root_path.iterdir() if _is_part_dir(p)):
            for _img_path, label_path in _part_images(part_dir):
                if _has_defect(label_path):
                    defect_count += 1
                else:
                    normal_count += 1

        return CategoryStats(
            category_name=category,
            total_images=normal_count + defect_count,
            normal_count=normal_count,
            defect_counts={_DEFECT_CLASS: defect_count} if defect_count else {},
        )

    def iter_records(self, root: str, category: str) -> Iterator[DatasetRecord]:
        root_path = Path(root)
        for part_dir in sorted(p for p in root_path.iterdir() if _is_part_dir(p)):
            for img_path, label_path in _part_images(part_dir):
                rel_path = (Path(part_dir.name) / img_path.name).as_posix()
                if _has_defect(label_path):
                    yield DatasetRecord(
                        rel_path=rel_path,
                        is_normal=False,
                        regions=[
                            RegionSource(
                                kind="mask_file",
                                class_name=_DEFECT_CLASS,
                                path=str(label_path),
                            )
                        ],
                    )
                else:
                    yield DatasetRecord(rel_path=rel_path, is_normal=True)
