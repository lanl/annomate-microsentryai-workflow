"""
MVTecAD2Adapter — adapter for the MVTec AD 2 benchmark's directory layout.

Each category directory under the chosen root has train/good and
validation/good normal images, and a test_public split with per-image
ground truth (good/bad/ground_truth/bad/{stem}_mask.png — a single "bad"
class, not per-defect-type subfolders like MVTec AD 1). Two private test
splits (test_private, test_private_mixed) ship with no released labels or
masks at all; their images import as unlabeled review candidates rather
than being skipped.
"""

from pathlib import Path
from typing import Iterator

from core.public_datasets.interface import (
    CategoryStats,
    DatasetRecord,
    PublicDatasetAdapter,
    RegionSource,
)
from core.utils.image_scan import IMAGE_EXTENSIONS

_UNLABELED_SPLITS = ("test_private", "test_private_mixed")


def _list_images(dir_path: Path) -> list:
    """Return sorted image filenames directly inside *dir_path* (no recursion)."""
    if not dir_path.is_dir():
        return []
    return sorted(
        f.name
        for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


class MVTecAD2Adapter(PublicDatasetAdapter):
    """Adapter for MVTec AD 2's train/validation/test_public/test_private* layout."""

    format_name = "MVTec AD 2"

    def detect_categories(self, root: str) -> list:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        categories = []
        for entry in sorted(root_path.iterdir()):
            if (
                entry.is_dir()
                and (entry / "train" / "good").is_dir()
                and (entry / "test_public" / "good").is_dir()
                and (entry / "test_public" / "bad").is_dir()
                and (entry / "test_public" / "ground_truth" / "bad").is_dir()
            ):
                categories.append(entry.name)
        return categories

    def scan_category(self, root: str, category: str) -> CategoryStats:
        cat_dir = Path(root) / category

        normal_count = len(_list_images(cat_dir / "train" / "good"))
        normal_count += len(_list_images(cat_dir / "validation" / "good"))
        normal_count += len(_list_images(cat_dir / "test_public" / "good"))

        bad_count = len(_list_images(cat_dir / "test_public" / "bad"))
        defect_counts = {"bad": bad_count} if bad_count else {}

        unlabeled_count = sum(
            len(_list_images(cat_dir / split)) for split in _UNLABELED_SPLITS
        )

        return CategoryStats(
            category_name=category,
            total_images=normal_count + bad_count + unlabeled_count,
            normal_count=normal_count,
            defect_counts=defect_counts,
            unlabeled_count=unlabeled_count,
        )

    def iter_records(self, root: str, category: str) -> Iterator[DatasetRecord]:
        cat_dir = Path(root) / category

        for split in ("train", "validation"):
            for fname in _list_images(cat_dir / split / "good"):
                yield DatasetRecord(
                    rel_path=(Path(split) / "good" / fname).as_posix(),
                    is_normal=True,
                )

        for fname in _list_images(cat_dir / "test_public" / "good"):
            yield DatasetRecord(
                rel_path=(Path("test_public") / "good" / fname).as_posix(),
                is_normal=True,
            )

        mask_dir = cat_dir / "test_public" / "ground_truth" / "bad"
        for fname in _list_images(cat_dir / "test_public" / "bad"):
            stem = Path(fname).stem
            mask_path = mask_dir / f"{stem}_mask.png"
            regions = []
            if mask_path.is_file():
                regions.append(
                    RegionSource(
                        kind="mask_file", class_name="bad", path=str(mask_path)
                    )
                )
            yield DatasetRecord(
                rel_path=(Path("test_public") / "bad" / fname).as_posix(),
                is_normal=False,
                regions=regions,
            )

        for split in _UNLABELED_SPLITS:
            for fname in _list_images(cat_dir / split):
                yield DatasetRecord(
                    rel_path=(Path(split) / fname).as_posix(),
                    is_normal=None,
                )
