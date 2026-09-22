"""
MVTecADAdapter — adapter for the MVTec AD benchmark's directory layout.

Each category directory under the chosen root is a self-contained object
class (e.g. "bottle", "cable") with its own train/good normal images,
test/good + test/<defect> images, and one binary mask per defective test
image under ground_truth/<defect>/{stem}_mask.png.
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


def _list_images(dir_path: Path) -> list:
    """Return sorted image filenames directly inside *dir_path* (no recursion)."""
    if not dir_path.is_dir():
        return []
    return sorted(
        f.name
        for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


class MVTecADAdapter(PublicDatasetAdapter):
    """Adapter for MVTec AD's train/test/ground_truth category layout."""

    format_name = "MVTec AD"

    def detect_categories(self, root: str) -> list:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        categories = []
        for entry in sorted(root_path.iterdir()):
            if (
                entry.is_dir()
                and (entry / "train" / "good").is_dir()
                and (entry / "test").is_dir()
                and (entry / "ground_truth").is_dir()
            ):
                categories.append(entry.name)
        return categories

    def scan_category(self, root: str, category: str) -> CategoryStats:
        cat_dir = Path(root) / category

        normal_count = len(_list_images(cat_dir / "train" / "good"))
        normal_count += len(_list_images(cat_dir / "test" / "good"))

        defect_counts = {}
        test_dir = cat_dir / "test"
        if test_dir.is_dir():
            for entry in sorted(test_dir.iterdir()):
                if entry.is_dir() and entry.name != "good":
                    defect_counts[entry.name] = len(_list_images(entry))

        return CategoryStats(
            category_name=category,
            total_images=normal_count + sum(defect_counts.values()),
            normal_count=normal_count,
            defect_counts=defect_counts,
        )

    def iter_records(self, root: str, category: str) -> Iterator[DatasetRecord]:
        cat_dir = Path(root) / category

        for fname in _list_images(cat_dir / "train" / "good"):
            yield DatasetRecord(
                rel_path=(Path("train") / "good" / fname).as_posix(),
                is_normal=True,
            )

        for fname in _list_images(cat_dir / "test" / "good"):
            yield DatasetRecord(
                rel_path=(Path("test") / "good" / fname).as_posix(),
                is_normal=True,
            )

        test_dir = cat_dir / "test"
        if not test_dir.is_dir():
            return
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue
            defect_name = defect_dir.name
            mask_dir = cat_dir / "ground_truth" / defect_name
            for fname in _list_images(defect_dir):
                stem = Path(fname).stem
                mask_path = mask_dir / f"{stem}_mask.png"
                regions = []
                if mask_path.is_file():
                    regions.append(
                        RegionSource(
                            kind="mask_file",
                            class_name=defect_name,
                            path=str(mask_path),
                        )
                    )
                yield DatasetRecord(
                    rel_path=(Path("test") / defect_name / fname).as_posix(),
                    is_normal=False,
                    regions=regions,
                )
