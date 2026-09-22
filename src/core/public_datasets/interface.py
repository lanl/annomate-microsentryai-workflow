"""
Abstract adapter interface for public-benchmark dataset import.

Defines the contract every dataset-format adapter must fulfill, mirroring
ai_strategies/interface.py's shape: one ABC, swappable concrete
implementations. build.py and the import dialog only ever talk to this
interface — never to a concrete adapter's own on-disk conventions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np


@dataclass
class RegionSource:
    """One annotated region within an image, in whatever form the source
    dataset provides it. Exactly one payload field is populated, matching
    ``kind``:

    - ``"mask_file"``: ``path`` is an absolute path to a binary raster mask.
    - ``"mask_array"``: ``array`` is an already-decoded mask (for formats
      that encode multiple classes in one file and must split it themselves
      before this point).
    - ``"polygon"``: ``points`` is already a list of (x, y) vertices.
    - ``"bbox"``: ``xyxy`` is ``(x_min, y_min, x_max, y_max)``.

    Attributes:
        kind: Which payload field is populated.
        class_name: Defect/class label for this region.
        path: Absolute path to a mask file (``mask_file`` only).
        array: Decoded mask array (``mask_array`` only).
        points: Polygon vertices (``polygon`` only).
        xyxy: Bounding box (``bbox`` only).
    """

    kind: Literal["mask_file", "mask_array", "polygon", "bbox"]
    class_name: str
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    points: Optional[list] = None
    xyxy: Optional[tuple] = None


@dataclass
class DatasetRecord:
    """One image's normalized ground truth, regardless of source format.

    Attributes:
        rel_path: Path relative to the category root, using forward
            slashes — becomes the image's filename key in DatasetState,
            the same convention ``scan_images()`` already produces.
        is_normal: ``True`` for known-normal images, ``False`` for
            known-defective, ``None`` when the source provides no label
            (e.g. a benchmark's held-out private test split).
        regions: Annotated regions for this image. Empty for normal or
            unlabeled images.
    """

    rel_path: str
    is_normal: Optional[bool]
    regions: list = field(default_factory=list)


@dataclass
class CategoryStats:
    """Summary counts for one category, computed without copying any images.

    Attributes:
        category_name: Name of the scanned category.
        total_images: Total image count across all splits.
        normal_count: Images with no defect (``is_normal=True``).
        defect_counts: Defective image count per class name.
        unlabeled_count: Images with no ground truth available
            (``is_normal=None``).
    """

    category_name: str
    total_images: int
    normal_count: int
    defect_counts: dict
    unlabeled_count: int = 0


class PublicDatasetAdapter(ABC):
    """Abstract adapter for one public benchmark's on-disk layout or manifest.

    Concrete adapters translate their format's own conventions —
    directory structure, CSV manifests, whatever — into DatasetRecord and
    CategoryStats. Those two shapes are the only things the rest of the
    import pipeline (build.py, the import dialog) understands, so adding a
    new format never requires touching anything outside this adapter.

    Attributes:
        format_name: Human-readable name shown in the import dialog's
            format dropdown.
    """

    format_name: str = "Unknown"

    @abstractmethod
    def detect_categories(self, root: str) -> list:
        """Return category names available under *root*.

        Args:
            root: Absolute path to the dataset root chosen by the user.

        Returns:
            list[str]: Category names, or an empty list if none are found.
        """

    def category_root(self, root: str, category: str) -> str:
        """Return the directory DatasetState.image_dir should point at.

        Defaults to ``root/category``, true for formats where a category is
        a real subdirectory (MVTec AD, MVTec AD 2). Adapters whose category
        is a synthetic label rather than an actual subdirectory (e.g. a
        format with no category concept at all) must override this.

        Args:
            root: Absolute path to the dataset root.
            category: One of the names returned by ``detect_categories``.

        Returns:
            str: Absolute path to use as the imported project's image_dir.
        """
        return str(Path(root, category).resolve())

    @abstractmethod
    def scan_category(self, root: str, category: str) -> CategoryStats:
        """Count images and classes for *category* without copying any files.

        Args:
            root: Absolute path to the dataset root.
            category: One of the names returned by ``detect_categories``.

        Returns:
            CategoryStats: Summary counts for the import dialog's stats panel.
        """

    @abstractmethod
    def iter_records(self, root: str, category: str) -> Iterator[DatasetRecord]:
        """Yield one DatasetRecord per image in *category*.

        Args:
            root: Absolute path to the dataset root.
            category: One of the names returned by ``detect_categories``.

        Yields:
            DatasetRecord: Normalized ground truth for one image.
        """
