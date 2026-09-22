"""Recursive image folder scanning — shared by IOController and ProjectController."""

import os
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Ground-truth artifact conventions to skip during a scan, so a folder
# containing both photos and previously-exported (or imported) ground truth
# doesn't have that ground truth re-ingested as if it were new photos.
# "ground_truth" and the "_mask" suffix are this app's own export
# conventions (see export_pixel_train_structure / export_binary_masks);
# "_label" additionally covers third-party benchmarks like KolektorSDD, and
# "masks" covers VisA, whose per-image masks carry no distinguishing suffix
# at all (just "000.png"), only a sibling "Masks/" folder instead.
_EXCLUDED_DIR_NAMES = {"ground_truth", "masks"}
_EXCLUDED_STEM_SUFFIXES = ("_mask", "_label")


def scan_images(directory: str) -> list:
    """Recursively find images under *directory*.

    Returns a sorted list of POSIX-style paths relative to *directory*
    (e.g. ``"nest1/nest2.png"``; a file directly in *directory* has no
    prefix, e.g. ``"root.png"``). Directories starting with ``"."`` or
    named ``"ground_truth"`` are skipped, as are files whose name (before
    the extension) ends with ``"_mask"`` or ``"_label"`` — these are
    ground-truth artifacts, not photos. Symlinks are not followed (matches
    ``os.walk``'s default, avoiding cycles).

    Args:
        directory (str): Absolute path to the folder to scan.

    Returns:
        list: Sorted relative image paths, POSIX-separated.
    """
    root = Path(directory)
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d.lower() not in _EXCLUDED_DIR_NAMES
        ]
        for fname in filenames:
            stem_lower = Path(fname).stem.lower()
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS and not stem_lower.endswith(
                _EXCLUDED_STEM_SUFFIXES
            ):
                rel = Path(dirpath, fname).relative_to(root).as_posix()
                results.append(rel)
    return sorted(results)


def to_native_path(directory: str, rel_path: str) -> str:
    """Join *directory* with a POSIX-style relative path from ``scan_images``.

    ``os.path.join`` does not convert the forward slashes embedded inside
    *rel_path* on Windows, leaving a mixed-separator string (e.g.
    ``"C:\\images\\nest1/dup.jpg"``) that silently fails to string-match an
    equivalent path built through ``pathlib`` — breaking any dict keyed by
    these absolute paths (inference scores/labels/score maps) for nested
    images. ``normpath`` is a no-op on POSIX, where ``/`` is already native.

    Args:
        directory (str): Absolute path to the dataset root.
        rel_path (str): POSIX-style path relative to *directory*.

    Returns:
        str: Absolute path using the current OS's native separator.
    """
    return os.path.normpath(os.path.join(directory, rel_path))
