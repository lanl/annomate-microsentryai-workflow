"""
IOController — headless business logic for file I/O.

Rules:
  - No Qt GUI types (no QFileDialog, QMessageBox, QColor).
  - All methods accept plain Python values (paths, strings).
  - Errors are signalled by raising exceptions; callers (Views) handle display.
"""

import os
import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.utils.constants import DEFAULT_CLASS_COLORS

logger = logging.getLogger("AnnoMate.IOController")


class IOController:
    """Headless business logic for all dataset file I/O operations.

    Handles folder scanning, image loading, polygon/overlay export, CSV
    export, and JSON import in both custom and COCO formats. Contains zero
    Qt GUI dependencies — all methods accept and return plain Python values;
    errors are raised as exceptions for callers (Views) to handle.

    Attributes:
        model: Dataset model exposing ``state``, ``load_folder``,
            ``get_image_path``, ``beginResetModel``, and ``endResetModel``.
    """

    def __init__(self, model) -> None:
        """Initialize IOController with a bound dataset model.

        Args:
            model: Dataset model that exposes ``state``, ``load_folder``,
                ``get_image_path``, ``beginResetModel``, and
                ``endResetModel``.
        """
        self.model = model

    def load_folder(self, directory: str) -> None:
        """Scan a directory for images and load them into the model.

        Only files with extensions ``.png``, ``.jpg``, ``.jpeg``, ``.bmp``,
        ``.tif``, and ``.tiff`` (case-insensitive) are included. Results are
        sorted alphabetically before being passed to the model.

        Args:
            directory (str): Absolute path to the folder to scan.
        """
        logger.debug("Scanning directory for images: %s", directory)

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted(
            f for f in os.listdir(directory) if Path(f).suffix.lower() in exts
        )

        logger.debug("Found %d valid images in folder.", len(files))
        self.model.load_folder(directory, files)

    def load_image_for_display(self, row: int) -> Optional[np.ndarray]:
        """Read the image at *row* from disk and return a BGR ndarray.

        Args:
            row (int): Zero-based row index into the model's image list.

        Returns:
            Optional[np.ndarray]: BGR image array, or ``None`` if the file
                cannot be read.
        """
        path = self.model.get_image_path(row)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning(f"Could not read image: {path}")
        return bgr


    def export_csv(self, out_path: str) -> str:
        """Write per-image metadata to a CSV file at *out_path*.

        Each row contains tray name, image filename, accept/reject decision,
        inspector, note, and a comma-separated list of unique annotation class
        names (or ``"good"`` when the image has been reviewed but carries no
        annotations).

        Args:
            out_path (str): Absolute path for the output CSV file.

        Returns:
            str: Human-readable success message containing *out_path*.

        Raises:
            RuntimeError: If no images are currently loaded in the model.
        """
        state = self.model.state
        if not state.image_files:
            raise RuntimeError("No images loaded.")

        tray_name = Path(state.image_dir).name if state.image_dir else ""
        rows = []
        for name in state.image_files:
            anns = state.annotations.get(name, [])
            unique_classes = sorted({a["category_name"] for a in anns})
            reviewed = state.is_reviewed(name)
            rows.append(
                {
                    "tray": tray_name,
                    "image_name": name,
                    "decision": state.review_decisions.get(name, "") if reviewed else "",
                    "inspector": state.inspectors.get(name, "") if reviewed else "",
                    "note": state.notes.get(name, "") if reviewed else "",
                    "classes": (",".join(unique_classes) if unique_classes else "good")
                    if reviewed
                    else "",
                }
            )

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tray", "image_name", "decision", "classes", "inspector", "note"],
            )
            writer.writeheader()
            writer.writerows(rows)

        return f"CSV saved to:\n{out_path}"

    def export_binary_masks(self, out_dir: str) -> str:
        """Write binary mask PNGs to *out_dir* from in-memory annotations.

        For each image that has at least one polygon annotation, renders all
        polygons onto a black canvas as white-filled regions and writes the
        result as a PNG. Images without annotations are skipped — they
        represent defect-free ("good") samples that need no mask.

        Args:
            out_dir (str): Absolute path to the output directory.

        Returns:
            str: Human-readable success message with the count saved.

        Raises:
            RuntimeError: If no images are currently loaded in the model.
        """
        state = self.model.state
        if not state.image_files:
            raise RuntimeError("No images loaded.")

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved = 0
        for name in state.image_files:
            anns = state.annotations.get(name, [])
            if not anns:
                continue

            src = Path(state.image_dir) / name
            if not src.exists():
                logger.warning("Image not found on disk, skipping: %s", src)
                continue

            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Could not read image: %s", src)
                continue

            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            for a in anns:
                pts = np.array(a["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)

            stem = Path(name).stem
            cv2.imwrite(str(out_path / f"{stem}.png"), mask)
            saved += 1

        logger.debug("Exported %d binary mask(s) to: %s", saved, out_dir)
        return f"Saved {saved} binary mask(s) to:\n{out_dir}"

    def export_annotation_classes(self, out_dir: str) -> str:
        """Write annotation class names to annotation_classes.txt in *out_dir*.

        The file is a simple UTF-8 text file with one class name per line,
        preserving the current class registry order.

        Args:
            out_dir (str): Directory where ``annotation_classes.txt`` is saved.

        Returns:
            str: Human-readable success message containing the saved path.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        class_path = out_path / "annotation_classes.txt"

        with open(class_path, "w", encoding="utf-8") as f:
            for name in self.model.state.class_names:
                f.write(f"{name}\n")

        return f"Annotation classes saved to:\n{class_path}"

    def export_train_structure(self, out_dir: str) -> str:
        """Export an MVTec-style anomaly detection training directory.

        Structure:
            {out_dir}/
            ├── train/good/          reviewed images with no annotations
            ├── test/{defect}/       annotated images; multi-class joined with "-"
            └── ground_truth/{defect}/  binary mask PNGs (same subfolder as test)

        Args:
            out_dir: Root directory to write the structure into.

        Returns:
            str: Human-readable summary of counts written.

        Raises:
            RuntimeError: If no images are loaded.
        """
        state = self.model.state
        if not state.image_files:
            raise RuntimeError("No images loaded.")

        root = Path(out_dir)
        counts = {"train_good": 0, "test": 0, "masks": 0, "skipped": 0}

        for name in state.image_files:
            src = Path(state.image_dir) / name
            if not src.exists():
                counts["skipped"] += 1
                logger.warning("Image not found, skipping: %s", src)
                continue

            anns = state.annotations.get(name, [])
            reviewed = state.is_reviewed(name)

            if not anns and reviewed:
                dest = root / "train" / "good"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / name)
                counts["train_good"] += 1

            elif anns:
                folder = "-".join(sorted({a["category_name"] for a in anns}))

                test_dest = root / "test" / folder
                test_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, test_dest / name)
                counts["test"] += 1

                img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for a in anns:
                        pts = np.array(a["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [pts], 255)
                    gt_dest = root / "ground_truth" / folder
                    gt_dest.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(gt_dest / f"{src.stem}.png"), mask)
                    counts["masks"] += 1

        logger.debug("Exported train structure to: %s — %s", out_dir, counts)
        return (
            f"Train structure exported to:\n{out_dir}\n"
            f"  train/good:      {counts['train_good']} images\n"
            f"  test/*:          {counts['test']} images\n"
            f"  ground_truth/*:  {counts['masks']} masks"
            + (f"\n  skipped:         {counts['skipped']} (file not found)" if counts["skipped"] else "")
        )

    # ------------------------------------------------------------------ #
    # Import
    # ------------------------------------------------------------------ #

    def import_annotation_classes(self, path: str) -> str:
        """Merge annotation class names from a simple UTF-8 text file.

        Reads one class name per line. Blank lines and lines beginning with
        ``#`` after trimming whitespace are ignored. Existing class names are
        skipped and keep their current colors; new classes receive the next
        default color in registration order.

        Args:
            path (str): Absolute path to the class-name text file.

        Returns:
            str: Human-readable summary of imported and skipped class counts.
        """
        with open(path, "r", encoding="utf-8") as f:
            candidates = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        state = self.model.state
        imported = 0
        skipped = 0

        self.model.beginResetModel()
        try:
            for name in candidates:
                if name in state.class_names:
                    skipped += 1
                    continue

                idx = len(state.class_names)
                color = DEFAULT_CLASS_COLORS[idx % len(DEFAULT_CLASS_COLORS)]
                state.add_class(name, color)
                imported += 1
        finally:
            self.model.endResetModel()

        return f"Imported {imported} class(es), skipped {skipped} duplicate(s)."

    def import_data_json(self, path: str) -> None:
        """Load annotations from a custom or COCO JSON file into the model.

        Clears existing annotations, inspectors, and notes before importing.
        Dispatches to :meth:`_import_custom_format` when ``images`` is a
        ``dict``, or :meth:`_import_coco_format` when it is a ``list``.
        Triggers a full model reset after loading so all attached views
        refresh.

        Args:
            path (str): Absolute path to the JSON file to import.

        Raises:
            json.JSONDecodeError: If the file is not valid JSON.
            OSError: If the file cannot be opened.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = self.model.state
        state.annotations.clear()
        state.inspectors.clear()
        state.notes.clear()

        images_node = data.get("images")
        if isinstance(images_node, dict):
            self._import_custom_format(state, data, images_node)
        elif isinstance(images_node, list):
            self._import_coco_format(state, data, images_node)

        # Tell all attached views to fully refresh
        self.model.beginResetModel()
        self.model.endResetModel()

    def _import_custom_format(self, state, data: dict, images_node: dict) -> None:
        """Populate *state* from a custom AnnoMate JSON export.

        Rebuilds the class registry from the ``classes`` and ``class_colors``
        keys, then populates annotations, inspector, and note fields for each
        image entry. Missing colors fall back to
        :data:`~core.utils.constants.DEFAULT_CLASS_COLORS`.

        Args:
            state: Dataset state object whose fields are mutated in place.
            data (dict): Top-level parsed JSON object.
            images_node (dict): The ``images`` sub-object from *data*, keyed
                by image filename.
        """
        classes = data.get("classes", [])
        if classes:
            state.class_names = list(classes)
            saved_colors = data.get("class_colors", {})
            state.class_colors = {}
            for i, name in enumerate(state.class_names):
                raw = saved_colors.get(name)
                if isinstance(raw, (list, tuple)) and len(raw) == 3:
                    state.class_colors[name] = (int(raw[0]), int(raw[1]), int(raw[2]))
                else:
                    state.class_colors[name] = DEFAULT_CLASS_COLORS[
                        i % len(DEFAULT_CLASS_COLORS)
                    ]
            state.class_visibility = {name: True for name in state.class_names}

        for name, info in images_node.items():
            state.inspectors[name] = info.get("inspector", "")
            state.notes[name] = info.get("note", "")
            recs = [
                {
                    "category_name": a.get("class", ""),
                    "polygon": a.get("polygon", []),
                    "thickness": a.get("thickness", 2.0),
                }
                for a in info.get("annotations", [])
            ]
            if recs:
                state.annotations[name] = recs

    def _import_coco_format(self, state, data: dict, images_node: list) -> None:
        """Populate *state* from a COCO-format JSON annotation file.

        Registers any new categories into the class registry, then maps COCO
        annotation records to per-filename polygon entries using the image ID
        lookup. Segmentation data is expected in flattened
        ``[x0, y0, x1, y1, …]`` format.

        Args:
            state: Dataset state object whose fields are mutated in place.
            data (dict): Top-level parsed JSON object, expected to contain
                ``categories`` and ``annotations`` keys.
            images_node (list): The ``images`` list from *data*, each entry
                containing ``id`` and ``file_name``.
        """
        cat_map = {}
        if "categories" in data:
            for c in data["categories"]:
                name = c["name"]
                cat_map[c["id"]] = name
                if name not in state.class_names:
                    idx = len(state.class_names)
                    state.class_names.append(name)
                    state.class_colors[name] = DEFAULT_CLASS_COLORS[
                        idx % len(DEFAULT_CLASS_COLORS)
                    ]
                    state.class_visibility[name] = True

        img_id_map = {img["id"]: img["file_name"] for img in images_node}

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_id_map:
                continue
            filename = img_id_map[img_id]
            cat_name = cat_map.get(ann["category_id"], "Unknown")
            seg = ann.get("segmentation", [])
            final_poly = []
            if isinstance(seg, list) and seg:
                pts_list = seg[0] if isinstance(seg[0], list) else seg
                for i in range(0, len(pts_list) - 1, 2):
                    final_poly.append((float(pts_list[i]), float(pts_list[i + 1])))
            if final_poly:
                state.annotations.setdefault(filename, []).append(
                    {"category_name": cat_name, "polygon": final_poly}
                )
