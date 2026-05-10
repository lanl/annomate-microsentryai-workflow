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
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

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

    def export_polygons_and_data(self, out_dir: str) -> str:
        """Write overlay images and a JSON data file to *out_dir*.

        For each annotated image, composites filled polygon overlays onto the
        source image using Pillow and saves the result as a JPEG. A single
        JSON file containing all annotations, metadata, and class definitions
        is written alongside the overlay images. Colors are stored as plain
        ``[r, g, b]`` lists for JSON serialisability.

        Args:
            out_dir (str): Absolute path to the output directory.

        Returns:
            str: Human-readable success message including the count of overlay
                images saved and the path to the JSON data file.

        Raises:
            RuntimeError: If no images are currently loaded in the model.
        """
        state = self.model.state
        if not state.image_files:
            logger.warning("Attempted to export, but no images are loaded.")
            raise RuntimeError("No images loaded.")

        logger.debug("Starting polygon export to: %s", out_dir)

        out_path = Path(out_dir)
        tray_name = Path(state.image_dir).name if state.image_dir else "tray"
        timestamp = datetime.now().strftime("%m-%d-%y-%H-%M-%S")

        payload = {
            "meta": {"tray": tray_name, "exported_at": timestamp},
            "classes": list(state.class_names),
            # Colors stored as (r,g,b) tuples — JSON-serialisable, no Qt needed.
            "class_colors": {
                name: list(rgb) for name, rgb in state.class_colors.items()
            },
            "images": {},
        }

        saved_count = 0
        for name in state.image_files:
            anns = state.annotations.get(name, [])
            is_rev = state.is_reviewed(name)

            payload["images"][name] = {
                "inspector": state.inspectors.get(name, "") if is_rev else "",
                "note": state.notes.get(name, "") if is_rev else "",
                "annotations": [
                    {
                        "class": a["category_name"],
                        "polygon": [(float(x), float(y)) for (x, y) in a["polygon"]],
                        "thickness": a.get("thickness", 2.0),
                    }
                    for a in anns
                ],
            }

            if not anns:
                continue

            src = Path(state.image_dir) / name
            if not src.exists():
                continue

            try:
                base = Image.open(src).convert("RGBA")
                overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay, "RGBA")

                for a in anns:
                    pts = [(float(x), float(y)) for (x, y) in a["polygon"]]
                    if len(pts) < 2:
                        continue
                    rgb = state.class_colors.get(a["category_name"], (255, 255, 255))
                    draw.polygon(pts, fill=(*rgb, 80), outline=(*rgb, 255))
                    draw.line(pts + [pts[0]], fill=(*rgb, 255), width=3)

                composed = Image.alpha_composite(base, overlay).convert("RGB")
                out_name = f"{tray_name}_{Path(name).stem}_{timestamp}_poly.jpg"
                composed.save(out_path / out_name, "JPEG", quality=95)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to export overlay for {name}: {e}")

        data_path = out_path / f"{tray_name}_{timestamp}_data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.debug(
            "Successfully exported %d overlay images and data JSON.", saved_count
        )
        return f"Saved {saved_count} image(s) + data JSON:\n{data_path}"

    def export_csv(self, out_path: str) -> str:
        """Write per-image metadata to a CSV file at *out_path*.

        Each row contains tray name, image filename, inspector, note, and a
        comma-separated list of unique annotation class names (or ``"good"``
        when the image has been reviewed but carries no annotations).

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
                    "inspector": state.inspectors.get(name, "") if reviewed else "",
                    "note": state.notes.get(name, "") if reviewed else "",
                    "classes": (", ".join(unique_classes) if unique_classes else "good")
                    if reviewed
                    else "",
                }
            )

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["tray", "image_name", "inspector", "note", "classes"]
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

    # ------------------------------------------------------------------ #
    # Import
    # ------------------------------------------------------------------ #

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
