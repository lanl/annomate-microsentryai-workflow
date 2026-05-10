"""
ProjectIO — headless save/load for .annoproj project files.

No Qt dependencies. Accepts state objects and plain paths. Separates
load_project (disk read) from apply_project_to_states (state mutation)
so callers can inspect data before applying it.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.utils.constants import DEFAULT_CLASS_COLORS
from core.utils.geometry import polygon_area, polygon_bbox

logger = logging.getLogger("AnnoMate.ProjectIO")

_SCHEMA_VERSION = "1.0"
_COCO_FILENAME = "annotations.coco.json"
_SCOREMAPS_FILENAME = "scoremaps.npz"


class ProjectIO:
    """Headless save/load for .annoproj project files.

    All methods accept plain Python values. No Qt dependencies. Raises
    standard exceptions (OSError, json.JSONDecodeError, KeyError) on failure;
    callers are responsible for user-facing error display.
    """

    SCHEMA_VERSION = _SCHEMA_VERSION
    COCO_FILENAME = _COCO_FILENAME
    SCOREMAPS_FILENAME = _SCOREMAPS_FILENAME

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def save_project(
        self,
        project_dir: str,
        project_name: str,
        dataset_state,
        validation_state,
        inference_state,
        created_at: Optional[str] = None,
        save_score_maps: bool = True,
        model_path: str = "",
    ) -> str:
        """Write .annoproj + annotations.coco.json to project_dir.

        Creates project_dir if it does not exist. Returns the absolute path
        to the written .annoproj file. Raises OSError if the directory cannot
        be created or any file cannot be written.

        Args:
            project_dir: Directory that will contain all project files.
            project_name: Human-readable project name (used as filename stem).
            dataset_state: DatasetState instance.
            validation_state: ValidationState instance.
            inference_state: InferenceState instance.
            created_at: ISO timestamp from the original save; if None, uses now.
            save_score_maps: When True, write inference score maps to NPZ.
            model_path: Absolute path to the inference model file (informational).
        """
        project_dir = str(Path(project_dir).resolve())
        os.makedirs(project_dir, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        if created_at is None:
            created_at = now

        coco_path = os.path.join(project_dir, _COCO_FILENAME)
        self.export_coco(coco_path, dataset_state)

        score_maps_file = ""
        if save_score_maps and inference_state.score_maps:
            npz_path = os.path.join(project_dir, _SCOREMAPS_FILENAME)
            try:
                np.savez_compressed(
                    npz_path,
                    **{
                        self._filename_to_npz_key(f): arr
                        for f, arr in inference_state.score_maps.items()
                    },
                )
                score_maps_file = _SCOREMAPS_FILENAME
            except Exception as exc:
                logger.warning("Could not save score maps: %s", exc)

        review_status = {}
        for fname in dataset_state.image_files:
            inspector = dataset_state.inspectors.get(fname, "")
            note = dataset_state.notes.get(fname, "")
            if inspector or note:
                review_status[fname] = {"inspector": inspector, "note": note}

        proj = {
            "version": _SCHEMA_VERSION,
            "created_at": created_at,
            "modified_at": now,
            "project_name": project_name,
            "dataset": {
                "image_dir": self._make_relative_if_inside(
                    dataset_state.image_dir or "", project_dir
                ),
                "class_names": list(dataset_state.class_names),
                "class_colors": {
                    name: list(rgb) for name, rgb in dataset_state.class_colors.items()
                },
            },
            "annotations_file": _COCO_FILENAME,
            "validation": {
                "poly_path": validation_state.poly_path,
                "json_path": validation_state.json_path,
                "mask_out_path": validation_state.mask_out_path,
                "gt_path": validation_state.gt_path,
                "pred_path": validation_state.pred_path,
                "eval_out_path": validation_state.eval_out_path,
            },
            "review_status": review_status,
            "review_decisions": {
                fname: dataset_state.review_decisions[fname]
                for fname in dataset_state.image_files
                if fname in dataset_state.review_decisions
            },
            "inference": {
                "score_cache": dict(inference_state.inference_cache),
                "score_maps_file": score_maps_file,
                "model_path": self._make_relative_if_inside(model_path, project_dir),
            },
        }

        annoproj_path = os.path.join(project_dir, f"{project_name}.annoproj")
        with open(annoproj_path, "w", encoding="utf-8") as f:
            json.dump(proj, f, indent=2)

        logger.debug("Project saved to: %s", annoproj_path)
        return annoproj_path

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_project(self, annoproj_path: str) -> dict:
        """Read a .annoproj file and return its parsed data.

        Does NOT mutate any state objects. Call apply_project_to_states()
        after this to apply the data. Resolves the COCO annotations path
        and adds a 'resolved_coco_path' key to the returned dict.

        Args:
            annoproj_path: Absolute path to the .annoproj file.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        annoproj_path = str(Path(annoproj_path).resolve())
        with open(annoproj_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["_annoproj_path"] = annoproj_path

        proj_dir = str(Path(annoproj_path).parent)
        ds_data = data.get("dataset", {})
        if "image_dir" in ds_data:
            ds_data["image_dir"] = self._resolve_path(ds_data["image_dir"], proj_dir)
        inf_data = data.get("inference", {})
        if "model_path" in inf_data:
            inf_data["model_path"] = self._resolve_path(
                inf_data["model_path"], proj_dir
            )

        data["resolved_coco_path"] = self._resolve_coco_path(annoproj_path, data)

        npz_file = data.get("inference", {}).get("score_maps_file", "")
        if npz_file:
            proj_dir = str(Path(annoproj_path).parent)
            abs_npz = os.path.join(proj_dir, npz_file)
            data["_resolved_npz_path"] = abs_npz if os.path.exists(abs_npz) else ""
        else:
            data["_resolved_npz_path"] = ""

        return data

    def apply_project_to_states(
        self,
        project_data: dict,
        dataset_state,
        validation_state,
        inference_state,
    ) -> None:
        """Mutate the three state objects from load_project() output.

        Does NOT touch image_dir or image_files — those must be set by the
        caller before invoking this method (e.g. via ProjectController which
        scans the directory first). This method repopulates annotations,
        class registry, inspectors, notes, validation paths, and inference
        cache on top of whatever image list is already in state.

        Args:
            project_data: Dict returned by load_project().
            dataset_state: DatasetState to populate.
            validation_state: ValidationState to populate.
            inference_state: InferenceState to populate.
        """
        ds = project_data.get("dataset", {})

        # Class registry
        class_names = ds.get("class_names", [])
        if class_names:
            dataset_state.class_names = list(class_names)
            raw_colors = ds.get("class_colors", {})
            dataset_state.class_colors = {}
            for i, name in enumerate(dataset_state.class_names):
                raw = raw_colors.get(name)
                if isinstance(raw, (list, tuple)) and len(raw) == 3:
                    dataset_state.class_colors[name] = (
                        int(raw[0]),
                        int(raw[1]),
                        int(raw[2]),
                    )
                else:
                    dataset_state.class_colors[name] = DEFAULT_CLASS_COLORS[
                        i % len(DEFAULT_CLASS_COLORS)
                    ]

        # Annotations from COCO file
        coco_path = project_data.get("resolved_coco_path", "")
        if coco_path and os.path.exists(coco_path):
            try:
                self.import_coco(coco_path, dataset_state)
            except Exception as exc:
                logger.warning("Could not load COCO annotations: %s", exc)

        # Review status (inspector / note)
        for fname, info in project_data.get("review_status", {}).items():
            dataset_state.inspectors[fname] = info.get("inspector", "")
            dataset_state.notes[fname] = info.get("note", "")

        # Image-level review decisions
        for fname, decision in project_data.get("review_decisions", {}).items():
            dataset_state.review_decisions[fname] = decision

        # Validation paths
        vdata = project_data.get("validation", {})
        validation_state.poly_path = vdata.get("poly_path", "")
        validation_state.json_path = vdata.get("json_path", "")
        validation_state.mask_out_path = vdata.get("mask_out_path", "")
        validation_state.gt_path = vdata.get("gt_path", "")
        validation_state.pred_path = vdata.get("pred_path", "")
        validation_state.eval_out_path = vdata.get("eval_out_path", "")

        # Inference cache (float scores only — fast to restore)
        inf_data = project_data.get("inference", {})
        inference_state.inference_cache = dict(inf_data.get("score_cache", {}))

        # Score maps from NPZ (optional)
        npz_path = project_data.get("_resolved_npz_path", "")
        if npz_path and os.path.exists(npz_path):
            try:
                npz = np.load(npz_path)
                for key in npz.files:
                    fname = self._npz_key_to_filename(key)
                    inference_state.score_maps[fname] = npz[key]
            except Exception as exc:
                logger.warning("Could not load score maps from NPZ: %s", exc)

    # ------------------------------------------------------------------ #
    # COCO export / import
    # ------------------------------------------------------------------ #

    def export_coco(self, coco_path: str, dataset_state) -> None:
        """Write a standard COCO Instance Segmentation JSON to coco_path.

        Reads image dimensions from disk via PIL. Missing images produce
        width=0, height=0 entries (valid COCO sentinel). Only images that
        have at least one annotation are included in the annotations array;
        all loaded images appear in the images array.

        Args:
            coco_path: Absolute path for the output JSON file.
            dataset_state: DatasetState whose annotations are exported.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        coco = {
            "info": {
                "description": "AnnoMate export",
                "version": _SCHEMA_VERSION,
                "date_created": now_str,
            },
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": [],
        }

        cat_id_map = {}
        for i, name in enumerate(dataset_state.class_names, start=1):
            coco["categories"].append({"id": i, "name": name, "supercategory": ""})
            cat_id_map[name] = i

        ann_id = 1
        for img_id, fname in enumerate(dataset_state.image_files, start=1):
            img_path = (
                os.path.join(dataset_state.image_dir, fname)
                if dataset_state.image_dir
                else fname
            )
            w, h = self._read_image_size(img_path)
            coco["images"].append(
                {"id": img_id, "file_name": fname, "width": w, "height": h}
            )

            for ann_rec in dataset_state.annotations.get(fname, []):
                polygon = ann_rec["polygon"]
                cat_name = ann_rec["category_name"]
                cat_id = cat_id_map.get(cat_name)
                if cat_id is None:
                    cat_id = len(coco["categories"]) + 1
                    coco["categories"].append(
                        {"id": cat_id, "name": cat_name, "supercategory": ""}
                    )
                    cat_id_map[cat_name] = cat_id

                coco["annotations"].append(
                    self._build_coco_annotation(ann_id, img_id, cat_id, polygon)
                )
                ann_id += 1

        os.makedirs(str(Path(coco_path).parent), exist_ok=True)
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        logger.debug("COCO JSON written to: %s (%d annotations)", coco_path, ann_id - 1)

    def import_coco(self, coco_path: str, dataset_state) -> None:
        """Read a COCO JSON file into dataset_state (annotations + classes).

        Does NOT clear existing data — merges on top of whatever is already
        in state. Callers should clear state if a full replacement is needed.
        Follows the same logic as IOController._import_coco_format.

        Args:
            coco_path: Absolute path to the COCO JSON file.
            dataset_state: DatasetState to populate.
        """
        with open(coco_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cat_map = {}
        for c in data.get("categories", []):
            name = c["name"]
            cat_map[c["id"]] = name
            if name not in dataset_state.class_names:
                idx = len(dataset_state.class_names)
                dataset_state.class_names.append(name)
                dataset_state.class_colors[name] = DEFAULT_CLASS_COLORS[
                    idx % len(DEFAULT_CLASS_COLORS)
                ]

        img_id_map = {img["id"]: img["file_name"] for img in data.get("images", [])}

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_id_map:
                continue
            filename = img_id_map[img_id]
            cat_name = cat_map.get(ann["category_id"], "Unknown")
            seg = ann.get("segmentation", [])
            polygon = self._coco_seg_to_poly(seg)
            if polygon:
                dataset_state.annotations.setdefault(filename, []).append(
                    {"category_name": cat_name, "polygon": polygon}
                )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _poly_to_coco_seg(self, polygon: list) -> list:
        """Convert [(x,y), ...] to COCO segmentation [[x0,y0,x1,y1,...]]."""
        flat = [coord for pt in polygon for coord in (float(pt[0]), float(pt[1]))]
        return [flat]

    def _coco_seg_to_poly(self, segmentation: list) -> list:
        """Convert COCO segmentation [[x0,y0,...]] to [(x,y), ...]."""
        if not isinstance(segmentation, list) or not segmentation:
            return []
        pts_list = (
            segmentation[0] if isinstance(segmentation[0], list) else segmentation
        )
        poly = []
        for i in range(0, len(pts_list) - 1, 2):
            poly.append((float(pts_list[i]), float(pts_list[i + 1])))
        return poly

    def _build_coco_annotation(
        self, ann_id: int, image_id: int, category_id: int, polygon: list
    ) -> dict:
        """Build a single COCO annotation dict with area, bbox, segmentation."""
        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": self._poly_to_coco_seg(polygon),
            "area": float(polygon_area(polygon)),
            "bbox": [float(v) for v in polygon_bbox(polygon)],
            "iscrowd": 0,
        }

    def _read_image_size(self, image_path: str) -> tuple:
        """Return (width, height) via PIL. Returns (0, 0) if file is missing."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            logger.debug("Could not read dimensions for: %s", image_path)
            return (0, 0)

    def _resolve_coco_path(self, annoproj_path: str, proj_data: dict) -> str:
        """Resolve the COCO JSON path: try absolute first, then relative."""
        abs_path = proj_data.get("annotations_file_abs", "")
        if abs_path and os.path.exists(abs_path):
            return abs_path

        rel_file = proj_data.get("annotations_file", _COCO_FILENAME)
        proj_dir = str(Path(annoproj_path).parent)
        return os.path.join(proj_dir, rel_file)

    def _filename_to_npz_key(self, fname: str) -> str:
        """Sanitize a filename to a valid NPZ array key."""
        return (
            fname.replace(".", "__dot__")
            .replace("/", "__slash__")
            .replace("\\", "__bslash__")
        )

    def _npz_key_to_filename(self, key: str) -> str:
        """Reverse _filename_to_npz_key."""
        return (
            key.replace("__bslash__", "\\")
            .replace("__slash__", "/")
            .replace("__dot__", ".")
        )

    def _make_relative_if_inside(self, path: str, base_dir: str) -> str:
        """Return path relative to base_dir if it lives inside it, else unchanged."""
        if not path:
            return path
        try:
            return str(Path(path).relative_to(base_dir))
        except ValueError:
            return path

    def _resolve_path(self, path: str, base_dir: str) -> str:
        """Resolve a possibly-relative path against base_dir. Absolute paths pass through."""
        if not path:
            return path
        p = Path(path)
        if p.is_absolute():
            return path
        return str((Path(base_dir) / p).resolve())
