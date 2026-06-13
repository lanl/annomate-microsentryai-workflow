"""
ProjectIO — headless save/load for .annoproj project files.

No Qt dependencies. Accepts state objects and plain paths. Separates
load_project (disk read) from apply_project_to_states (state mutation)
so callers can inspect data before applying it.
"""

import json
import logging
import os
import shutil
import time
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
        inference_state,
        created_at: Optional[str] = None,
        save_score_maps: bool = True,
        model_path: str = "",
        calibration_state=None,
        center_template_state=None,
        anomaly_constraint_state=None,
    ) -> str:
        """Write .annoproj + annotations.coco.json to project_dir.

        Creates project_dir if it does not exist. Returns the absolute path
        to the written .annoproj file. Raises OSError if the directory cannot
        be created or any file cannot be written.

        Args:
            project_dir: Directory that will contain all project files.
            project_name: Human-readable project name (used as filename stem).
            dataset_state: DatasetState instance.
            inference_state: InferenceState instance.
            created_at: ISO timestamp from the original save; if None, uses now.
            save_score_maps: When True, write inference score maps to NPZ.
            model_path: Absolute path to the inference model file (informational).
        """
        _t0 = time.perf_counter()

        project_dir = str(Path(project_dir).resolve())
        os.makedirs(project_dir, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        if created_at is None:
            created_at = now

        coco_path = os.path.join(project_dir, _COCO_FILENAME)
        _t1 = time.perf_counter()
        self.export_coco(coco_path, dataset_state)
        _t2 = time.perf_counter()
        logger.info("save_project [export_coco]:      %.3fs", _t2 - _t1)

        score_maps_file = ""
        if (
            save_score_maps
            and inference_state.score_maps
            and inference_state.score_maps_dirty
        ):
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
                inference_state.score_maps_dirty = False
            except Exception as exc:
                logger.warning("Could not save score maps: %s", exc)
        elif inference_state.score_maps:
            score_maps_file = _SCOREMAPS_FILENAME
        _t3 = time.perf_counter()
        logger.info(
            "save_project [score_maps npz]:   %.3fs (dirty=%s)",
            _t3 - _t2,
            inference_state.score_maps_dirty,
        )

        scores_by_fname = {
            os.path.basename(k): v for k, v in inference_state.scores.items()
        }
        labels_by_fname = {
            os.path.basename(k): v for k, v in inference_state.labels.items()
        }
        per_image = {}
        all_fnames = (
            set(dataset_state.image_files) | set(scores_by_fname) | set(labels_by_fname)
        )
        for fname in all_fnames:
            entry = {}
            score = scores_by_fname.get(fname)
            label = labels_by_fname.get(fname)
            decision = dataset_state.review_decisions.get(fname, "")
            decision_at = dataset_state.decision_timestamps.get(fname, "")
            inspector = dataset_state.inspectors.get(fname, "")
            note = dataset_state.notes.get(fname, "")
            omit_reason = dataset_state.omit_reasons.get(fname, "")
            if score is not None:
                entry["score"] = score
            if label is not None:
                entry["label"] = label
            if decision:
                entry["decision"] = decision
            if decision_at:
                entry["decision_at"] = decision_at
            if inspector:
                entry["inspector"] = inspector
            if note:
                entry["note"] = note
            if omit_reason:
                entry["omit_reason"] = omit_reason
            if entry:
                per_image[fname] = entry
        _t4 = time.perf_counter()
        logger.info("save_project [build per_image]:  %.3fs", _t4 - _t3)

        proj = {
            "version": _SCHEMA_VERSION,
            "created_at": created_at,
            "modified_at": now,
            "project_name": project_name,
            "dataset": {
                "image_dir": self._as_relative_path(
                    dataset_state.image_dir or "", project_dir
                ),
                "class_names": list(dataset_state.class_names),
                "class_colors": {
                    name: list(rgb) for name, rgb in dataset_state.class_colors.items()
                },
            },
            "annotations_file": _COCO_FILENAME,
            "per_image": per_image,
            "inference": {
                "model_path": self._as_relative_path(model_path, project_dir),
                "score_maps_file": score_maps_file,
            },
        }

        if calibration_state is not None:
            cs = calibration_state
            proj["calibration"] = {
                "scale": cs.scale,
                "unit": cs.unit,
                "px_count": cs.px_count,
                "world_val": cs.world_val,
                "user_calibrated": cs.user_calibrated,
                "calib_p1": list(cs.calib_p1) if cs.calib_p1 else None,
                "calib_p2": list(cs.calib_p2) if cs.calib_p2 else None,
                "real_distance": cs.real_distance,
                "grid_visible": cs.grid_visible,
                "grid_color": list(cs.grid_color),
                "grid_opacity": cs.grid_opacity,
                "grid_spacing_world": cs.grid_spacing_world,
                "grid_spacing_auto": cs.grid_spacing_auto,
            }

        if center_template_state is not None:
            ts = center_template_state
            proj["center_template"] = {
                "enabled": ts.enabled,
                "template_file": self._as_relative_path(
                    ts.template_path or ts.template_file, project_dir
                ),
                "anchor_x": ts.anchor_x,
                "anchor_y": ts.anchor_y,
                "crop_shape": ts.crop_shape,
                "crop_width": ts.crop_width,
                "crop_height": ts.crop_height,
                "center_x": ts.center_x,
                "center_y": ts.center_y,
            }
            logger.debug(
                "Project center template saved: enabled=%s file=%s "
                "anchor=(%s, %s) crop=%s %sx%s center=(%s, %s)",
                ts.enabled,
                proj["center_template"]["template_file"],
                ts.anchor_x,
                ts.anchor_y,
                ts.crop_shape,
                ts.crop_width,
                ts.crop_height,
                ts.center_x,
                ts.center_y,
            )

        if anomaly_constraint_state is not None:
            proj["anomaly_constraints"] = anomaly_constraint_state.to_dict()

        annoproj_path = os.path.join(project_dir, f"{project_name}.annoproj")
        _t5 = time.perf_counter()
        with open(annoproj_path, "w", encoding="utf-8") as f:
            json.dump(proj, f, indent=2)
        _t6 = time.perf_counter()
        logger.info("save_project [write annoproj]:   %.3fs", _t6 - _t5)
        logger.info("save_project [total]:            %.3fs", _t6 - _t0)

        logger.debug("Project saved to: %s", annoproj_path)
        return annoproj_path

    def export_template(
        self,
        template_path: str,
        project_name: str,
        dataset_state,
        calibration_state=None,
        center_template_state=None,
        anomaly_constraint_state=None,
    ) -> str:
        """Write a settings-only .annoproj template (no images, annotations, or inference).

        The written file carries is_template=true and omits image_dir, annotations_file,
        per_image, and inference sections. It can be opened with load_project/
        apply_project_to_states — missing sections are handled gracefully by the loader.

        Args:
            template_path: Absolute path for the output .annoproj file.
            project_name: Human-readable name embedded in the template.
            dataset_state: DatasetState (only class_names/class_colors are used).
            calibration_state: CalibrationState or None.
            center_template_state: CenterTemplateState or None.
            anomaly_constraint_state: AnomalyConstraintState or None.

        Returns:
            The absolute path to the written file.
        """
        template_path = str(Path(template_path).resolve())
        template_dir = str(Path(template_path).parent)
        os.makedirs(template_dir, exist_ok=True)

        tmpl = {
            "version": _SCHEMA_VERSION,
            "is_template": True,
            "project_name": project_name,
            "dataset": {
                "class_names": list(dataset_state.class_names),
                "class_colors": {
                    name: list(rgb) for name, rgb in dataset_state.class_colors.items()
                },
            },
        }

        if calibration_state is not None:
            cs = calibration_state
            tmpl["calibration"] = {
                "scale": cs.scale,
                "unit": cs.unit,
                "px_count": cs.px_count,
                "world_val": cs.world_val,
                "user_calibrated": cs.user_calibrated,
                "calib_p1": list(cs.calib_p1) if cs.calib_p1 else None,
                "calib_p2": list(cs.calib_p2) if cs.calib_p2 else None,
                "real_distance": cs.real_distance,
                "grid_visible": cs.grid_visible,
                "grid_color": list(cs.grid_color),
                "grid_opacity": cs.grid_opacity,
                "grid_spacing_world": cs.grid_spacing_world,
                "grid_spacing_auto": cs.grid_spacing_auto,
            }

        if center_template_state is not None:
            ts = center_template_state
            src = ts.template_path or ts.template_file
            if src and os.path.isfile(src):
                img_filename = os.path.basename(src)
                dst = os.path.join(template_dir, img_filename)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                stored_file = img_filename
            else:
                stored_file = self._as_relative_path(src, template_dir) if src else ""
            tmpl["center_template"] = {
                "enabled": ts.enabled,
                "template_file": stored_file,
                "anchor_x": ts.anchor_x,
                "anchor_y": ts.anchor_y,
                "crop_shape": ts.crop_shape,
                "crop_width": ts.crop_width,
                "crop_height": ts.crop_height,
                "center_x": ts.center_x,
                "center_y": ts.center_y,
            }

        if anomaly_constraint_state is not None:
            tmpl["anomaly_constraints"] = anomaly_constraint_state.to_dict()

        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(tmpl, f, indent=2)

        logger.debug("Project template exported to: %s", template_path)
        return template_path

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
        tmpl_data = data.get("center_template", {})
        if "template_file" in tmpl_data:
            tmpl_data["_resolved_template_path"] = self._resolve_path(
                tmpl_data.get("template_file", ""), proj_dir
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
        inference_state,
        calibration_state=None,
        center_template_state=None,
        anomaly_constraint_state=None,
    ) -> None:
        """Mutate state objects from load_project() output.

        Does NOT touch image_dir or image_files — those must be set by the
        caller before invoking this method (e.g. via ProjectController which
        scans the directory first). This method repopulates annotations,
        class registry, inspectors, notes, and inference cache on top of
        whatever image list is already in state.

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
            dataset_state.class_names = [n.lower() for n in class_names]
            raw_colors = {k.lower(): v for k, v in ds.get("class_colors", {}).items()}
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
            dataset_state.class_visibility = {
                name: True for name in dataset_state.class_names
            }
        else:
            dataset_state.reset_classes()

        # Annotations from COCO file
        coco_path = project_data.get("resolved_coco_path", "")
        if coco_path and os.path.exists(coco_path):
            try:
                self.import_coco(coco_path, dataset_state)
            except Exception as exc:
                logger.warning("Could not load COCO annotations: %s", exc)

        image_dir = project_data.get("dataset", {}).get("image_dir", "")

        if "per_image" in project_data:
            for fname, info in project_data["per_image"].items():
                abs_path = os.path.join(image_dir, fname) if image_dir else fname
                score = info.get("score")
                label = info.get("label")
                if score is not None:
                    inference_state.scores[abs_path] = score
                if label is not None:
                    inference_state.labels[abs_path] = label
                if info.get("decision"):
                    dataset_state.review_decisions[fname] = info["decision"]
                if info.get("decision_at"):
                    dataset_state.decision_timestamps[fname] = info["decision_at"]
                if info.get("omit_reason"):
                    dataset_state.omit_reasons[fname] = info["omit_reason"]
                dataset_state.inspectors[fname] = info.get("inspector", "")
                dataset_state.notes[fname] = info.get("note", "")
        else:
            # Legacy format: separate review_status, review_decisions, score_cache, label_cache
            for fname, info in project_data.get("review_status", {}).items():
                dataset_state.inspectors[fname] = info.get("inspector", "")
                dataset_state.notes[fname] = info.get("note", "")
            for fname, decision in project_data.get("review_decisions", {}).items():
                dataset_state.review_decisions[fname] = decision
            inf_data = project_data.get("inference", {})
            for k, v in inf_data.get("score_cache", {}).items():
                abs_k = k if os.path.isabs(k) else os.path.join(image_dir, k)
                inference_state.scores[abs_k] = v
            for k, v in inf_data.get("label_cache", {}).items():
                abs_k = k if os.path.isabs(k) else os.path.join(image_dir, k)
                inference_state.labels[abs_k] = v

        inference_state.inference_cache = dict(inference_state.scores)

        # Score maps from NPZ (optional)
        npz_path = project_data.get("_resolved_npz_path", "")
        if npz_path and os.path.exists(npz_path):
            try:
                npz = np.load(npz_path)
                for key in npz.files:
                    fname = os.path.normpath(self._npz_key_to_filename(key))
                    inference_state.score_maps[fname] = npz[key]
                inference_state.score_maps_dirty = False
            except Exception as exc:
                logger.warning("Could not load score maps from NPZ: %s", exc)

        # Calibration (optional — absent in old project files)
        if calibration_state is not None:
            cdata = project_data.get("calibration", {})
            using_default_pixel_scale = not cdata
            if not cdata:
                calibration_state.clear_calibration()
            else:
                scale = cdata.get("scale", None)
                using_default_pixel_scale = scale is None
                if scale is None:
                    calibration_state.clear_calibration()
                else:
                    calibration_state.scale = scale
                    calibration_state.unit = cdata.get("unit", "mm")
                    calibration_state.px_count = cdata.get("px_count", 1.0)
                    calibration_state.world_val = cdata.get("world_val", scale)
                    calibration_state.user_calibrated = cdata.get(
                        "user_calibrated", True
                    )
            p1 = cdata.get("calib_p1")
            calibration_state.calib_p1 = (
                tuple(p1) if p1 and not using_default_pixel_scale else None
            )
            p2 = cdata.get("calib_p2")
            calibration_state.calib_p2 = (
                tuple(p2) if p2 and not using_default_pixel_scale else None
            )
            calibration_state.real_distance = cdata.get("real_distance", 1.0)
            calibration_state.grid_visible = cdata.get("grid_visible", False)
            color = cdata.get("grid_color", [58, 90, 122])
            calibration_state.grid_color = tuple(color)
            calibration_state.grid_opacity = cdata.get("grid_opacity", 0.5)
            calibration_state.grid_spacing_world = (
                100.0
                if using_default_pixel_scale
                else cdata.get("grid_spacing_world", 100.0)
            )
            calibration_state.grid_spacing_auto = cdata.get("grid_spacing_auto", True)

        if center_template_state is not None:
            tdata = project_data.get("center_template", {})
            resolved_template_path = tdata.get("_resolved_template_path", "")
            template_exists = bool(
                resolved_template_path and os.path.exists(resolved_template_path)
            )
            center_template_state.enabled = bool(tdata.get("enabled", False))
            center_template_state.template_file = tdata.get("template_file", "")
            center_template_state.template_path = resolved_template_path
            center_template_state.anchor_x = int(tdata.get("anchor_x", 0))
            center_template_state.anchor_y = int(tdata.get("anchor_y", 0))
            center_template_state.crop_shape = tdata.get("crop_shape", "circle")
            center_template_state.crop_width = int(tdata.get("crop_width", 1210))
            center_template_state.crop_height = int(tdata.get("crop_height", 1210))
            center_template_state.center_x = tdata.get("center_x", None)
            center_template_state.center_y = tdata.get("center_y", None)
            center_template_state.last_score = None
            if center_template_state.template_file and not template_exists:
                center_template_state.enabled = False
            logger.debug(
                "Project center template loaded: enabled=%s file=%s exists=%s "
                "anchor=(%s, %s) crop=%s %sx%s center=(%s, %s)",
                center_template_state.enabled,
                center_template_state.template_file,
                template_exists,
                center_template_state.anchor_x,
                center_template_state.anchor_y,
                center_template_state.crop_shape,
                center_template_state.crop_width,
                center_template_state.crop_height,
                center_template_state.center_x,
                center_template_state.center_y,
            )

        if anomaly_constraint_state is not None:
            adata = project_data.get("anomaly_constraints", {})
            if adata:
                from core.states.anomaly_constraint_state import AnomalyConstraintState
                loaded = AnomalyConstraintState.from_dict(adata)
                anomaly_constraint_state.enabled = loaded.enabled
                anomaly_constraint_state.area_check_enabled = loaded.area_check_enabled
                anomaly_constraint_state.area_threshold = loaded.area_threshold
                anomaly_constraint_state.area_color = loaded.area_color
                anomaly_constraint_state.distance_check_enabled = loaded.distance_check_enabled
                anomaly_constraint_state.distance_threshold = loaded.distance_threshold
                anomaly_constraint_state.distance_method = loaded.distance_method
                anomaly_constraint_state.distance_color = loaded.distance_color

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
            if fname in dataset_state.image_sizes:
                w, h = dataset_state.image_sizes[fname]
            else:
                img_path = (
                    os.path.join(dataset_state.image_dir, fname)
                    if dataset_state.image_dir
                    else fname
                )
                w, h = self._read_image_size(img_path)
                dataset_state.image_sizes[fname] = (w, h)
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

        _cache_hits = sum(
            1 for f in dataset_state.image_files if f in dataset_state.image_sizes
        )
        logger.info(
            "export_coco: %d images, %d size-cache hits, %d PIL reads",
            len(dataset_state.image_files),
            _cache_hits,
            len(dataset_state.image_files) - _cache_hits,
        )
        _tw = time.perf_counter()
        os.makedirs(str(Path(coco_path).parent), exist_ok=True)
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        logger.info("export_coco [json write]:        %.3fs", time.perf_counter() - _tw)

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
            name = c["name"].lower()
            cat_map[c["id"]] = name
            if name not in dataset_state.class_names:
                idx = len(dataset_state.class_names)
                dataset_state.class_names.append(name)
                dataset_state.class_colors[name] = DEFAULT_CLASS_COLORS[
                    idx % len(DEFAULT_CLASS_COLORS)
                ]
                dataset_state.class_visibility[name] = True

        img_id_map = {}
        for img in data.get("images", []):
            img_id_map[img["id"]] = img["file_name"]
            w, h = img.get("width", 0), img.get("height", 0)
            if w and h:
                dataset_state.image_sizes[img["file_name"]] = (w, h)

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
            fname.replace("\\", "/").replace(".", "__dot__").replace("/", "__slash__")
        )

    def _npz_key_to_filename(self, key: str) -> str:
        """Reverse _filename_to_npz_key."""
        return (
            key.replace("__bslash__", "\\")
            .replace("__slash__", "/")
            .replace("__dot__", ".")
        )

    def _as_relative_path(self, path: str, base_dir: str) -> str:
        if not path:
            return path
        try:
            return Path(os.path.relpath(path, base_dir)).as_posix()
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
