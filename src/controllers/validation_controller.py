"""
ValidationController — headless validation orchestration.

Rules:
  - No Qt GUI types (no QFileDialog, QMessageBox, QColor).
  - All methods accept plain Python values; errors raised as exceptions.
  - QThread is permitted here — it is infrastructure, not UI.
"""

import os
import glob
import json
import logging
import re
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal

from core.logic.mask_comparator import MaskComparator
from core.logic.comparison_logger import write_log_header, log_results, log_skip
from models.validation_model import ValidationModel

logger = logging.getLogger("Validation.Controller")


# ---------------------------------------------------------------------------
# Filename ID extractor (pure Python — shared by both workers)
# ---------------------------------------------------------------------------


def get_robust_id(filename: str) -> str:
    """Extract a stable identifier used to match JSON keys to GT/prediction masks.

    Applies a series of regex patterns in priority order so that a variety of
    real-world filename conventions map to a consistent key.

    Patterns tried in order:

    1. ``<num>_images_<num>`` → ``"<num>_<num>"``
       (e.g. ``118_images_003_01-25-26_poly.jpg`` → ``"118_003"``)
    2. ``_<3+digits>_``       → the captured digits
       (e.g. ``hole_003_02-16-26_poly.jpg`` → ``"003"``)
    3. Two or more groups of 3+ digits → ``"<first>_<second>"``
    4. One group of 3+ digits → that group alone
    5. First digit sequence anywhere in the name
    6. Bare stem (no extension)

    Args:
        filename (str): Basename or full path of the file to identify.

    Returns:
        str: Stable identifier string derived from *filename*.
    """
    m = re.search(r"(\d+)_images_(\d+)", filename)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    m = re.search(r"_(\d{3,})_", filename)
    if m:
        return m.group(1)

    fallback = re.findall(r"(\d{3,})", filename)
    if len(fallback) >= 2:
        return f"{fallback[0]}_{fallback[1]}"
    if len(fallback) == 1:
        return fallback[0]

    first = re.search(r"(\d+)", filename)
    if first:
        return first.group(1)

    return os.path.splitext(filename)[0]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class MaskGenWorker(QThread):
    """Background thread that parses JSON annotations and writes binary mask PNGs.

    Reads polygon annotations from a VIA-style or custom JSON file, renders
    each annotated polygon onto a blank canvas sized to the source image, and
    saves the result as a binary PNG mask. Emits progress and log signals
    throughout.

    Attributes:
        input_dir (str): Directory containing the source images.
        json_path (str): Path to the JSON annotation file.
        output_dir (str): Directory where binary mask PNGs are written.

    Signals:
        progress (int): Emitted after each image with a 0–100 completion %.
        log_message (str): Emitted for informational, warning, and error text.
        finished (): Emitted once the worker exits, regardless of outcome.
    """

    progress = Signal(int)  # 0–100
    log_message = Signal(str)
    finished = Signal()

    def __init__(self, input_dir: str, json_path: str, output_dir: str) -> None:
        """Initialize MaskGenWorker with paths needed for mask generation.

        Args:
            input_dir (str): Directory containing the source images.
            json_path (str): Path to the JSON annotation file.
            output_dir (str): Directory where binary mask PNGs will be written.
        """
        super().__init__()
        self.input_dir = input_dir
        self.json_path = json_path
        self.output_dir = output_dir

    def run(self) -> None:
        """Execute mask generation for all images in *input_dir*.

        Matches each image file to a JSON entry via :func:`get_robust_id`,
        renders annotated polygons to a binary mask, and saves the result.
        Always emits :attr:`finished` on exit, even when an error occurs.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_data_map = data.get("images", {})
                if not image_data_map and isinstance(data, dict):
                    image_data_map = data.get("_via_img_metadata", data)
            except Exception as e:
                self.log_message.emit(f"Critical Error loading JSON: {e}")
                return

            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                files.extend(glob.glob(os.path.join(self.input_dir, ext)))

            total = len(files)
            if total == 0:
                self.log_message.emit("Error: No images found in input folder.")
                return

            processed = 0
            for i, filepath in enumerate(files):
                try:
                    filename = os.path.basename(filepath)
                    image_id = get_robust_id(filename)

                    json_key = next(
                        (
                            k
                            for k in image_data_map
                            if k == f"{image_id}.png" or k == f"{image_id}.jpg"
                        ),
                        None,
                    )
                    if not json_key:
                        json_key = next(
                            (k for k in image_data_map if image_id in k), None
                        )
                    if not json_key and "_" in image_id:
                        simple = image_id.split("_")[-1]
                        json_key = next(
                            (k for k in image_data_map if k.startswith(f"{simple}.")),
                            None,
                        )

                    if json_key:
                        img = cv2.imread(filepath)
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        final_mask = np.zeros((h, w), dtype=np.uint8)

                        entry = image_data_map[json_key]
                        annotations = entry.get("annotations", [])
                        if isinstance(annotations, dict):
                            annotations = annotations.values()

                        drawn = 0
                        for ann in annotations:
                            poly_points = ann.get("polygon")
                            if not poly_points and "shape_attributes" in ann:
                                sa = ann["shape_attributes"]
                                if sa.get("name") == "polygon":
                                    poly_points = list(
                                        zip(
                                            sa.get("all_points_x", []),
                                            sa.get("all_points_y", []),
                                        )
                                    )
                            if poly_points:
                                pts = np.array(poly_points, dtype=np.int32).reshape(
                                    (-1, 1, 2)
                                )
                                cv2.fillPoly(final_mask, [pts], 255)
                                drawn += 1

                        if drawn > 0:
                            out_name = f"{image_id}_binary_mask.png"
                            cv2.imwrite(
                                os.path.join(self.output_dir, out_name), final_mask
                            )
                            self.log_message.emit(f"✓ Matched {filename} → mask saved")
                            processed += 1
                    else:
                        self.log_message.emit(
                            f"Warning: ID {image_id} not found in JSON."
                        )

                except Exception as e:
                    self.log_message.emit(f"Error processing {filename}: {e}")

                self.progress.emit(int((i + 1) / total * 100))

            self.log_message.emit(
                f"Generation complete. Processed {processed}/{total}."
            )
        finally:
            self.finished.emit()


class EvaluationWorker(QThread):
    """Background thread that compares GT masks against prediction masks.

    For each ground-truth mask, locates the corresponding prediction mask
    using :func:`get_robust_id`, runs
    :class:`~core.logic.mask_comparator.MaskComparator`, saves the overlay
    image, and emits per-match metric signals. Writes an
    ``evaluation_log.txt`` to *out_dir* on completion.

    Attributes:
        gt_dir (str): Directory containing ground-truth binary mask images.
        pred_dir (str): Directory containing prediction binary mask images.
        out_dir (str): Directory where overlay PNGs and the log are written.

    Signals:
        progress (int): Emitted after each GT image with a 0–100 completion %.
        log_message (str): Emitted for status, warning, and error messages.
        match_found (str, str, float): Emitted for each successful match as
            ``(overlay_image_path, display_text, iou)``.
        finished (): Emitted once the worker exits, regardless of outcome.
    """

    progress = Signal(int)  # 0–100
    log_message = Signal(str)
    match_found = Signal(str, str, float)  # (overlay_image_path, display_text, iou)
    finished = Signal()

    def __init__(self, gt_dir: str, pred_dir: str, out_dir: str) -> None:
        """Initialize EvaluationWorker with paths needed for mask evaluation.

        Args:
            gt_dir (str): Directory containing ground-truth binary mask images.
            pred_dir (str): Directory containing prediction binary mask images.
            out_dir (str): Directory where overlay PNGs and the log are written.
        """
        super().__init__()
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.out_dir = out_dir

    def run(self) -> None:
        """Execute evaluation for all ground-truth masks found in *gt_dir*.

        Builds a prediction lookup map keyed by robust ID, then iterates
        sorted GT files, thresholds and optionally resizes masks to match GT
        dimensions, computes metrics via
        :class:`~core.logic.mask_comparator.MaskComparator`, and writes an
        ``evaluation_log.txt`` to *out_dir*. Always emits :attr:`finished` on
        exit, even when an error occurs.
        """
        try:
            outline_color = (0, 0, 255)
            thickness = 2
            comparator = MaskComparator(
                gt_outline_color=outline_color,
                gt_outline_thickness=thickness,
            )
            os.makedirs(self.out_dir, exist_ok=True)

            log_path = os.path.join(self.out_dir, "evaluation_log.txt")
            with open(log_path, "w", encoding="utf-8") as log_file:
                write_log_header(
                    log_file,
                    self.gt_dir,
                    self.pred_dir,
                    self.out_dir,
                    outline_color,
                    thickness,
                )

                valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
                gt_files = []
                pred_files_raw = []
                for ext in valid_exts:
                    gt_files.extend(glob.glob(os.path.join(self.gt_dir, ext)))
                    pred_files_raw.extend(glob.glob(os.path.join(self.pred_dir, ext)))

                total = len(gt_files)
                if total == 0:
                    self.log_message.emit(
                        "Error: No images found in Ground Truth folder."
                    )
                    log_file.write("ERROR: No Ground Truth images found.\n")
                    return

                pred_map = {
                    get_robust_id(os.path.basename(p)): p for p in pred_files_raw
                }

                for i, gt_path in enumerate(sorted(gt_files)):
                    gt_filename = os.path.basename(gt_path)
                    gt_id = get_robust_id(gt_filename)

                    if gt_id in pred_map:
                        pred_path = pred_map[gt_id]
                        try:
                            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                            if gt is None or pred is None:
                                continue

                            _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
                            _, pred = cv2.threshold(pred, 1, 255, cv2.THRESH_BINARY)

                            if gt.shape != pred.shape:
                                pred = cv2.resize(
                                    pred,
                                    (gt.shape[1], gt.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )

                            _, overlay, metrics = comparator.compare_masks(gt, pred)

                            out_path = os.path.join(self.out_dir, f"eval_{gt_id}.png")
                            cv2.imwrite(out_path, overlay)

                            iou = metrics["iou"]
                            self.log_message.emit(f"✓ Match: {gt_id} | IoU: {iou:.1f}%")
                            self.match_found.emit(
                                out_path, f"Tray_Image: {gt_id} | IoU: {iou:.1f}%", iou
                            )
                            log_results(log_file, gt_id, metrics)

                        except Exception as e:
                            self.log_message.emit(f"Error evaluating {gt_id}: {e}")
                            log_skip(log_file, gt_id, f"Processing Error: {e}")
                    else:
                        self.log_message.emit(
                            f"⚠ Skip {gt_id}: No matching prediction found."
                        )
                        log_skip(log_file, gt_id, "No prediction match found")

                    self.progress.emit(int((i + 1) / total * 100))

            self.log_message.emit(f"Evaluation complete. Log saved to: {log_path}")
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class ValidationController:
    """Headless orchestration for the two-step validation workflow.

    Creates and returns :class:`MaskGenWorker` or :class:`EvaluationWorker`
    instances pre-configured from the model's current paths. The caller
    (View) is responsible for connecting signals and calling ``start()``.

    Attributes:
        model (ValidationModel): Data model supplying the path configuration.
    """

    def __init__(self, model: ValidationModel) -> None:
        """Initialize ValidationController with a bound ValidationModel.

        Args:
            model (ValidationModel): Data model supplying paths for both the
                mask generation and evaluation workflows.
        """
        self.model = model
        self._gen_worker: Optional[MaskGenWorker] = None
        self._eval_worker: Optional[EvaluationWorker] = None

    def start_generation(self) -> MaskGenWorker:
        """Build and return a MaskGenWorker from current model paths.

        Stops any previously running generation worker before creating a new
        one, ensuring only one generation job runs at a time.

        Returns:
            MaskGenWorker: A new, un-started worker pre-configured with the
                model's polygon source, JSON annotation, and mask output paths.

        Raises:
            ValueError: If ``model.can_generate()`` returns ``False`` — i.e.
                the images folder, JSON file, or mask output folder is unset.
        """
        if not self.model.can_generate():
            raise ValueError(
                "Images folder, JSON file, and mask output folder must all be set."
            )
        self._stop(self._gen_worker)
        self._gen_worker = MaskGenWorker(
            self.model.get_poly_path(),
            self.model.get_json_path(),
            self.model.get_mask_out_path(),
        )
        return self._gen_worker

    def start_evaluation(self) -> EvaluationWorker:
        """Build and return an EvaluationWorker from current model paths.

        Stops any previously running evaluation worker before creating a new
        one, ensuring only one evaluation job runs at a time.

        Returns:
            EvaluationWorker: A new, un-started worker pre-configured with
                the model's GT, prediction, and evaluation output paths.

        Raises:
            ValueError: If ``model.can_evaluate()`` returns ``False`` — i.e.
                the ground truth or predictions folder is unset.
        """
        if not self.model.can_evaluate():
            raise ValueError(
                "Ground truth folder and predictions folder must both be set."
            )
        self._stop(self._eval_worker)
        self._eval_worker = EvaluationWorker(
            self.model.get_gt_path(),
            self.model.get_pred_path(),
            self.model.get_eval_out_path(),
        )
        return self._eval_worker

    @staticmethod
    def _stop(worker: Optional[QThread]) -> None:
        """Gracefully stop a QThread worker if it is currently running.

        Args:
            worker (Optional[QThread]): The worker thread to stop, or ``None``
                (in which case this method is a no-op).
        """
        if worker and worker.isRunning():
            worker.quit()
            worker.wait()
