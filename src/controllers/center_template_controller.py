import os
import logging

import cv2
from PySide6.QtCore import QObject, QThread, Signal

from core.logic.center_template import extract_template, locate_center


_TEMPLATE_FILENAME = "center_template.png"
logger = logging.getLogger("AnnoMate.CenterTemplateController")


class CenterCropWorker(QThread):
    result_ready = Signal(str, float, float, float)  # (path, cx, cy, score)
    progress = Signal(int)

    def __init__(self, template, anchor_x, anchor_y, file_list):
        super().__init__()
        self._template = template
        self._anchor_x = anchor_x
        self._anchor_y = anchor_y
        self.file_list = file_list
        self._running = True

    def run(self):
        tpl_h, tpl_w = self._template.shape[:2]
        for i, path in enumerate(self.file_list):
            if not self._running:
                break
            try:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    self.progress.emit(i + 1)
                    continue
                img_h, img_w = image.shape[:2]
                if tpl_w > img_w or tpl_h > img_h:
                    logger.warning(
                        "CenterCropWorker: template larger than image %s, skipping",
                        path,
                    )
                    self.progress.emit(i + 1)
                    continue
                cx, cy, score = locate_center(
                    image, self._template, self._anchor_x, self._anchor_y
                )
                self.result_ready.emit(path, float(cx), float(cy), float(score))
            except Exception as e:
                logger.error("CenterCropWorker: failed for %s: %s", path, e)
            self.progress.emit(i + 1)

    def stop(self):
        self._running = False


class CenterTemplateController(QObject):
    """Headless service for saving and matching center templates."""

    template_saved = Signal(str)
    template_cleared = Signal()
    match_ready = Signal(float, float, float)
    match_failed = Signal(str)
    preload_result = Signal(str, float, float, float)  # (path, cx, cy, score)

    def __init__(self, center_template_model, parent=None) -> None:
        super().__init__(parent)
        self._model = center_template_model
        self._loaded_template = None  # np.ndarray | None — in-memory template cache
        self._match_cache: dict = {}  # path -> (cx, cy, score)
        self._worker = None  # CenterCropWorker | None
        center_template_model.template_changed.connect(self.invalidate_cache)

    def save_template(
        self,
        project_dir: str,
        image_bgr,
        center_x: float,
        center_y: float,
        crop_shape: str,
        crop_width: int,
        crop_height: int,
    ) -> str:
        if not project_dir:
            raise ValueError("Save the project before accepting center calibration.")
        logger.info(
            "Saving center template: project_dir=%s center=(%.1f, %.1f) "
            "shape=%s size=%dx%d",
            project_dir,
            center_x,
            center_y,
            crop_shape,
            crop_width,
            crop_height,
        )
        os.makedirs(project_dir, exist_ok=True)
        template, anchor_x, anchor_y = extract_template(
            image_bgr,
            center_x,
            center_y,
            crop_width,
            crop_height,
        )
        template_path = os.path.join(project_dir, _TEMPLATE_FILENAME)
        if not cv2.imwrite(template_path, template):
            logger.error("Could not write center template: %s", template_path)
            raise OSError(f"Could not write template: {template_path}")
        tpl_h, tpl_w = template.shape[:2]
        logger.info(
            "Center template saved: path=%s template_size=%dx%d anchor=(%d, %d)",
            template_path,
            tpl_w,
            tpl_h,
            anchor_x,
            anchor_y,
        )
        self._model.set_template(
            _TEMPLATE_FILENAME,
            template_path,
            anchor_x,
            anchor_y,
            crop_shape,
            crop_width,
            crop_height,
            center_x,
            center_y,
        )
        self._loaded_template = template
        self._match_cache.clear()
        self._stop_worker()
        self.template_saved.emit(template_path)
        return template_path

    def clear_template(self) -> None:
        logger.info("Clearing center template state.")
        self._stop_worker()
        self._loaded_template = None
        self._match_cache.clear()
        self._model.clear_template()
        self.template_cleared.emit()

    def invalidate_cache(self) -> None:
        """Flush the per-image match cache and stop any running preload worker.

        Called automatically when the model's template_changed signal fires
        (e.g. project load, new template saved) so stale crop positions from
        a previous project can never bleed into the current one.
        """
        self._stop_worker()
        self._loaded_template = None
        self._match_cache.clear()

    def match_image(
        self, image_bgr, image_path: str = ""
    ) -> tuple[float, float, float] | None:
        if image_path and image_path in self._match_cache:
            logger.debug("Center template cache hit for %s", image_path)
            cx, cy, score = self._match_cache[image_path]
            self._model.set_match(cx, cy, score)
            self.match_ready.emit(float(cx), float(cy), float(score))
            return float(cx), float(cy), float(score)

        if not self._model.enabled() or not self._model.has_template():
            logger.debug(
                "Skipping center template match: enabled=%s has_template=%s",
                self._model.enabled(),
                self._model.has_template(),
            )
            return None

        template = self._loaded_template
        if template is None:
            template_path = self._model.template_path()
            logger.debug(
                "Loading center template for match from disk: %s", template_path
            )
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is not None:
                self._loaded_template = template

        if template is None:
            msg = f"Template not found: {self._model.template_path()}"
            logger.warning(msg)
            self.match_failed.emit(msg)
            return None

        img_h, img_w = image_bgr.shape[:2]
        tpl_h, tpl_w = template.shape[:2]
        if tpl_w > img_w or tpl_h > img_h:
            msg = "Template is larger than the loaded image."
            logger.warning(
                "%s template_size=%dx%d image_size=%dx%d",
                msg,
                tpl_w,
                tpl_h,
                img_w,
                img_h,
            )
            self.match_failed.emit(msg)
            return None
        anchor_x, anchor_y = self._model.anchor()
        logger.debug(
            "Running center template match: image_size=%dx%d template_size=%dx%d "
            "anchor=(%d, %d)",
            img_w,
            img_h,
            tpl_w,
            tpl_h,
            anchor_x,
            anchor_y,
        )
        cx, cy, score = locate_center(image_bgr, template, anchor_x, anchor_y)
        logger.info(
            "Center template matched: center=(%.1f, %.1f) score=%.4f",
            cx,
            cy,
            score,
        )
        if image_path:
            self._match_cache[image_path] = (float(cx), float(cy), float(score))
        self._model.set_match(cx, cy, score)
        self.match_ready.emit(float(cx), float(cy), float(score))
        return float(cx), float(cy), float(score)

    def start_batch_preload(self, file_paths: list) -> None:
        if not self._model.enabled() or not self._model.has_template():
            return
        template = self._loaded_template
        if template is None:
            template_path = self._model.template_path()
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                logger.warning(
                    "start_batch_preload: template not found at %s", template_path
                )
                return
            self._loaded_template = template
        if not file_paths:
            return
        self._stop_worker()
        anchor_x, anchor_y = self._model.anchor()
        self._worker = CenterCropWorker(template, anchor_x, anchor_y, file_paths)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        logger.info("CenterCropWorker started for %d paths", len(file_paths))

    def shutdown(self) -> None:
        self._stop_worker()

    def _stop_worker(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.blockSignals(True)
            self._worker.stop()
            self._worker.wait()
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_worker_finished(self) -> None:
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_worker_result(self, path: str, cx: float, cy: float, score: float) -> None:
        self._match_cache[path] = (cx, cy, score)
        self.preload_result.emit(path, cx, cy, score)

    def is_cached(self, path: str) -> bool:
        return path in self._match_cache
