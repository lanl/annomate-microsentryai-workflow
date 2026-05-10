"""
SAMController — Qt infrastructure for SAM 2 inference.

Two QThreads:
  _ModelLoadWorker  — downloads / initialises the SAM checkpoint off the
                      main thread (can take 30+ s on first run).
  SAMWorker         — runs a single predict_bbox() call off the main thread.

SAMController      — QObject that owns the strategy and both workers, and
                      re-emits results as typed signals for AnnoMateWindow.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ai_strategies.sam_strategy import SAMStrategy, weights_cached

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Workers
# ------------------------------------------------------------------ #


class _ModelLoadWorker(QThread):
    """Loads (and optionally downloads) a SAM checkpoint off the main thread."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, strategy: SAMStrategy, parent: QObject = None) -> None:
        super().__init__(parent)
        self._strategy = strategy

    def run(self) -> None:
        try:
            self._strategy.load()
            logger.info("SAM model loaded successfully")
            self.done.emit()
        except Exception as exc:
            logger.error("SAM model load failed: %s", exc)
            self.failed.emit(str(exc))


class SAMWorker(QThread):
    """Runs a single bounding-box SAM prediction off the main thread."""

    resultReady = Signal(list, float)  # (polygon_pts, confidence)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        strategy: SAMStrategy,
        image_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        epsilon: float,
        parent: QObject = None,
    ) -> None:
        super().__init__(parent)
        self._strategy = strategy
        self._image_bgr = image_bgr
        self._bbox = bbox
        self._epsilon = epsilon

    def run(self) -> None:
        try:
            pts, conf = self._strategy.predict_bbox(
                self._image_bgr, self._bbox, self._epsilon
            )
            self.resultReady.emit(pts, conf)
        except Exception as exc:
            logger.warning("SAM inference error: %s", exc)
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


# ------------------------------------------------------------------ #
# Controller
# ------------------------------------------------------------------ #


class SAMController(QObject):
    """Manages SAMStrategy lifecycle and exposes async signals to the view.

    Signals:
        result_ready (list, float): polygon pts in original coords + confidence.
        inference_failed (str): Error string from a failed inference.
        loading_done (): Model loaded and ready.
        loading_failed (str): Error string from a failed model load.
    """

    result_ready = Signal(list, float)
    inference_failed = Signal(str)
    loading_done = Signal()
    loading_failed = Signal(str)

    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        self._strategy: Optional[SAMStrategy] = None
        self._load_worker: Optional[_ModelLoadWorker] = None
        self._infer_worker: Optional[SAMWorker] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_variant(self, variant: str) -> None:
        """Configure (or reconfigure) the SAM variant to use."""
        if self._strategy is None:
            self._strategy = SAMStrategy(variant)
        else:
            self._strategy.set_variant(variant)

    def try_autoload(self, variant: str) -> bool:
        """Start a background load if the checkpoint is already on disk.

        Returns True if loading was started, False if the weights are absent.
        """
        if not weights_cached(variant):
            logger.info("SAM autoload skipped — %s not found on disk", variant)
            return False
        logger.info("SAM weights found on disk (%s) — starting autoload", variant)
        self.set_variant(variant)
        self.ensure_loaded_async()
        return True

    def ensure_loaded_async(self) -> None:
        """Trigger model load on a background thread.

        Emits loading_done immediately if already loaded.
        """
        if self._strategy is None:
            self._strategy = SAMStrategy()

        if self._strategy.is_loaded:
            logger.info("SAM model already loaded, skipping reload")
            self.loading_done.emit()
            return

        logger.info("SAM model load starting in background thread")
        self._stop_load_worker()
        self._load_worker = _ModelLoadWorker(self._strategy, parent=self)
        self._load_worker.done.connect(self.loading_done)
        self._load_worker.done.connect(self._on_load_worker_finished)
        self._load_worker.failed.connect(self.loading_failed)
        self._load_worker.failed.connect(self._on_load_worker_finished)
        self._load_worker.start()

    def run_inference(
        self,
        image_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        epsilon: float = 2.0,
    ) -> None:
        """Start a background inference for *bbox* on *image_bgr*."""
        if self._strategy is None or not self._strategy.is_loaded:
            self.inference_failed.emit("Model not loaded.")
            return

        self._stop_infer_worker()
        self._infer_worker = SAMWorker(
            self._strategy, image_bgr, bbox, epsilon, parent=self
        )
        self._infer_worker.resultReady.connect(self.result_ready)
        self._infer_worker.failed.connect(self.inference_failed)
        self._infer_worker.finished.connect(self._on_infer_worker_finished)
        self._infer_worker.start()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _on_load_worker_finished(self) -> None:
        """Clear the Python reference after the load worker completes."""
        if self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None

    def _on_infer_worker_finished(self) -> None:
        """Clear the Python reference after an inference worker completes."""
        if self._infer_worker is not None:
            self._infer_worker.deleteLater()
            self._infer_worker = None

    def _stop_load_worker(self) -> None:
        if self._load_worker is not None:
            self._load_worker.blockSignals(True)
            self._load_worker.quit()
            self._load_worker.wait()
            self._load_worker = None

    def _stop_infer_worker(self) -> None:
        if self._infer_worker is not None:
            self._infer_worker.blockSignals(True)
            self._infer_worker.quit()
            self._infer_worker.wait()
            self._infer_worker = None
