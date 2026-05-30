"""
InferenceController — headless inference orchestration for MicroSentryAI.

Rules:
  - No Qt GUI types (no QFileDialog, QMessageBox, QColor).
  - All methods accept plain Python values; errors raised as exceptions.
  - QThread is permitted here — it is infrastructure, not UI.
"""

import logging
from typing import Optional, List, Tuple, Type

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib import colormaps as mpl_cmaps

from PySide6.QtCore import QObject, QThread, Signal

from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel

logger = logging.getLogger("MicroSentryAI.InferenceController")


class InferenceWorker(QThread):
    """Background thread that runs batch inference without blocking the UI.

    Iterates over a list of file paths, calls the strategy's ``predict``
    method for each, and emits the resulting score map. Supports cooperative
    cancellation via :meth:`stop`.

    Attributes:
        strategy: Inference strategy object exposing a ``predict(path: str)``
            method that returns ``(_, score_map: np.ndarray)``.
        file_list (List[str]): Ordered list of absolute image paths to process.

    Signals:
        resultReady (str, float, object): Emitted after each successful prediction
            as ``(absolute_path, score, score_map)``.
        progress (int): Emitted after each image with the count processed so far.
        finished (): Emitted once the loop exits, regardless of outcome.
    """

    resultReady = Signal(
        str, float, object
    )  # (absolute_path, score: float, score_map: np.ndarray)
    progress = Signal(int)  # count of images processed so far

    def __init__(self, strategy, file_list: List[str]) -> None:
        """Initialize InferenceWorker with a strategy and file list.

        Args:
            strategy: Inference strategy exposing a ``predict(path: str)``
                method.
            file_list (List[str]): Ordered list of absolute image paths to
                process in batch.
        """
        super().__init__()
        self.strategy = strategy
        self.file_list = file_list
        self._running = True

    def run(self) -> None:
        """Execute inference for every path in *file_list*.

        Iterates until all paths are processed or :meth:`stop` sets
        ``_running`` to ``False``. Errors for individual images are logged
        but do not abort the remaining batch. Always emits :attr:`finished`
        on exit.
        """
        for i, path in enumerate(self.file_list):
            if not self._running:
                break
            try:
                score, score_map = self.strategy.predict(path)
                self.resultReady.emit(path, score, score_map)
            except Exception as e:
                logger.error("Inference failed for %s: %s", path, e)
            self.progress.emit(i + 1)

    def stop(self) -> None:
        """Request cooperative cancellation of the inference loop.

        Sets an internal flag that causes :meth:`run` to exit after the
        current image completes. Does not forcefully terminate the thread.
        """
        self._running = False


class InferenceController(QObject):
    """Own the InferenceWorker lifecycle and expose proxy signals for the View.

    Manages model loading, image I/O, and worker lifecycle so the View never
    holds a direct reference to the underlying thread. All visualisation
    helpers are pure Python and contain zero Qt GUI dependencies.

    Attributes:
        dataset_model (DatasetTableModel): Model supplying image file metadata.
        inference_model (InferenceModel): Model storing per-image score maps.

    Signals:
        result_ready (str, float, object): Proxy for :attr:`InferenceWorker.resultReady`
            — emitted as ``(absolute_path, score, score_map)``.
        progress (int): Proxy for :attr:`InferenceWorker.progress` — count of
            images processed so far.
        batch_done (): Emitted when the current batch finishes.
    """

    result_ready = Signal(
        str, float, object
    )  # (path, score: float, score_map: np.ndarray)
    progress = Signal(int)  # images processed so far
    batch_done = Signal()  # all images in a batch finished

    def __init__(
        self,
        dataset_model: DatasetTableModel,
        inference_model: InferenceModel,
        strategy_class: Optional[Type] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize InferenceController.

        Args:
            dataset_model (DatasetTableModel): Model providing image metadata.
            inference_model (InferenceModel): Model for storing score maps.
            strategy_class (Optional[Type]): Inference strategy class to
                instantiate on :meth:`load_model`. Defaults to
                ``AnomalibStrategy`` when ``None``.
            parent (Optional[QObject]): Qt parent object. Defaults to ``None``.
        """
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self._strategy_class = strategy_class
        self._strategy = None
        self._worker: Optional[InferenceWorker] = None
        self._model_path: str = ""

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def load_model(self, model_path: str, device: str = "auto") -> str:
        """Load a ``.pt`` or ``.ckpt`` model file and prepare the strategy.

        Args:
            model_path (str): Absolute path to the model checkpoint file.
            device (str): Target device — ``"auto"`` (default) detects CUDA →
                MPS → CPU in that order; or pass ``"cpu"``/``"cuda"``/``"mps"``.

        Returns:
            str: The loaded model's name as reported by the strategy.

        Raises:
            RuntimeError: If the strategy fails to load the model file.
        """
        if self._strategy_class is None:
            from ai_strategies.anomalib_strategy import AnomalibStrategy

            strategy_class = AnomalibStrategy
        else:
            strategy_class = self._strategy_class
        strategy = strategy_class()
        strategy.set_device(device.lower())
        strategy.load_from_file(model_path)  # raises on failure
        self._strategy = strategy
        self._model_path = model_path
        logger.info("Model loaded: %s on %s", strategy.model_name, model_path)
        return strategy.model_name

    def get_model_name(self) -> str:
        """Return the backend/device label of the loaded model, or empty string."""
        return self._strategy.model_name if self._strategy else ""

    def get_model_path(self) -> str:
        """Return the file path of the currently loaded model, or empty string."""
        return self._model_path

    def has_model(self) -> bool:
        """Check whether a model has been successfully loaded.

        Returns:
            bool: ``True`` if a strategy with a loaded model is available,
                ``False`` otherwise.
        """
        return self._strategy is not None

    # ------------------------------------------------------------------ #
    # Image loading
    # ------------------------------------------------------------------ #

    def load_image(self, path: str) -> Optional[Image.Image]:
        """Load an image from disk as an RGB PIL Image.

        Args:
            path (str): Absolute path to the image file.

        Returns:
            Optional[Image.Image]: RGB PIL Image, or ``None`` if the file
                cannot be read.
        """
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning("Could not read image %s: %s", path, e)
            return None

    # ------------------------------------------------------------------ #
    # Background inference
    # ------------------------------------------------------------------ #

    def start_batch_inference(self, file_paths: List[str]) -> None:
        """Stop any running worker and start a new batch inference job.

        Connects the new worker's signals to this controller's proxy signals
        before starting. Callers should connect to :attr:`result_ready`,
        :attr:`progress`, and :attr:`batch_done` once (e.g. in the view's
        ``__init__``) rather than per call.

        Args:
            file_paths (List[str]): Ordered list of absolute image paths to
                process in this batch.
        """
        self._stop_worker()
        self._worker = InferenceWorker(self._strategy, file_paths)
        self._worker.resultReady.connect(self.result_ready)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self.batch_done)
        self._worker.start()

    def shutdown(self) -> None:
        self._stop_worker()

    def _stop_worker(self) -> None:
        """Gracefully stop and clean up the current inference worker.

        Blocks signals on the worker before stopping to prevent stale
        ``resultReady`` emissions from reaching the proxy signals after
        cancellation.
        """
        if self._worker and self._worker.isRunning():
            self._worker.blockSignals(
                True
            )  # prevent stale signals from reaching proxies
            self._worker.stop()
            self._worker.wait()
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------ #
    # Visualisation computation (pure Python, no Qt)
    # ------------------------------------------------------------------ #

    def compute_heatmap(
        self,
        pil_image: Image.Image,
        score_map: np.ndarray,
        alpha: float,
        sigma: float,
        display_target: int,
        heat_min_pct: int,
    ) -> Tuple[Image.Image, Image.Image, float, tuple, Optional[np.ndarray]]:
        """Compute a resized raw image and a heatmap overlay composited on it.

        Applies optional Gaussian smoothing to *score_map*, percentile-clips
        and normalises it, maps it through the ``"jet"`` colormap, and
        alpha-composites the result onto the resized source image. The
        smoothed score array is returned so :meth:`compute_segmentation` can
        reuse it without re-running the Gaussian.

        Args:
            pil_image (Image.Image): Source image in any PIL mode.
            score_map (np.ndarray): 2-D anomaly score array aligned with
                *pil_image*. Pass ``None`` to skip heatmap computation.
            alpha (float): Heatmap overlay opacity in the range ``[0.0, 1.0]``.
            sigma (float): Standard deviation for Gaussian smoothing applied
                to *score_map* before normalisation. ``0`` disables smoothing.
            display_target (int): Target size in pixels for the longer edge of
                the output images.
            heat_min_pct (int): Percentile (0–100) used to clip the lower tail
                of the score distribution before normalisation.

        Returns:
            Tuple[Image.Image, Image.Image, float, tuple, Optional[np.ndarray]]:
            A five-element tuple:

            - ``left_image``: Raw source resized to *display_target*.
            - ``right_image``: Heatmap composited on the resized source.
            - ``scale``: Uniform scale factor applied (display / original).
            - ``offset``: ``(off_x, off_y)`` crop offset; always ``(0, 0)``.
            - ``smoothed_s``: Smoothed score map ndarray for reuse in
              :meth:`compute_segmentation`, or ``None`` if *score_map* is
              ``None``.
        """
        w, h = pil_image.size
        scale = display_target / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        left_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        offset = (0, 0)

        if score_map is None:
            return left_image, left_image.copy(), scale, offset, None

        s = gaussian_filter(score_map, sigma=sigma) if sigma > 0 else score_map.copy()

        # Suppress the background noise floor using the percentile slider, then
        # keep the absolute [0, 1] scale intact. Re-normalizing per image (old
        # behaviour) collapsed the distinction between NORMAL and ANOMALY images:
        # both would fill the full colormap range regardless of actual severity.
        v_min_thr = np.percentile(s, heat_min_pct)
        s_norm = np.clip(s, v_min_thr, 1.0)

        s_resized = cv2.resize(s_norm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        heat_arr = (s_resized * 255).astype(np.uint8)
        colored = (mpl_cmaps["jet"](heat_arr / 255.0) * 255).astype(np.uint8)

        overlay_pil = Image.fromarray(colored, mode="RGBA")
        r, g, b, a_ch = overlay_pil.split()
        a_ch = a_ch.point(lambda p: int(p * alpha))
        overlay_pil = Image.merge("RGBA", (r, g, b, a_ch))

        comp = left_image.convert("RGBA")
        right_image = Image.alpha_composite(comp, overlay_pil).convert("RGB")

        return left_image, right_image, scale, offset, s

    def compute_segmentation(
        self,
        smoothed_s: Optional[np.ndarray],
        seg_pct: float,
        epsilon: float,
        display_w: int,
        display_h: int,
    ) -> list:
        """Compute polygon contours from a smoothed score map at a percentile threshold.

        Thresholds *smoothed_s* at the *seg_pct* percentile, resizes the
        resulting binary mask to display dimensions, applies one round of
        erosion to reduce noise, then extracts simplified external contours
        via Douglas-Peucker approximation.

        Args:
            smoothed_s (Optional[np.ndarray]): Smoothed 2-D score map as
                returned by :meth:`compute_heatmap`. Returns ``[]``
                immediately when ``None``.
            seg_pct (int): Percentile threshold (0–100) applied to
                *smoothed_s* to produce the binary segmentation mask.
            epsilon (float): Douglas-Peucker approximation tolerance in
                pixels. Larger values produce fewer vertices per contour.
            display_w (int): Target width in pixels for the resized mask.
            display_h (int): Target height in pixels for the resized mask.

        Returns:
            list: List of contours, where each contour is a list of
                ``(x, y)`` float tuples in display coordinates. Contours
                with fewer than three points are discarded.
        """
        if smoothed_s is None:
            return []

        seg_thr = np.percentile(smoothed_s, seg_pct)
        mask = (smoothed_s > seg_thr).astype(np.uint8) * 255
        mask = cv2.resize(mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        raw_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = []
        for cnt in raw_contours:
            if len(cnt) < 3:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
            pts = [
                (float(pt[0][0]), float(pt[0][1]))
                for pt in (approx if approx is not None else cnt)
            ]
            if len(pts) >= 3:
                contours.append(pts)

        return contours

    def compute_visualization(
        self,
        pil_image: Image.Image,
        score_map: np.ndarray,
        alpha: float,
        sigma: float,
        display_target: int,
        heat_min_pct: int,
        seg_pct: int,
        epsilon: float,
    ) -> Tuple[Image.Image, Image.Image, list, float, tuple]:
        """Compute heatmap and segmentation contours in a single call.

        Compatibility shim that delegates to :meth:`compute_heatmap` followed
        by :meth:`compute_segmentation` and returns their combined outputs.

        Args:
            pil_image (Image.Image): Source image in any PIL mode.
            score_map (np.ndarray): 2-D anomaly score array aligned with
                *pil_image*.
            alpha (float): Heatmap overlay opacity in the range ``[0.0, 1.0]``.
            sigma (float): Standard deviation for Gaussian smoothing. ``0``
                disables smoothing.
            display_target (int): Target size in pixels for the longer edge.
            heat_min_pct (int): Percentile used to clip the lower score tail.
            seg_pct (int): Percentile threshold for the segmentation mask.
            epsilon (float): Douglas-Peucker approximation tolerance in pixels.

        Returns:
            Tuple[Image.Image, Image.Image, list, float, tuple]: A five-element
            tuple of ``(left_image, right_image, contours, scale, offset)``
            as described in :meth:`compute_heatmap` and
            :meth:`compute_segmentation`.
        """
        left_pil, right_pil, scale, offset, s = self.compute_heatmap(
            pil_image, score_map, alpha, sigma, display_target, heat_min_pct
        )
        w, h = left_pil.size
        contours = self.compute_segmentation(s, seg_pct, epsilon, w, h)
        return left_pil, right_pil, contours, scale, offset
