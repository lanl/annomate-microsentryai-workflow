"""
ONNX Anomaly Strategy for MicroSentryAI.

Runs anomaly detection models exported to ONNX from Anomalib
(``Engine.export(..., export_type=ExportType.ONNX)``). The export bakes
resize, normalisation, and PostProcessor score calibration into the graph,
so this strategy only scales pixels to 0–1 and reads the calibrated outputs.
No Qt dependencies.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ai_strategies.interface import AnomalyDetectionStrategy
from ai_strategies.ort_backend import create_session, log_ort_environment

logger = logging.getLogger("MicroSentryAI.OnnxAnomalyStrategy")


class OnnxAnomalyStrategy(AnomalyDetectionStrategy):
    """Strategy for ONNX (``.onnx``) anomaly detection models.

    Expects models exported by Anomalib 2.x, whose graph includes the
    pre-processing (resize + normalisation) and PostProcessor calibration.
    The model must expose ``pred_score`` and ``anomaly_map`` outputs;
    ``pred_label``/``pred_mask`` outputs are ignored when present.

    Attributes:
        session (Optional[onnxruntime.InferenceSession]): Active inference
            session, or ``None`` before :meth:`load_from_file`.
        device (str): Requested compute device — ``"auto"``, ``"cpu"``,
            ``"cuda"``, or ``"mps"`` (mapped to execution providers).
        model_type (str): Short tag set to ``"onnx"`` after a successful load.
        model_name (str): Human-readable label including the backend and
            provider; updated by :meth:`load_from_file`.
    """

    def __init__(self) -> None:
        """Initialize the strategy with empty model state and ``"auto"`` device."""
        super().__init__()
        self.session = None
        self.device = "auto"
        self.model_type = "unknown"
        self._input_name: str = "input"
        self._input_hw: Optional[Tuple[int, int]] = None

    def set_device(self, device_code: str) -> None:
        """Set the target compute device for inference.

        Args:
            device_code (str): Device identifier — one of ``"auto"``,
                ``"cpu"``, ``"cuda"``, or ``"mps"``. The value is
                lower-cased before storage.
        """
        self.device = device_code.lower()
        logger.info("Target device set to: %s", self.device)

    def load_from_folder(self, folder_path: str) -> None:
        """Not supported — ONNX strategies require a single model file.

        Args:
            folder_path (str): Unused directory path.

        Raises:
            NotImplementedError: Always; use :meth:`load_from_file` instead.
        """
        raise NotImplementedError("Use load_from_file() for ONNX models.")

    def load_from_file(self, model_path: str) -> None:
        """Load a ``.onnx`` model file and validate its inference signature.

        Args:
            model_path (str): Absolute path to the model file. Must have a
                ``.onnx`` extension.

        Raises:
            RuntimeError: If the file is a PyTorch checkpoint, has an
                unsupported extension, lacks the required outputs, or the
                session cannot be created.
        """
        path = Path(model_path)
        self.session = None

        t_total = time.perf_counter()

        try:
            if path.suffix.lower() in (".pt", ".pth", ".ckpt"):
                raise ValueError(
                    "PyTorch checkpoints (.pt/.ckpt) are no longer supported. "
                    "Export the model to ONNX from your training pipeline "
                    "(anomalib: Engine.export(..., export_type=ExportType.ONNX)) "
                    "and load the resulting model.onnx instead."
                )
            if path.suffix.lower() != ".onnx":
                raise ValueError(
                    f"Unsupported file type: {path.suffix}. Expected .onnx"
                )

            log_ort_environment()
            session, provider_label = create_session(str(path), self.device)

            model_input = session.get_inputs()[0]
            self._input_name = model_input.name
            # Anomalib exports use dynamic H/W with resize in-graph; a static
            # export needs the image resized to the declared dims first.
            height, width = model_input.shape[2], model_input.shape[3]
            if isinstance(height, int) and isinstance(width, int):
                self._input_hw = (height, width)
            else:
                self._input_hw = None

            output_names = {out.name for out in session.get_outputs()}
            missing = {"pred_score", "anomaly_map"} - output_names
            if missing:
                raise ValueError(
                    f"Model is missing required outputs {sorted(missing)} "
                    f"(found: {sorted(output_names)}). Expected an Anomalib "
                    "ONNX export."
                )

            self.session = session
            self.model_type = "onnx"
            self.model_name = f"Anomalib (ONNX) [{provider_label}]"
            logger.info(
                "Loaded %s — %.2fs", self.model_name, time.perf_counter() - t_total
            )

        except Exception as e:
            logger.error("Critical failure loading model: %s", e)
            raise RuntimeError(f"Load Error: {e}")

    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Run inference on a single image.

        Reads the image with OpenCV, converts to RGB, scales pixels to 0–1
        (resize and normalisation are inside the exported graph), and reads
        the calibrated ``pred_score`` and ``anomaly_map`` outputs. Returns a
        zero score and a blank heatmap when no model is loaded or on error.

        Args:
            image_path (str): Absolute path to the input image file.

        Returns:
            Tuple[float, np.ndarray]: ``(anomaly_score, heatmap)`` where
                *heatmap* is a 2-D ``float32`` array normalised to 0–1.
        """
        if self.session is None:
            return 0.0, np.zeros((256, 256), dtype=np.float32)

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self._input_hw is not None:
                img = cv2.resize(img, (self._input_hw[1], self._input_hw[0]))
            tensor = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

            pred_score, anomaly_map = self.session.run(
                ["pred_score", "anomaly_map"], {self._input_name: tensor}
            )

            score = float(np.asarray(pred_score).ravel()[0])
            heatmap = np.asarray(anomaly_map).squeeze()

            # Do NOT locally re-normalize. The PostProcessor baked into the
            # export has already mapped the anomaly map to [0, 1] with 0.5 as
            # the calibrated decision boundary. Re-normalizing per image would
            # make a good image (values 0.35–0.49) look identical to a bad one
            # (0.5–1.0) in the heatmap display.
            return score, heatmap.astype(np.float32)

        except Exception as e:
            logger.error("ONNX inference failed: %s", e)
            return 0.0, np.zeros((256, 256), dtype=np.float32)
