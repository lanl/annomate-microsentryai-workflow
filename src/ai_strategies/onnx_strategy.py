"""
ONNX Runtime strategy for anomaly detection models.

Loads .onnx models exported from anomalib (or any model with
'anomaly_map' and 'pred_score' output nodes) and runs inference
using onnxruntime — no PyTorch required.
"""

import logging
import platform
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

from ai_strategies.interface import AnomalyDetectionStrategy

logger = logging.getLogger("MicroSentryAI.OnnxStrategy")

_DEFAULT_INPUT_SIZE = 256


class OnnxStrategy(AnomalyDetectionStrategy):
    """Anomaly detection strategy backed by ONNX Runtime.

    Implements the same interface as AnomalibStrategy so it is a drop-in
    replacement for .onnx model files. Supports CUDA via
    CUDAExecutionProvider; falls back to CPU on any platform including macOS
    (onnxruntime has no MPS provider).

    Attributes:
        device (str): Requested device — ``"auto"``, ``"cpu"``, or ``"cuda"``.
        model_name (str): Human-readable label set after a successful load.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device: str = "auto"
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._input_hw: Tuple[int, int] = (_DEFAULT_INPUT_SIZE, _DEFAULT_INPUT_SIZE)

    def set_device(self, device_code: str) -> None:
        """Set the target compute device.

        Args:
            device_code: ``"auto"``, ``"cpu"``, or ``"cuda"``. ``"mps"`` is
                silently treated as ``"cpu"`` — onnxruntime has no MPS provider.
        """
        self.device = device_code.lower()
        if self.device == "mps":
            logger.info("MPS requested but onnxruntime has no MPS provider — using CPU")
            self.device = "cpu"

    def _resolve_providers(self) -> list:
        available = ort.get_available_providers()
        providers = []
        if self.device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _detected_device_label(self, active_providers: list) -> str:
        """Build a device label from the providers onnxruntime actually activated."""
        if "CUDAExecutionProvider" in active_providers:
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    timeout=3,
                    stderr=subprocess.DEVNULL,
                ).decode().strip().splitlines()
                gpu = out[0] if out else "GPU"
                return f"CUDA ({gpu})"
            except Exception:
                return "CUDA"
        return f"CPU ({platform.processor() or platform.machine()})"

    def load_from_file(self, model_path: str) -> None:
        """Load an .onnx model file.

        Args:
            model_path: Absolute path to a .onnx file.

        Raises:
            RuntimeError: If the file extension is wrong or the session fails
                to initialise.
        """
        path = Path(model_path)
        if path.suffix.lower() != ".onnx":
            raise RuntimeError(
                f"OnnxStrategy requires a .onnx file, got: {path.suffix!r}"
            )

        self._session = None
        providers = self._resolve_providers()
        logger.info("Loading ONNX model with providers: %s", providers)

        try:
            self._session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise RuntimeError(f"ONNX session creation failed: {exc}") from exc

        # Use the providers that actually activated (CUDA may have fallen back to CPU)
        active_providers = self._session.get_providers()

        # Read input metadata
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no input nodes.")
        self._input_name = inputs[0].name
        shape = inputs[0].shape  # e.g. [None, 3, None, None] or [1, 3, 256, 256]
        if (
            len(shape) == 4
            and isinstance(shape[2], int) and shape[2] > 0
            and isinstance(shape[3], int) and shape[3] > 0
        ):
            self._input_hw = (shape[2], shape[3])
        else:
            self._input_hw = (_DEFAULT_INPUT_SIZE, _DEFAULT_INPUT_SIZE)

        # Validate output names
        output_names = [o.name for o in self._session.get_outputs()]
        if "anomaly_map" not in output_names or "pred_score" not in output_names:
            raise RuntimeError(
                f"Expected ONNX outputs 'anomaly_map' and 'pred_score' "
                f"but found: {output_names}. "
                f"Export the model with anomalib's ONNX export."
            )

        device_label = self._detected_device_label(active_providers)
        self.model_name = f"ONNX [{device_label}]"
        logger.info("Loaded %s — input=%r shape=%s", self.model_name, self._input_name, shape)

    def load_from_folder(self, folder_path: str) -> None:
        raise NotImplementedError("Use load_from_file() for ONNX models.")

    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Run inference on a single image.

        Args:
            image_path: Absolute path to the input image.

        Returns:
            (anomaly_score, heatmap) where heatmap is a 2-D float32 array
            with values in [0, 1]. The heatmap is NOT re-normalized — the
            anomalib PostProcessor already calibrates it to [0, 1] with 0.5
            as the decision boundary, exactly as the torch path does.
        """
        if self._session is None:
            return 0.0, np.zeros(self._input_hw, dtype=np.float32)

        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Could not read image: %s", image_path)
                return 0.0, np.zeros(self._input_hw, dtype=np.float32)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = self._input_hw
            img = cv2.resize(img, (w, h))

            tensor = img.astype(np.float32) / 255.0          # [H, W, 3]
            tensor = tensor.transpose(2, 0, 1)[np.newaxis]   # [1, 3, H, W]

            outputs = self._session.run(None, {self._input_name: tensor})
            out_map = {
                o.name: arr
                for o, arr in zip(self._session.get_outputs(), outputs)
            }

            # anomaly_map: [batch, 1, H, W] or [batch, H, W] → 2-D
            heatmap = out_map["anomaly_map"].squeeze().astype(np.float32)

            # pred_score: [batch, 1] or [batch] → scalar
            score = float(out_map["pred_score"].flat[0])

            return score, heatmap

        except Exception as exc:
            logger.error("ONNX inference failed: %s", exc)
            return 0.0, np.zeros(self._input_hw, dtype=np.float32)
