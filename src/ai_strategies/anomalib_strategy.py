"""
Anomalib Strategy for MicroSentryAI.

Primary path: Anomalib TorchInferencer.
Fallback: raw torch.load with DynamicUnpickler to handle missing checkpoint classes.
No Qt dependencies.
"""

import os
import io
import time
import pickle
import pathlib
import logging
import platform
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from anomalib.deploy import TorchInferencer
import cv2

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger("MicroSentryAI.AnomalibStrategy")


# ---------------------------------------------------------------------------
# Dynamic Unpickler — bypasses missing classes in torch.load
# ---------------------------------------------------------------------------


class DummyMeta(type):
    """Metaclass that returns :class:`DummyClass` for any unknown class attribute.

    Used so that :class:`DummyClass` instances silently absorb any attribute
    access that would normally raise :exc:`AttributeError` on a real class.
    """

    def __getattr__(cls, name: str) -> type:
        """Return :class:`DummyClass` for any attribute looked up on the class.

        Args:
            name (str): Name of the attribute being accessed.

        Returns:
            type: Always returns :class:`DummyClass`.
        """
        return DummyClass


class DummyClass(metaclass=DummyMeta):
    """Catch-all stub that silently absorbs any interaction.

    Used by :class:`DynamicUnpickler` as a replacement for classes that are
    missing at unpickling time. Every operation — construction, calling,
    attribute access, item access, and ``__setstate__`` — is a no-op or
    returns another :class:`DummyClass` instance so deserialization
    continues without raising exceptions.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Accept and discard any positional or keyword arguments."""

    def __call__(self, *args, **kwargs) -> "DummyClass":
        """Return a new :class:`DummyClass` instance when called as a function.

        Returns:
            DummyClass: A fresh no-op stub.
        """
        return DummyClass()

    def __getattr__(self, name: str) -> "DummyClass":
        """Return a :class:`DummyClass` instance for any attribute access.

        Args:
            name (str): Name of the attribute being accessed.

        Returns:
            DummyClass: A fresh no-op stub.
        """
        return DummyClass()

    def __getitem__(self, key) -> "DummyClass":
        """Return a :class:`DummyClass` instance for any item lookup.

        Args:
            key: The key being accessed (ignored).

        Returns:
            DummyClass: A fresh no-op stub.
        """
        return DummyClass()

    def __setitem__(self, key, value) -> None:
        """Silently ignore item assignment.

        Args:
            key: The key being assigned (ignored).
            value: The value being assigned (ignored).
        """

    def __setstate__(self, state) -> None:
        """Silently ignore ``__setstate__`` calls during unpickling.

        Args:
            state: The pickle state dict (ignored).
        """


class DynamicUnpickler(pickle.Unpickler):
    """Pickle unpickler that replaces missing classes with :class:`DummyClass`.

    Handles the common cross-platform mismatch where a checkpoint saved on
    POSIX contains ``pathlib.PosixPath`` objects that cannot be unpickled on
    Windows (and vice-versa). Any other import or attribute error during
    class lookup is caught and replaced with a :class:`DummyClass` stub so
    that ``torch.load`` can proceed despite missing checkpoint dependencies.
    """

    def find_class(self, module: str, name: str) -> type:
        """Resolve a pickled class reference, substituting stubs for missing classes.

        Applies two fixes before the standard lookup:

        1. ``pathlib.PosixPath`` → ``pathlib.WindowsPath`` on Windows.
        2. ``pathlib.WindowsPath`` → ``pathlib.PosixPath`` on non-Windows.

        Falls back to :class:`DummyClass` for any class that raises
        :exc:`ImportError`, :exc:`AttributeError`, or
        :exc:`ModuleNotFoundError`.

        Args:
            module (str): Dotted module path of the class to look up.
            name (str): Class name within *module*.

        Returns:
            type: The resolved class, or :class:`DummyClass` if the class
                cannot be imported.
        """
        if module == "pathlib":
            if name == "PosixPath" and platform.system() == "Windows":
                return pathlib.WindowsPath
            if name == "WindowsPath" and platform.system() != "Windows":
                return pathlib.PosixPath
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            logger.warning("Mocking missing checkpoint class: %s.%s", module, name)
            return DummyClass


class DynamicPickleModule:
    """Drop-in replacement for the ``pickle`` module accepted by ``torch.load``.

    Passed as the ``pickle_module`` argument to ``torch.load`` so that
    :class:`DynamicUnpickler` is used instead of the standard unpickler,
    enabling deserialization of checkpoints with missing classes.

    Attributes:
        Unpickler (type): Points to :class:`DynamicUnpickler`.
    """

    Unpickler = DynamicUnpickler

    @staticmethod
    def load(file, **kwargs) -> object:
        """Deserialize an object from an open binary file using :class:`DynamicUnpickler`.

        Args:
            file: Readable binary file-like object containing pickled data.
            **kwargs: Ignored; present to match the ``pickle.load`` signature.

        Returns:
            object: The deserialized Python object.
        """
        return DynamicUnpickler(file).load()

    @staticmethod
    def loads(b: bytes, **kwargs) -> object:
        """Deserialize an object from a bytes buffer using :class:`DynamicUnpickler`.

        Args:
            b (bytes): Bytes buffer containing pickled data.
            **kwargs: Ignored; present to match the ``pickle.loads`` signature.

        Returns:
            object: The deserialized Python object.
        """
        return DynamicUnpickler(io.BytesIO(b)).load()


# ---------------------------------------------------------------------------


class AnomalibStrategy:
    """Strategy for PyTorch (``.pt``, ``.ckpt``) anomaly detection models.

    Implements a two-attempt loading pipeline:

    1. **Anomalib TorchInferencer** — the preferred path; handles model
       metadata, device placement, and structured output automatically.
    2. **Raw ``torch.load`` fallback** — used when the Anomalib path fails
       (e.g. missing checkpoint classes); :class:`DynamicPickleModule` is
       passed as the ``pickle_module`` to suppress import errors.

    Attributes:
        torch_inferencer (Optional[TorchInferencer]): Active Anomalib
            inferencer, or ``None`` if the fallback path was used.
        raw_model: Raw PyTorch model loaded via ``torch.load``, or ``None``
            if the Anomalib path succeeded.
        device (str): Requested compute device — ``"auto"``, ``"cpu"``,
            ``"cuda"``, or ``"mps"``.
        model_type (str): Short tag set to ``"torch"`` after a successful
            load.
        model_name (str): Human-readable label including the backend and
            device; updated by :meth:`load_from_file`.
        _device_verified (bool): Set to ``True`` once the tensor device has
            been confirmed on the first inference call.
    """

    def __init__(self) -> None:
        """Initialize the strategy with empty model state and ``"auto"`` device."""
        self.torch_inferencer: Optional[TorchInferencer] = None
        self.raw_model = None
        self.device = "auto"
        self.model_type = "unknown"
        self.model_name = "Unknown"
        self._device_verified = False

    def set_device(self, device_code: str) -> None:
        """Set the target compute device for inference.

        Args:
            device_code (str): Device identifier — one of ``"auto"``,
                ``"cpu"``, ``"cuda"``, or ``"mps"``. The value is
                lower-cased before storage.
        """
        self.device = device_code.lower()
        logger.info("Target device set to: %s", self.device)

    def _log_hardware_environment(self) -> None:
        """Log a full hardware availability report to help diagnose device selection."""
        logger.info("── Hardware Environment ─────────────────────────────────")
        logger.info("  Platform   : %s %s", platform.system(), platform.release())
        logger.info("  Python     : %s", platform.python_version())
        logger.info("  PyTorch    : %s", torch.__version__)

        # CUDA
        cuda_available = torch.cuda.is_available()
        logger.info(
            "  CUDA       : %s", "available" if cuda_available else "not available"
        )
        if cuda_available:
            logger.info("    CUDA toolkit : %s", torch.version.cuda)
            device_count = torch.cuda.device_count()
            logger.info("    GPU count    : %d", device_count)
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                logger.info("    GPU %d        : %s (%.1f GB VRAM)", i, name, vram_gb)
        else:
            if not torch.cuda.is_available():
                # Distinguish: no NVIDIA driver vs CUDA build missing
                cuda_built = torch.version.cuda is not None
                if cuda_built:
                    logger.info(
                        "    (PyTorch was built with CUDA %s but no NVIDIA GPU/driver found)",
                        torch.version.cuda,
                    )
                else:
                    logger.info("    (PyTorch was not built with CUDA support)")

        # MPS
        mps_built = hasattr(torch.backends, "mps")
        if mps_built:
            mps_available = torch.backends.mps.is_available()
            logger.info(
                "  MPS        : %s", "available" if mps_available else "not available"
            )
            if not mps_available:
                mps_built_flag = torch.backends.mps.is_built()
                if not mps_built_flag:
                    logger.info("    (PyTorch was not built with MPS support)")
                else:
                    logger.info(
                        "    (MPS built but not available — requires macOS 12.3+ with Apple GPU)"
                    )
        else:
            logger.info(
                "  MPS        : not available (PyTorch < 1.12 — no MPS backend)"
            )

        logger.info(
            "  CPU        : always available (%d logical cores)", os.cpu_count() or 0
        )
        logger.info("─────────────────────────────────────────────────────────")

    def _resolve_device(self) -> str:
        """Resolve the effective compute device, applying auto-detection if needed.

        Returns:
            str: The resolved device string — ``"cuda"``, ``"mps"``, or
                ``"cpu"`` when :attr:`device` is ``"auto"``; otherwise
                returns :attr:`device` unchanged.
        """
        self._log_hardware_environment()

        if self.device != "auto":
            logger.info(
                "Device selection: using explicitly requested device '%s'", self.device
            )
            return self.device

        if torch.cuda.is_available():
            logger.info("Device selection: auto → CUDA (NVIDIA GPU detected)")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info(
                "Device selection: auto → MPS (Apple GPU detected, CUDA not available)"
            )
            return "mps"
        logger.info("Device selection: auto → CPU (no GPU acceleration available)")
        return "cpu"

    def load_from_folder(self, folder_path: str) -> None:
        """Not supported — PyTorch strategies require a single model file.

        Args:
            folder_path (str): Unused directory path.

        Raises:
            NotImplementedError: Always; use :meth:`load_from_file` instead.
        """
        raise NotImplementedError("Use load_from_file() for PyTorch models.")

    def load_from_file(self, model_path: str) -> None:
        """Load a ``.pt`` or ``.ckpt`` model file.

        Attempts :class:`~anomalib.deploy.TorchInferencer` first (Attempt 1).
        On failure falls back to raw ``torch.load`` with
        :class:`DynamicPickleModule` (Attempt 2). MPS devices are handled by
        loading on CPU first and then moving tensors to the MPS device.

        Args:
            model_path (str): Absolute path to the model file. Must have a
                ``.pt`` or ``.ckpt`` extension.

        Raises:
            RuntimeError: If both loading attempts fail, wrapping the
                underlying exception message.
        """
        path = Path(model_path)
        self._device_verified = False
        self.torch_inferencer = None
        self.raw_model = None

        t_total = time.perf_counter()

        try:
            os.environ["TRUST_REMOTE_CODE"] = "1"

            if path.suffix not in (".pt", ".ckpt"):
                raise ValueError(
                    f"Unsupported file type: {path.suffix}. Expected .pt or .ckpt"
                )

            self.model_type = "torch"
            resolved_device = self._resolve_device()
            if resolved_device == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    final_device = f"CUDA ({gpu_name})"
                except Exception:
                    final_device = "CUDA"
            elif resolved_device == "mps":
                final_device = "MPS (Apple Silicon)"
            else:
                final_device = "CPU"

            import functools

            original_torch_load = torch.load
            original_posix_path = pathlib.PosixPath

            try:
                # Attempt 1: Anomalib TorchInferencer (monkey-patch forces CPU deserialisation)
                try:
                    torch.load = functools.partial(
                        original_torch_load, map_location="cpu"
                    )
                    if platform.system() == "Windows":
                        pathlib.PosixPath = pathlib.WindowsPath

                    if resolved_device == "mps":
                        logger.debug("Applying MPS shim: initialising on CPU first.")
                        t_inferencer = time.perf_counter()
                        self.torch_inferencer = TorchInferencer(path=path, device="cpu")
                        logger.info(
                            "Load phase: TorchInferencer init — %.2fs",
                            time.perf_counter() - t_inferencer,
                        )

                        t_mps = time.perf_counter()
                        mps_device = torch.device("mps")
                        if hasattr(self.torch_inferencer, "model"):
                            self.torch_inferencer.model = (
                                self.torch_inferencer.model.to(mps_device)
                            )
                        self.torch_inferencer.device = mps_device
                        logger.info(
                            "Load phase: MPS device move — %.2fs",
                            time.perf_counter() - t_mps,
                        )

                        final_device = "MPS (Apple Silicon)"
                    else:
                        t_inferencer = time.perf_counter()
                        self.torch_inferencer = TorchInferencer(
                            path=path, device=resolved_device
                        )
                        logger.info(
                            "Load phase: TorchInferencer init — %.2fs",
                            time.perf_counter() - t_inferencer,
                        )

                    self.model_name = f"Anomalib (Torch) [{final_device}]"
                    logger.info("Loaded %s via TorchInferencer", self.model_name)

                finally:
                    torch.load = original_torch_load
                    pathlib.PosixPath = original_posix_path

            except Exception as anomalib_err:
                # Attempt 2: Raw PyTorch fallback with DynamicUnpickler
                logger.warning(
                    "TorchInferencer rejected the model (%s). Trying raw fallback.",
                    anomalib_err,
                )
                self.torch_inferencer = None

                device_obj = torch.device(
                    resolved_device if resolved_device != "mps" else "cpu"
                )
                t_raw = time.perf_counter()
                loaded_data = torch.load(
                    path, map_location=device_obj, pickle_module=DynamicPickleModule
                )
                logger.info(
                    "Load phase: raw torch.load — %.2fs", time.perf_counter() - t_raw
                )

                if isinstance(loaded_data, dict):
                    if "state_dict" in loaded_data and "model" not in loaded_data:
                        raise ValueError(
                            "You loaded a training .ckpt file. "
                            "Please select the exported model.pt from your weights/torch folder."
                        )
                    elif "model" in loaded_data:
                        self.raw_model = loaded_data["model"]
                    else:
                        raise ValueError(
                            "Loaded dict does not contain a recognisable model graph."
                        )
                else:
                    self.raw_model = loaded_data

                if hasattr(self.raw_model, "eval"):
                    self.raw_model.eval()

                if resolved_device == "mps":
                    try:
                        t_mps = time.perf_counter()
                        self.raw_model = self.raw_model.to(torch.device("mps"))
                        logger.info(
                            "Load phase: MPS device move — %.2fs",
                            time.perf_counter() - t_mps,
                        )
                        final_device = "MPS (Apple Silicon)"
                    except Exception as mps_err:
                        logger.debug("Failed to push raw model to MPS: %s", mps_err)

                self.model_name = f"PyTorch Checkpoint [{final_device}]"
                logger.info("Loaded %s via raw torch.load", self.model_name)

            logger.info("Load total — %.2fs", time.perf_counter() - t_total)

        except Exception as e:
            logger.error("Critical failure loading model: %s", e)
            raise RuntimeError(f"Load Error: {e}")

    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Run inference on a single image using whichever backend is loaded.

        Dispatches to :meth:`_predict_anomalib` when a
        :class:`~anomalib.deploy.TorchInferencer` is active, or to
        :meth:`_predict_raw` when a raw model is loaded. Returns a zero
        score and a blank heatmap when no model is loaded.

        Args:
            image_path (str): Absolute path to the input image file.

        Returns:
            Tuple[float, np.ndarray]: ``(anomaly_score, heatmap)`` where
                *heatmap* is a 2-D ``float32`` array normalised to 0–1.
        """
        if self.torch_inferencer:
            return self._predict_anomalib(image_path)
        if self.raw_model:
            return self._predict_raw(image_path)
        return 0.0, np.zeros((256, 256), dtype=np.float32)

    def _predict_anomalib(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Run inference via :attr:`torch_inferencer` and return a normalised result.

        On the first call, logs the device the model tensors are resident on.
        Synchronises the MPS stream after inference when running on Apple
        Silicon. On any exception, logs the error and returns a zero score
        with a blank heatmap.

        Args:
            image_path (str): Absolute path to the input image file.

        Returns:
            Tuple[float, np.ndarray]: ``(anomaly_score, heatmap)`` where
                *heatmap* is a 2-D ``float32`` array normalised to 0–1.
        """
        try:
            if not self._device_verified:
                try:
                    p = next(self.torch_inferencer.model.parameters())
                    logger.debug("Tensors are on %s", p.device)
                    self._device_verified = True
                except StopIteration:
                    pass

            result = self.torch_inferencer.predict(image=image_path)

            if self.device == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            score = 0.0
            if hasattr(result, "pred_score") and result.pred_score is not None:
                score = float(
                    result.pred_score.item()
                    if isinstance(result.pred_score, torch.Tensor)
                    else result.pred_score
                )

            heatmap = result.anomaly_map
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
            heatmap = heatmap.squeeze()

            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap.astype(np.float32)

        except Exception as e:
            logger.error("Anomalib inference failed: %s", e)
            return 0.0, np.zeros((256, 256), dtype=np.float32)

    def _predict_raw(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Run inference via :attr:`raw_model` using a manual preprocessing pipeline.

        Reads the image with OpenCV, converts to RGB, resizes to 256×256,
        normalises to 0–1, and runs a ``torch.no_grad`` forward pass.
        Extracts the heatmap from the first multi-element floating-point
        tensor in the output and the score from any scalar tensor. On any
        exception, logs the error and returns a zero score with a blank
        heatmap.

        Args:
            image_path (str): Absolute path to the input image file.

        Returns:
            Tuple[float, np.ndarray]: ``(anomaly_score, heatmap)`` where
                *heatmap* is a 2-D ``float32`` array normalised to 0–1.
        """
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))

            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            device_obj = (
                next(self.raw_model.parameters()).device
                if hasattr(self.raw_model, "parameters")
                else torch.device("cpu")
            )
            tensor = tensor.to(device_obj)

            with torch.no_grad():
                output = self.raw_model(tensor)

            score = 0.0
            heatmap = np.zeros((256, 256), dtype=np.float32)

            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        if (
                            item.ndim >= 2
                            and item.numel() > 1
                            and item.is_floating_point()
                        ):
                            heatmap = item.squeeze().cpu().numpy()
                        elif item.numel() == 1:
                            score = float(item.cpu().item())
            elif isinstance(output, torch.Tensor):
                heatmap = output.squeeze().cpu().numpy()

            heatmap = heatmap.astype(np.float32)
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap

        except Exception as e:
            logger.error("Raw PyTorch inference failed: %s", e)
            return 0.0, np.zeros((256, 256), dtype=np.float32)
