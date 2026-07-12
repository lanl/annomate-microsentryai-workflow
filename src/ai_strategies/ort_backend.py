"""
Shared ONNX Runtime session plumbing for all AI strategies.

Provider-agnostic: the same code runs in CPU-only, DirectML, and CUDA builds
of the application — only the installed onnxruntime package differs. Execution
providers are picked from ``ort.get_available_providers()`` by priority.
No Qt dependencies.
"""

import os
import logging
import platform
from typing import List, Tuple

import onnxruntime as ort

logger = logging.getLogger("MicroSentryAI.OrtBackend")

# In CUDA builds the cuBLAS/cuDNN/cudart DLLs come from the nvidia-*-cu12 pip
# packages, whose site-packages/nvidia/*/bin directories are not on the OS DLL
# search path. preload_dlls() loads the core DLLs by absolute path, but cuDNN
# also lazy-loads engine sub-DLLs by bare name at kernel-execution time (e.g.
# cudnn_engines_tensor_ir64_9.dll, which is missing from preload_dlls()'s
# hard-coded list), so the bin directories must also be on the search path.
# No-op on CPU/DirectML builds (guarded) and when the packages are absent.
if platform.system() == "Windows" and "CUDAExecutionProvider" in ort.get_available_providers():
    _nvidia_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(ort.__file__))), "nvidia"
    )
    _found_nvidia_bin = False
    if os.path.isdir(_nvidia_root):
        for _pkg in sorted(os.listdir(_nvidia_root)):
            _bin_dir = os.path.join(_nvidia_root, _pkg, "bin")
            if os.path.isdir(_bin_dir):
                _found_nvidia_bin = True
                os.add_dll_directory(_bin_dir)
                os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")
    # Without the pip packages (e.g. system-wide CUDA/cuDNN found via PATH by
    # the native loader), preload_dlls would only print noise: ctypes does not
    # search PATH on Python 3.8+.
    if _found_nvidia_bin and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

# Preferred execution providers, best first. CPU is always appended as the
# final fallback so a session can be created in any build.
_EP_PRIORITY = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

_EP_LABELS = {
    "CUDAExecutionProvider": "CUDA",
    "DmlExecutionProvider": "DirectML",
    "CoreMLExecutionProvider": "CoreML",
    "CPUExecutionProvider": "CPU",
}


def resolve_providers(device: str) -> Tuple[List[str], str]:
    """Map a device string to an ONNX Runtime execution-provider list.

    Args:
        device (str): Requested device — ``"auto"`` picks the best available
            provider by priority (CUDA → DirectML → CoreML → CPU); ``"cpu"``
            forces CPU; ``"cuda"`` requests the CUDA provider; ``"mps"`` is
            accepted for backward compatibility and maps to CoreML.

    Returns:
        Tuple[List[str], str]: ``(providers, label)`` where *providers* is the
            ordered provider list to pass to ``InferenceSession`` (always
            ending in ``CPUExecutionProvider``) and *label* is a
            human-readable name of the primary provider (e.g. ``"CUDA"``).
    """
    available = ort.get_available_providers()
    device = device.lower()

    if device == "cpu":
        logger.info("Device selection: using explicitly requested device 'cpu'")
        return ["CPUExecutionProvider"], "CPU"

    requested = {
        "cuda": "CUDAExecutionProvider",
        "mps": "CoreMLExecutionProvider",
    }.get(device)

    if requested is not None:
        if requested in available:
            logger.info(
                "Device selection: using explicitly requested provider %s", requested
            )
            return [requested, "CPUExecutionProvider"], _EP_LABELS[requested]
        logger.warning(
            "Requested device '%s' (%s) is not available in this build "
            "(available: %s). Falling back to CPU.",
            device,
            requested,
            available,
        )
        return ["CPUExecutionProvider"], "CPU"

    # "auto" (and anything unrecognised): best available by priority.
    for provider in _EP_PRIORITY:
        if provider in available:
            label = _EP_LABELS[provider]
            logger.info("Device selection: auto → %s", label)
            if provider == "CPUExecutionProvider":
                return ["CPUExecutionProvider"], label
            return [provider, "CPUExecutionProvider"], label

    logger.info("Device selection: auto → CPU (no accelerated provider available)")
    return ["CPUExecutionProvider"], "CPU"


def create_session(model_path: str, device: str) -> Tuple[ort.InferenceSession, str]:
    """Create an ``InferenceSession`` for *model_path* honouring *device*.

    If an accelerated provider is selected but fails at session-creation time
    (e.g. the CUDA provider is installed but the system CUDA/cuDNN libraries
    are missing), the session is retried on CPU with a logged warning instead
    of failing the load.

    Args:
        model_path (str): Absolute path to the ``.onnx`` model file.
        device (str): Device string as accepted by :func:`resolve_providers`.

    Returns:
        Tuple[ort.InferenceSession, str]: The created session and the
            human-readable label of the provider actually in use.

    Raises:
        Exception: Whatever ``InferenceSession`` raises when even the CPU
            provider cannot load the model (corrupt/invalid file).
    """
    providers, label = resolve_providers(device)
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        if providers == ["CPUExecutionProvider"]:
            raise
        logger.warning(
            "Session creation with %s failed (%s). Retrying on CPU.", label, exc
        )
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        label = "CPU"

    active = session.get_providers()[0]
    label = _EP_LABELS.get(active, active)
    logger.info("Session created for %s using %s", model_path, label)
    return session, label


def log_ort_environment() -> None:
    """Log the ONNX Runtime environment to help diagnose provider selection."""
    logger.info("── ONNX Runtime Environment ─────────────────────────────")
    logger.info("  Platform   : %s %s", platform.system(), platform.release())
    logger.info("  Python     : %s", platform.python_version())
    logger.info("  onnxruntime: %s", ort.__version__)
    logger.info("  Providers  : %s", ", ".join(ort.get_available_providers()))
    logger.info("  Device     : %s", ort.get_device())
    logger.info(
        "  CPU        : always available (%d logical cores)", os.cpu_count() or 0
    )
    logger.info("─────────────────────────────────────────────────────────")
