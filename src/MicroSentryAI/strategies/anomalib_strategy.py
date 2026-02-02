"""
Anomalib v2.2.0 Strategy Module for MicroSentryAI.

This module provides the concrete implementation of the anomaly detection strategy
using the Anomalib v2.2.0 library. It serves as an adapter between the
MicroSentryAI application and the underlying TorchInferencer, handling hardware
acceleration (CUDA, MPS, CPU) and model loading.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

# --- CRITICAL FIX: Enable MPS Fallback ---
# This must be set before any heavy torch operations to prevent 
# segfaults on operators not yet fully supported by Apple Silicon.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from anomalib.deploy import TorchInferencer

logger = logging.getLogger(__name__)


class AnomalibStrategy:
    """
    Unified strategy for Anomalib v2.2.0 supporting Torch (.pt) artifacts.
    
    This class implements the 'Shim Pattern' to enable Apple Silicon (MPS)
    support, which is natively supported by PyTorch but occasionally restricted
    by Anomalib's strict device validation.
    """

    def __init__(self):
        self.inferencer: Optional[TorchInferencer] = None
        self.device = "auto"
        self.model_name = "Unknown"
        self._device_verified = False  # Flag for one-time runtime check

    def set_device(self, device_code: str):
        """
        Sets the preferred hardware device for inference.

        Args:
            device_code (str): One of "auto", "cpu", "cuda", or "mps".
        """
        self.device = device_code.lower()
        logger.info(f"Target Device set to: {self.device}")

    def _resolve_device(self) -> str:
        """
        Resolves the 'auto' setting to the highest-priority available accelerator.
        
        Priority:
            1. CUDA (NVIDIA GPU)
            2. MPS (Apple Silicon)
            3. CPU (Fallback)

        Returns:
            str: The resolved device string ('cuda', 'mps', or 'cpu').
        """
        if self.device != "auto":
            return self.device

        if torch.cuda.is_available():
            logger.info("Auto-resolution: CUDA detected.")
            return "cuda"
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Auto-resolution: MPS (Apple Silicon) detected.")
            return "mps"
            
        logger.info("Auto-resolution: Defaulting to CPU.")
        return "cpu"

    def _apply_mps_shim(self, path: Path):
        """
        Applies a compatibility shim to force MPS execution.

        Anomalib's TorchInferencer validation may reject 'mps' during initialization.
        This method initializes on CPU (to pass validation) and then manually 
        transfers the underlying model and state to the MPS device.

        Args:
            path (Path): Path to the model weights file.
        """
        # 1. Initialize on CPU to bypass library validation
        logger.debug("Applying MPS Shim: Initializing on CPU.")
        self.inferencer = TorchInferencer(path=path, device="cpu")

        # 2. Manually transfer the model to MPS
        try:
            logger.info("Transferring model to MPS device...")
            mps_device = torch.device("mps")
            
            # Move the underlying PyTorch model
            if hasattr(self.inferencer, 'model'):
                self.inferencer.model = self.inferencer.model.to(mps_device)
            
            # Update the inferencer's internal device reference
            self.inferencer.device = mps_device
            
            logger.info("MPS Shim applied successfully.")
        except Exception as e:
            logger.error(f"MPS Shim failed: {e}. Reverting to CPU.")
            self.inferencer.device = torch.device("cpu")
            # Ensure model is back on CPU if partial failure occurred
            if hasattr(self.inferencer, 'model'):
                self.inferencer.model = self.inferencer.model.to("cpu")

    def load_from_file(self, model_path: str):
        """
        Loads exported Torch (.pt) models.

        Args:
            model_path (str): Absolute path to the .pt file.

        Raises:
            RuntimeError: If the model fails to load or the file type is unsupported.
        """
        path = Path(model_path)
        self._device_verified = False  # Reset verification flag on new load
        
        try:
            os.environ["TRUST_REMOTE_CODE"] = "1" 

            if path.suffix != ".pt":
                raise ValueError(f"Unsupported file type: {path.suffix}. Only .pt is supported.")

            resolved_device = self._resolve_device()
            
            if resolved_device == "mps":
                self._apply_mps_shim(path)
                final_device_name = "MPS (Apple Silicon)"
            else:
                # Standard initialization for fully supported devices (CPU/CUDA)
                self.inferencer = TorchInferencer(path=path, device=resolved_device)
                final_device_name = resolved_device.upper()

            self.model_name = f"Anomalib (Torch: {path.stem}) [{final_device_name}]"
            logger.info(f"Successfully loaded {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"v2.2.0 Load Error: {e}")

    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """
        Runs inference on the provided image using the loaded model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[float, np.ndarray]: A tuple containing the global anomaly score
                                      and the normalized pixel-level heatmap.
        """
        if self.inferencer is None:
            return 0.0, np.zeros((256, 256), dtype=np.float32)

        # --- RUNTIME HARDWARE VERIFICATION ---
        if not self._device_verified:
            try:
                # Check the device location of the first parameter in the model
                if hasattr(self.inferencer, 'model'):
                    first_param = next(self.inferencer.model.parameters())
                    actual_device = first_param.device
                    logger.critical(f"VERIFICATION: Model tensors are physically resident on: {actual_device}")
                self._device_verified = True
            except Exception as e:
                logger.warning(f"Could not verify device placement: {e}")
        # -------------------------------------

        try:
            # Execute Inference
            result = self.inferencer.predict(image=image_path)
            
            # If we are on Apple Silicon, we MUST ensure the GPU is finished 
            # writing memory before we let the CPU try to read it.
            if self.device == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            # --- 1. Extract Anomaly Score ---
            if isinstance(result.pred_score, torch.Tensor):
                score = float(result.pred_score.detach().cpu())
            else:
                score = float(result.pred_score)
            
            # --- 2. Extract and Process Heatmap ---
            heatmap = result.anomaly_map
            
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
            
            heatmap = heatmap.squeeze()

            # --- 3. Normalize Heatmap (0.0 - 1.0) ---
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap.astype(np.float32)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return 0.0, np.zeros((256, 256), dtype=np.float32)