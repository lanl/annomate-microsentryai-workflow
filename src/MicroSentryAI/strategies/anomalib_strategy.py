"""
Anomalib v2.2.0 Strategy Module for MicroSentryAI.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from anomalib.deploy import TorchInferencer, OpenVINOInferencer

logger = logging.getLogger(__name__)

class AnomalibStrategy:
    """
    Unified strategy for Anomalib v2.2.0 supporting Torch (.pt) and OpenVINO (.xml).
    """

    def __init__(self):
        self.inferencer: Optional[TorchInferencer] = None
        self.device = "auto"
        self.model_name = "Unknown"

    def set_device(self, device_code: str):
        """Sets the hardware device for v2.2.0 inference."""
        # Options: "auto", "cpu", "cuda"
        self.device = device_code.lower()
        logger.info(f"Target Device set to: {self.device}")

    def load_from_file(self, model_path: str):
        """
        Loads exported models. In v2.2.0, .pt files exported via 
        'anomalib export' contain embedded metadata.
        """
        path = Path(model_path)
        
        try:
            os.environ["TRUST_REMOTE_CODE"] = "1" 

            if path.suffix == ".pt":
                # v2.2.0 TorchInferencer handles the metadata internal to the .pt
                self.inferencer = TorchInferencer(path=path, device=self.device)
                self.model_name = f"Anomalib v2 (Torch: {path.stem})"
            elif path.suffix == ".xml":
                # OpenVINO still prefers the .xml path
                self.inferencer = OpenVINOInferencer(path=path, device="CPU")
                self.model_name = f"Anomalib v2 (OpenVINO: {path.stem})"
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
                
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"v2.2.0 Load Error: {e}")

    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """
        Runs inference using the v2.2.0 predict API.
        Returns (anomaly_score, normalized_heatmap).
        """
        if self.inferencer is None:
            return 0.0, np.zeros((256, 256), dtype=np.float32)

        try:
            result = self.inferencer.predict(image=image_path)

            # --- 1. Handle the Score (EfficientAD Fix) ---
            if isinstance(result.pred_score, torch.Tensor):
                score = float(result.pred_score.detach().cpu())
            else:
                score = float(result.pred_score)
            
            heatmap = result.anomaly_map
            
            # --- 2. Handle the Heatmap (MISSING PART) ---
            # We must convert Tensor -> Numpy before doing anything else
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
            
            # --- 3. Fix Dimensions ---
            heatmap = heatmap.squeeze()

            # --- 4. Normalize ---
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap.astype(np.float32)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return 0.0, np.zeros((256, 256), dtype=np.float32)