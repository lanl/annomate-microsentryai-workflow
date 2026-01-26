"""
Abstract Strategy Interface.

This module defines the `AnomalyDetectionStrategy` abstract base class.
It establishes the contract that all specific model implementations (e.g.,
AnomalibStrategy) must fulfill, ensuring a consistent API for loading models
and running inference regardless of the underlying algorithm.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AnomalyDetectionStrategy(ABC):
    """
    Abstract base class for all anomaly detection model strategies.

    This interface mandates that any concrete strategy must implement methods
    for loading model artifacts from a folder and performing inference on a
    single image.
    """

    def __init__(self):
        """Initialize the base strategy with a default model name."""
        self.model_name = "Unknown"

    @abstractmethod
    def load_from_folder(self, folder_path: str) -> None:
        """
        Loads model weights, configuration, and metadata from the specified directory.

        Args:
            folder_path (str): The absolute path to the directory containing
                               model artifacts (e.g., .ckpt, .pt, .json files).

        Raises:
            FileNotFoundError: If required files are missing.
            RuntimeError: If the model fails to load.
        """
        pass

    @abstractmethod
    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """
        Runs inference on a single image.

        Args:
            image_path (str): The absolute path to the input image file.

        Returns:
            Tuple[float, np.ndarray]: A tuple containing:
                - anomaly_score (float): The maximum anomaly score for the image.
                - heatmap (np.ndarray): A normalized (0.0 to 1.0) anomaly heatmap
                                        as a float32 numpy array.
        """
        pass