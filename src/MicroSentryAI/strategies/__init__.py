"""
Strategy Factory Module.

This module serves as the package entry point for model inference strategies.
It provides a factory function to instantiate the appropriate strategy class
(e.g., Anomalib) which abstracts the underlying model execution logic.
"""

from .anomalib_strategy import AnomalibStrategy


def load_strategy_from_folder(folder_path: str) -> AnomalibStrategy:
    """
    Factory function to obtain a model strategy instance.

    This function initializes the default strategy (AnomalibStrategy). The strategy
    instance is returned ready to load specific model weights.

    Args:
        folder_path (str): The directory containing the model artifacts.
                           (Currently unused for dispatch, but reserved for
                           future strategy selection logic based on folder contents).

    Returns:
        AnomalibStrategy: An instance of the inference strategy.
    """
    # Instantiate the universal Anomalib strategy.
    # Note: We do not call load_from_folder() here; the UI controls the
    # lifecycle of when the model weights are actually loaded.
    strategy = AnomalibStrategy()
    return strategy