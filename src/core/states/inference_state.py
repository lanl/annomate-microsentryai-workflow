import numpy as np


class InferenceState:
    """MicroSentryAI domain state container for inference results.

    Stores inference score maps and peak anomaly scores keyed by image
    filename. Contains zero Qt dependencies.

    Attributes:
        score_maps (dict[str, np.ndarray]): Full heatmap arrays indexed
            by image filename (e.g. ``{"img.jpg": np.ndarray}``).
        inference_cache (dict[str, float]): Peak anomaly scores indexed
            by image filename (e.g. ``{"img.jpg": 0.93}``).
    """

    def __init__(self) -> None:
        """Initialize InferenceState with empty score maps and cache."""
        self.score_maps = {}  # { "img.jpg": np.ndarray }  full heatmap arrays
        self.inference_cache = {}  # { "img.jpg": float }  peak anomaly scores

    def clear(self) -> None:
        """Clear all stored score maps and cached anomaly scores."""
        self.score_maps.clear()
        self.inference_cache.clear()

    def set_score_map(self, filename: str, score_map: np.ndarray) -> None:
        """Store a score map and cache its peak anomaly score.

        Args:
            filename (str): Image filename used as the storage key.
            score_map (np.ndarray): 2-D heatmap array of anomaly scores.
        """
        self.score_maps[filename] = score_map
        self.inference_cache[filename] = float(score_map.max())

    def get_score_map(self, filename: str) -> np.ndarray | None:
        """Return the stored score map for the given filename.

        Args:
            filename (str): Image filename to look up.

        Returns:
            np.ndarray | None: The heatmap array, or ``None`` if the
                filename has not been processed.
        """
        return self.score_maps.get(filename)

    def is_processed(self, filename: str) -> bool:
        """Check whether an image has already been through inference.

        Args:
            filename (str): Image filename to check.

        Returns:
            bool: ``True`` if a score map exists for *filename*,
                ``False`` otherwise.
        """
        return filename in self.score_maps
