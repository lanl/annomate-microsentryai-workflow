import numpy as np


class InferenceState:
    """MicroSentryAI domain state container for inference results.

    Stores inference score maps and peak anomaly scores keyed by image
    filename. Contains zero Qt dependencies.

    Attributes:
        score_maps (dict[str, np.ndarray]): Full heatmap arrays indexed
            by image filename (e.g. ``{"img.jpg": np.ndarray}``).
        inference_cache (dict[str, float]): Normalized anomaly scores [0,1] indexed
            by image filename (e.g. ``{"img.jpg": 0.71}``). Mirrors ``scores``.
        scores (dict[str, float]): Actual pred_score [0,1] from the model's
            PostProcessor, keyed by image filename.
        labels (dict[str, str]): Classification label per image —
            ``"ANOMALY"`` when score >= 0.5, ``"NORMAL"`` otherwise.
    """

    def __init__(self) -> None:
        """Initialize InferenceState with empty score maps and cache."""
        self.score_maps = {}  # { "img.jpg": np.ndarray }  full heatmap arrays
        self.inference_cache = {}  # { "img.jpg": float }  peak anomaly scores
        self.scores: dict[
            str, float
        ] = {}  # { "img.jpg": float }  actual pred_score [0,1]
        self.labels: dict[str, str] = {}  # { "img.jpg": "ANOMALY" | "NORMAL" }

    def clear(self) -> None:
        """Clear all stored score maps and cached anomaly scores."""
        self.score_maps.clear()
        self.inference_cache.clear()
        self.scores.clear()
        self.labels.clear()

    def set_score_map(self, filename: str, score: float, score_map: np.ndarray) -> None:
        """Store a score map, the actual pred_score, and its ANOMALY/NORMAL label.

        Args:
            filename (str): Image filename used as the storage key.
            score (float): Normalized anomaly score [0, 1] from the model's PostProcessor.
            score_map (np.ndarray): 2-D heatmap array of anomaly scores.
        """
        self.score_maps[filename] = score_map
        self.scores[filename] = float(score)
        self.labels[filename] = "ANOMALY" if score >= 0.5 else "NORMAL"
        self.inference_cache[filename] = float(score)

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
