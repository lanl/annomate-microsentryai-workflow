import numpy as np

_DEFAULT_MICROSENTRY_SETTINGS = {
    "panel_enabled": False,
    "heatmap_enabled": False,
    "seg_enabled": False,
    "seg_pct": 95,
    "alpha": 0.45,
    "sigma": 4,
    "epsilon": 12,
    "heat_min": 0,
}


def default_microsentry_settings() -> dict:
    """Return a fresh copy of the default MicroSentryAI user settings."""
    return dict(_DEFAULT_MICROSENTRY_SETTINGS)


class InferenceState:
    """MicroSentryAI domain state container for inference results.

    Stores inference score maps, peak anomaly scores, and reloadable
    MicroSentryAI user settings. Contains zero Qt dependencies.

    Attributes:
        score_maps (dict[str, np.ndarray]): Full heatmap arrays indexed
            by image filename (e.g. ``{"img.jpg": np.ndarray}``).
        inference_cache (dict[str, float]): Peak anomaly scores indexed
            by image filename (e.g. ``{"img.jpg": 0.93}``).
        microsentry_settings (dict): User-facing MicroSentryAI tuning
            settings persisted in .annoproj project state.
    """

    def __init__(self) -> None:
        """Initialize InferenceState with empty score maps, cache, and settings."""
        self.score_maps = {}  # { "img.jpg": np.ndarray }  full heatmap arrays
        self.inference_cache = {}  # { "img.jpg": float }  peak anomaly scores
        self.microsentry_settings = default_microsentry_settings()

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

    def set_microsentry_settings(self, settings: dict) -> None:
        """Replace persisted MicroSentryAI settings with validated known keys."""
        if not isinstance(settings, dict):
            return
        merged = default_microsentry_settings()
        for key in merged:
            if key in settings:
                merged[key] = settings[key]
        self.microsentry_settings = merged
