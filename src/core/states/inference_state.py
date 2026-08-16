import numpy as np


class InferenceState:
    """MicroSentryAI domain state container for inference results.

    Stores inference score maps and peak anomaly scores keyed by image
    filename, for whichever model is currently active. Also tracks a
    registry of every model known to this project, and each one's scalar
    scores, so switching the active model doesn't lose the others' results.
    Contains zero Qt dependencies.

    Attributes:
        score_maps (dict[str, np.ndarray]): Full heatmap arrays indexed
            by image filename, for the active model only.
        inference_cache (dict[str, float]): Normalized anomaly scores [0,1] indexed
            by image filename (e.g. ``{"img.jpg": 0.71}``). Mirrors ``scores``.
        scores (dict[str, float]): Actual pred_score [0,1] from the model's
            PostProcessor, keyed by image filename, for the active model only.
            Aliases ``model_scores[active_model_key]`` — mutating it in place
            (e.g. via ``set_score_map``) keeps ``model_scores`` in sync for free.
        labels (dict[str, str]): Classification label per image —
            ``"ANOMALY"`` when score >= 0.5, ``"NORMAL"`` otherwise.
        known_models (dict[str, dict]): Every model known to this project,
            keyed by model key (derived from the ``.pt`` filename stem).
            Each value is ``{"model_path": str, "score_maps_file": str}``.
        model_scores (dict[str, dict[str, float]]): Every known model's
            scalar scores, keyed by model key then filename. Always fully
            resident — this is what gets persisted to the project JSON.
        active_model_key (str): Key of the model whose results are currently
            reflected in ``scores``/``labels``/``inference_cache``/``score_maps``.
        score_maps_dirty (bool): True when the active model's score_maps have
            changed since the last NPZ write.
    """

    def __init__(self) -> None:
        """Initialize InferenceState with empty score maps and cache."""
        self.score_maps = {}  # { "img.jpg": np.ndarray }  active model's heatmap arrays
        self.inference_cache = {}  # { "img.jpg": float }  peak anomaly scores
        self.scores: dict[
            str, float
        ] = {}  # { "img.jpg": float }  actual pred_score [0,1], active model only
        self.labels: dict[str, str] = {}  # { "img.jpg": "ANOMALY" | "NORMAL" }
        self.score_maps_dirty: bool = (
            False  # True when score_maps changed since last NPZ write
        )

        self.known_models: dict[str, dict] = {}  # model_key -> {"model_path", "score_maps_file"}
        self.model_scores: dict[str, dict[str, float]] = {}  # model_key -> {"img.jpg": float}
        self.active_model_key: str = ""

    def clear(self) -> None:
        """Clear all stored score maps, cached anomaly scores, and the model registry."""
        self.score_maps.clear()
        self.inference_cache.clear()
        self.scores.clear()
        self.labels.clear()
        self.score_maps_dirty = False
        self.known_models.clear()
        self.model_scores.clear()
        self.active_model_key = ""

    def set_score_map(self, filename: str, score: float, score_map: np.ndarray) -> None:
        """Store a score map, the actual pred_score, and its ANOMALY/NORMAL label.

        Updates the active model's entry in ``model_scores`` for free, since
        ``scores`` aliases it (see :meth:`switch_active_model`).

        Args:
            filename (str): Image filename used as the storage key.
            score (float): Normalized anomaly score [0, 1] from the model's PostProcessor.
            score_map (np.ndarray): 2-D heatmap array of anomaly scores.
        """
        self.score_maps[filename] = score_map
        self.scores[filename] = float(score)
        self.labels[filename] = "ANOMALY" if score >= 0.5 else "NORMAL"
        self.inference_cache[filename] = float(score)
        self.score_maps_dirty = True

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

    # ------------------------------------------------------------------ #
    # Multi-model registry
    # ------------------------------------------------------------------ #

    def register_model(self, key: str, model_path: str, score_maps_file: str) -> None:
        """Add or update a model in the known-models registry.

        Does not change which model is active. Safe to call for a key that's
        already registered (e.g. to update its path) — an empty
        *score_maps_file* never erases a real path already on record for
        this key, so re-loading a model's weights doesn't orphan its
        previously-saved heatmap cache.

        Args:
            key (str): Model key, derived from the ``.pt`` filename stem.
            model_path (str): Absolute path to the model checkpoint file.
            score_maps_file (str): Project-relative path to this model's
                cached heatmap NPZ (e.g. ``"scoremaps/efficientad-scoremaps.npz"``).
                Pass ``""`` when unknown — any existing path is kept.
        """
        existing = self.known_models.get(key, {})
        self.known_models[key] = {
            "model_path": model_path,
            "score_maps_file": score_maps_file or existing.get("score_maps_file", ""),
        }
        self.model_scores.setdefault(key, {})

    def switch_active_model(self, key: str) -> None:
        """Make *key* the active model.

        Repoints ``scores`` at ``model_scores[key]`` (creating an empty entry
        if this key has no cached scores yet) so it aliases the persistent
        dict — further ``set_score_map`` calls update both for free. Clears
        ``score_maps`` (heatmaps); the caller is responsible for loading the
        new active model's cached heatmaps from disk afterward, if any exist.

        Args:
            key (str): Model key to activate. Must already be registered via
                :meth:`register_model`.
        """
        self.active_model_key = key
        self.model_scores.setdefault(key, {})
        self.scores = self.model_scores[key]
        self.labels = {
            fname: ("ANOMALY" if s >= 0.5 else "NORMAL")
            for fname, s in self.scores.items()
        }
        self.inference_cache = dict(self.scores)
        self.score_maps = {}
        self.score_maps_dirty = False
