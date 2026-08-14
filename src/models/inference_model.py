import numpy as np

from core.states.inference_state import InferenceState


class InferenceModel:
    """Pure Python model for MicroSentryAI inference results.

    Wraps :class:`~core.states.inference_state.InferenceState` with a clean
    query/command API. Contains no Qt dependencies and is fully testable
    without a ``QApplication``. Views must use this API instead of accessing
    :class:`~core.states.inference_state.InferenceState` directly.

    Attributes:
        state (InferenceState): The underlying inference state this model
            wraps.
    """

    def __init__(self, state: InferenceState) -> None:
        """Initialize InferenceModel with a domain state object.

        Args:
            state (InferenceState): The inference state instance to wrap.
        """
        self.state = state

    def set_score_map(self, filename: str, score: float, score_map: np.ndarray) -> None:
        """Store a score map, the actual pred_score, and its classification label.

        Args:
            filename (str): Image filename used as the storage key.
            score (float): Normalized anomaly score [0, 1] from the model's PostProcessor.
            score_map (np.ndarray): 2-D heatmap array of anomaly scores.
        """
        self.state.set_score_map(filename, score, score_map)

    def get_score_map(self, filename: str) -> np.ndarray | None:
        """Return the stored score map for the given filename.

        Args:
            filename (str): Image filename to look up.

        Returns:
            np.ndarray | None: The heatmap array, or ``None`` if the filename
                has not been processed.
        """
        return self.state.get_score_map(filename)

    def is_processed(self, filename: str) -> bool:
        """Check whether an image has already been through inference.

        Args:
            filename (str): Image filename to check.

        Returns:
            bool: ``True`` if a score map exists for *filename*,
                ``False`` otherwise.
        """
        return self.state.is_processed(filename)

    def get_score(self, filename: str) -> float | None:
        """Return the normalized anomaly score [0,1] for the given image, or None.

        Args:
            filename (str): Image filename to look up.

        Returns:
            float | None: The pred_score, or ``None`` if not yet processed.
        """
        return self.state.scores.get(filename)

    def get_label(self, filename: str) -> str | None:
        """Return the classification label for the given image, or None.

        Args:
            filename (str): Image filename to look up.

        Returns:
            str | None: ``"ANOMALY"`` or ``"NORMAL"``, or ``None`` if not yet processed.
        """
        return self.state.labels.get(filename)

    def get_processed_count(self) -> int:
        """Return the number of images that have been processed.

        Returns:
            int: Count of image filenames for which a score map is stored.
        """
        return len(self.state.score_maps)

    def clear(self) -> None:
        """Clear all stored score maps and cached anomaly scores."""
        self.state.clear()

    # ------------------------------------------------------------------ #
    # Multi-model registry
    # ------------------------------------------------------------------ #

    def register_model(self, key: str, model_path: str, score_maps_file: str) -> None:
        """Add or update a model in the known-models registry.

        Args:
            key (str): Model key, derived from the ``.pt`` filename stem.
            model_path (str): Absolute path to the model checkpoint file.
            score_maps_file (str): Project-relative path to this model's
                cached heatmap NPZ.
        """
        self.state.register_model(key, model_path, score_maps_file)

    def switch_active_model(self, key: str) -> None:
        """Make *key* the active model (in-memory only — no disk I/O).

        Args:
            key (str): Model key to activate. Must already be registered via
                :meth:`register_model`.
        """
        self.state.switch_active_model(key)

    def get_active_model_key(self) -> str:
        """Return the currently active model's key, or empty string if none."""
        return self.state.active_model_key

    def get_known_models(self) -> dict:
        """Return the full model registry: ``{key: {"model_path", "score_maps_file"}}``."""
        return dict(self.state.known_models)

    def get_score_maps_file(self, key: str) -> str:
        """Return the project-relative NPZ path registered for *key*, or empty string."""
        entry = self.state.known_models.get(key)
        return entry.get("score_maps_file", "") if entry else ""

    def is_score_maps_dirty(self) -> bool:
        """Check whether the active model's heatmaps have unsaved changes."""
        return self.state.score_maps_dirty

    def clear_active_heatmaps(self) -> None:
        """Discard the active model's cached heatmap arrays and reset dirty.

        Loading a model's weights (Load New / Load Previous) is a deliberate
        request for fresh inference results, even if this key was already
        active with cached heatmaps from a previous session — those may be
        stale relative to whatever checkpoint was just loaded. Call this
        after switching so every image gets reprocessed instead of being
        skipped as "already done".
        """
        self.state.score_maps.clear()
        self.state.score_maps_dirty = False

    def load_score_maps_into_active(self, score_maps: dict) -> None:
        """Populate the active model's ``score_maps`` from a pre-loaded dict.

        Used after :meth:`switch_active_model` to hand off heatmap arrays
        read from disk (I/O itself is the caller's responsibility — this
        model has no Qt or disk dependencies).

        Args:
            score_maps (dict): ``{filename: np.ndarray}`` to install as the
                active model's heatmaps.
        """
        self.state.score_maps.update(score_maps)
        self.state.score_maps_dirty = False
