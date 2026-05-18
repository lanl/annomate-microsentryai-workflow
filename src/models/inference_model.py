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
