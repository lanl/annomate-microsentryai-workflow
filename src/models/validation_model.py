"""
ValidationModel — pure Python model for the Validation pane.

Wraps ValidationState with a typed query/command API.
Views must use this API instead of accessing ValidationState directly.
No Qt dependencies.
"""

from core.states.validation_state import ValidationState


class ValidationModel:
    """Pure Python model for the Validation pane.

    Wraps :class:`~core.states.validation_state.ValidationState` with a
    typed query/command API. Views must use this API instead of accessing
    :class:`~core.states.validation_state.ValidationState` directly.
    Contains no Qt dependencies.

    Attributes:
        state (ValidationState): The underlying validation state this model
            wraps.
    """

    def __init__(self, state: ValidationState) -> None:
        """Initialize ValidationModel with a domain state object.

        Args:
            state (ValidationState): The validation state instance to wrap.
        """
        self.state = state

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def set_poly_path(self, path: str) -> None:
        """Set the polygon annotation source directory path.

        Args:
            path (str): Absolute path to the image folder used for mask
                generation.
        """
        self.state.poly_path = path

    def set_json_path(self, path: str) -> None:
        """Set the JSON annotation file path.

        Args:
            path (str): Absolute path to the JSON annotation file.
        """
        self.state.json_path = path

    def set_mask_out_path(self, path: str) -> None:
        """Set the binary mask output directory path.

        As a convenience, also seeds the GT evaluation input path if it has
        not yet been set, since generated masks are commonly used directly
        as ground-truth inputs in Step 2.

        Args:
            path (str): Absolute path to the mask output directory.
        """
        self.state.mask_out_path = path
        # Convenience: mask output automatically seeds the GT eval input
        if not self.state.gt_path:
            self.state.gt_path = path

    def set_gt_path(self, path: str) -> None:
        """Set the ground truth masks directory path.

        Args:
            path (str): Absolute path to the ground truth mask directory.
        """
        self.state.gt_path = path

    def set_pred_path(self, path: str) -> None:
        """Set the prediction masks directory path.

        Args:
            path (str): Absolute path to the prediction mask directory.
        """
        self.state.pred_path = path

    def set_eval_out_path(self, path: str) -> None:
        """Set the evaluation results output directory path.

        Args:
            path (str): Absolute path to the evaluation output directory.
        """
        self.state.eval_out_path = path

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_poly_path(self) -> str:
        """Return the polygon annotation source directory path.

        Returns:
            str: Current ``poly_path`` value, or an empty string if unset.
        """
        return self.state.poly_path

    def get_json_path(self) -> str:
        """Return the JSON annotation file path.

        Returns:
            str: Current ``json_path`` value, or an empty string if unset.
        """
        return self.state.json_path

    def get_mask_out_path(self) -> str:
        """Return the binary mask output directory path.

        Returns:
            str: Current ``mask_out_path`` value, or an empty string if unset.
        """
        return self.state.mask_out_path

    def get_gt_path(self) -> str:
        """Return the ground truth masks directory path.

        Returns:
            str: Current ``gt_path`` value, or an empty string if unset.
        """
        return self.state.gt_path

    def get_pred_path(self) -> str:
        """Return the prediction masks directory path.

        Returns:
            str: Current ``pred_path`` value, or an empty string if unset.
        """
        return self.state.pred_path

    def get_eval_out_path(self) -> str:
        """Return the evaluation results output directory path.

        Returns:
            str: Current ``eval_out_path`` value; defaults to a
                ``evaluation_results`` subfolder of the cwd when unset.
        """
        return self.state.eval_out_path

    def can_generate(self) -> bool:
        """Return whether all Step 1 (mask generation) inputs are configured.

        Returns:
            bool: ``True`` when ``poly_path``, ``json_path``, and
                ``mask_out_path`` are all non-empty; ``False`` otherwise.
        """
        return bool(
            self.state.poly_path and self.state.json_path and self.state.mask_out_path
        )

    def can_evaluate(self) -> bool:
        """Return whether all Step 2 (evaluation) inputs are configured.

        Returns:
            bool: ``True`` when both ``gt_path`` and ``pred_path`` are
                non-empty; ``False`` otherwise.
        """
        return bool(self.state.gt_path and self.state.pred_path)
