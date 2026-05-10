"""
ValidationState — pure Python state container for the Validation pane.

Holds the six directory/file paths needed across the two-step workflow.
Zero Qt dependencies.
"""

import os


class ValidationState:
    """Pure Python state container for the Validation pane.

    Holds the six directory/file paths needed across the two-step
    validation workflow. Contains zero Qt dependencies.

    Attributes:
        poly_path (str): Path to the polygon annotation source directory.
        json_path (str): Path to the JSON annotations file.
        mask_out_path (str): Output directory for generated binary masks.
        gt_path (str): Path to the ground truth masks directory.
        pred_path (str): Path to the prediction masks directory.
        eval_out_path (str): Output directory for evaluation results,
            defaulting to an ``evaluation_results`` folder in the cwd.
    """

    def __init__(self) -> None:
        """Initialize ValidationState with empty paths and a default eval output dir."""
        self.poly_path: str = ""
        self.json_path: str = ""
        self.mask_out_path: str = ""
        self.gt_path: str = ""
        self.pred_path: str = ""
        self.eval_out_path: str = os.path.join(os.getcwd(), "evaluation_results")

    def clear(self) -> None:
        """Reset all path fields except ``eval_out_path`` to empty strings.

        ``eval_out_path`` is intentionally preserved so the user's chosen
        output directory survives a workflow reset.
        """
        self.poly_path = ""
        self.json_path = ""
        self.mask_out_path = ""
        self.gt_path = ""
        self.pred_path = ""
