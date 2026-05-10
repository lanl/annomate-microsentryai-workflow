"""ValidationWindow — view for the Validation pane.

See MVC.md § Architecture Rules for the full layer contract.
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QGroupBox,
    QScrollArea,
    QFrame,
)

from models.validation_model import ValidationModel
from controllers.validation_controller import ValidationController

logger = logging.getLogger("Validation.Window")


class ValidationWindow(QWidget):
    """Mask evaluation UI for comparing GT masks against model predictions.

    Owns all file dialogs. Delegates computation to
    :class:`~controllers.validation_controller.ValidationController` via a
    worker thread. Reads path configuration via the model's query API.

    Attributes:
        model (ValidationModel): Domain model holding path configuration.
        controller (ValidationController): Controller that owns the evaluation
            worker thread.
        _eval_worker: Active evaluation worker, or ``None``.
    """

    def __init__(
        self,
        model: ValidationModel,
        controller: ValidationController,
        parent=None,
    ) -> None:
        """Initialize ValidationWindow and build the UI.

        Args:
            model (ValidationModel): Validation domain model.
            controller (ValidationController): Validation controller.
            parent: Optional Qt parent widget. Defaults to ``None``.
        """
        super().__init__(parent)
        self.model = model
        self.controller = controller
        self._eval_worker = None

        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the validation UI with progress bar and results feed."""
        root = QVBoxLayout(self)

        # ---- Evaluation ----
        grp_eval = QGroupBox("Run Evaluation")
        eval_layout = QVBoxLayout(grp_eval)

        self.lbl_gt, r4 = self._make_row("Select GT Masks:", self._select_gt)
        self.lbl_pred, r5 = self._make_row("Select Predictions:", self._select_pred)
        eval_layout.addLayout(r4)
        eval_layout.addLayout(r5)

        self.btn_run = QPushButton("Run Comparison")
        self.btn_run.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; height: 35px;"
        )
        self.btn_run.clicked.connect(self._run_evaluation)
        eval_layout.addWidget(self.btn_run)
        root.addWidget(grp_eval)

        # ---- Progress ----
        self.pbar = QProgressBar()
        root.addWidget(self.pbar)

        # ---- Results feed ----
        root.addWidget(QLabel("Evaluation Feed:"))
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.results_container)
        root.addWidget(self.scroll_area, stretch=1)

    @staticmethod
    def _make_row(
        button_text: str,
        callback,
        tooltip: str = "",
    ):
        """Create a labelled file-selection row widget.

        Args:
            button_text (str): Text displayed on the picker button.
            callback: Callable connected to the button's ``clicked`` signal.
            tooltip (str): Optional tooltip shown on the button. Defaults to
                ``""``.

        Returns:
            Tuple[QLabel, QHBoxLayout]: The path-display label and the
                layout containing the button and label.
        """
        row = QHBoxLayout()
        btn = QPushButton(button_text)
        btn.setFixedWidth(150)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.clicked.connect(callback)

        lbl = QLabel("Not selected")
        lbl.setStyleSheet("color: gray;")
        lbl.setWordWrap(True)

        row.addWidget(btn)
        row.addWidget(lbl, 1)
        return lbl, row

    # ------------------------------------------------------------------
    # Dialog slots — the only place QFileDialog lives
    # ------------------------------------------------------------------

    def _select_gt(self) -> None:
        """Open a folder picker and store the selected ground-truth masks path in the model."""
        p = QFileDialog.getExistingDirectory(self, "Select Ground Truth Masks Folder")
        if p:
            self.model.set_gt_path(p)
            self.lbl_gt.setText(p)
            self.lbl_gt.setStyleSheet("color: black;")

    def _select_pred(self) -> None:
        """Open a folder picker and store the selected predictions path in the model."""
        p = QFileDialog.getExistingDirectory(self, "Select Predictions Folder")
        if p:
            self.model.set_pred_path(p)
            self.lbl_pred.setText(p)
            self.lbl_pred.setStyleSheet("color: black;")

    # ------------------------------------------------------------------
    # Worker launch
    # ------------------------------------------------------------------

    def _run_evaluation(self) -> None:
        """Validate inputs, launch the evaluation worker, and wire its signals."""
        if not self.model.can_evaluate():
            QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please select a ground truth folder and a predictions folder.",
            )
            return

        self._clear_results()
        self._set_ui_state(False)

        try:
            worker = self.controller.start_evaluation()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            self._set_ui_state(True)
            return

        worker.progress.connect(self.pbar.setValue)
        worker.log_message.connect(self._add_log_text)
        worker.match_found.connect(self._add_result_card)
        worker.finished.connect(lambda: self._set_ui_state(True))
        self._eval_worker = worker
        worker.start()

    # ------------------------------------------------------------------
    # Results feed helpers
    # ------------------------------------------------------------------

    def _add_log_text(self, text: str) -> None:
        """Append a monospace log message label to the results feed.

        Args:
            text (str): Log message to display.
        """
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #333; font-family: monospace;")
        self.results_layout.addWidget(lbl)
        self._scroll_to_bottom()

    def _add_result_card(self, image_path: str, text: str, iou: float) -> None:
        """Append a styled result card with an image and IoU summary to the feed.

        The card border is green when ``iou > 50`` and red otherwise.

        Args:
            image_path (str): Absolute path to the comparison visualization
                image to display inside the card.
            text (str): Summary text shown as the card title.
            iou (float): IoU score used to select the border color.
        """
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        color = "#4CAF50" if iou > 50 else "#F44336"
        card.setStyleSheet(
            f"background-color: white; border: 2px solid {color}; "
            "border-radius: 5px; margin-bottom: 10px;"
        )

        layout = QVBoxLayout(card)

        title = QLabel(text)
        title.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")
        layout.addWidget(title)

        img_lbl = QLabel()
        pix = QPixmap(image_path)
        if not pix.isNull():
            img_lbl.setPixmap(pix.scaledToWidth(400, Qt.SmoothTransformation))
        layout.addWidget(img_lbl)

        self.results_layout.addWidget(card)
        self._scroll_to_bottom()

    def _clear_results(self) -> None:
        """Remove and delete all widgets from the results feed layout."""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _scroll_to_bottom(self) -> None:
        """Scroll the results feed scroll area to its maximum vertical position."""
        sb = self.scroll_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_ui_state(self, enabled: bool) -> None:
        """Enable or disable the Run button.

        Called with ``False`` before starting a worker and with ``True``
        from the worker's ``finished`` signal.

        Args:
            enabled (bool): ``True`` to enable button; ``False`` to disable.
        """
        self.btn_run.setEnabled(enabled)
