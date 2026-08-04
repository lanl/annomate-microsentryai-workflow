from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout

from views.annomate.sections import DataNavigatorSection


class LeftPanel(QWidget):
    """Left panel hosting the dataset navigator, to the left of the tool palette.

    Signals:
        image_selected (int): Forwarded from DataNavigatorSection.
        prev_requested (): Forwarded from DataNavigatorSection.
        next_requested (): Forwarded from DataNavigatorSection.
        annotation_selected (int): Forwarded from DataNavigatorSection.
    """

    image_selected = Signal(int)
    prev_requested = Signal()
    next_requested = Signal()
    annotation_selected = Signal(int)

    def __init__(
        self,
        dataset_model,
        inference_model=None,
        calibration_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        # Right border separating the panel from the tool palette
        self.setStyleSheet("LeftPanel { border-right: 1px solid palette(mid); }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._title_lbl = QLabel("Dataset Navigator")
        self._title_lbl.setStyleSheet(
            "font-weight: bold; padding: 6px 8px 2px 8px;"
        )
        outer.addWidget(self._title_lbl)

        self.navigator = DataNavigatorSection(
            dataset_model, inference_model, calibration_model
        )
        self.navigator.image_selected.connect(self.image_selected)
        self.navigator.prev_requested.connect(self.prev_requested)
        self.navigator.next_requested.connect(self.next_requested)
        self.navigator.annotation_selected.connect(self.annotation_selected)

        outer.addWidget(self.navigator, stretch=1)

    def select_row(self, row: int) -> None:
        """Silently highlight *row* in the navigator list."""
        self.navigator.select_row(row)

    def set_counter(self, current: int, total: int) -> None:
        """Update the image position counter in the navigator."""
        self.navigator.set_counter(current, total)

    def navigator_adjacent_source_row(self, current_source_row: int, step: int) -> int:
        """Return the navigator-adjacent source row in current visible order."""
        return self.navigator.adjacent_source_row(current_source_row, step)

    def navigator_set_inference(self, row: int, score: float) -> None:
        self.navigator.set_row_inference(row, score)

    def navigator_set_microsentry_mode(self, enabled: bool) -> None:
        self.navigator.set_microsentry_mode(enabled)

    def navigator_enable_inference_columns(self) -> None:
        self.navigator.enable_inference_columns()

    def navigator_select_annotation(self, idx: int) -> None:
        self.navigator.select_annotation(idx)

    def navigator_set_annotation_mode(self, mode: str) -> None:
        self.navigator.set_annotation_mode(mode)

    def navigator_header(self) -> QWidget:
        return self._title_lbl
