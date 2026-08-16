from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QWidget,
    QVBoxLayout,
)

from views.annomate.sections import DataNavigatorSection
from views.icons import material_icon


_COLLAPSED_WIDTH = 56


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class _CollapsedNavigatorRail(QWidget):
    """Toolbar-width, at-a-glance presentation of dataset navigation state."""

    expand_requested = Signal()
    prev_requested = Signal()
    next_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignTop)

        self._btn_expand = self._button(
            "keyboard_double_arrow_right", "Expand Dataset Navigator"
        )
        self._btn_expand.clicked.connect(self.expand_requested)
        layout.addWidget(self._btn_expand)
        layout.addWidget(_divider())

        self._btn_prev = self._button("chevron_left", "Previous image (A)")
        self._btn_prev.clicked.connect(self.prev_requested)
        layout.addWidget(self._btn_prev)

        self._btn_next = self._button("chevron_right", "Next image (D)")
        self._btn_next.clicked.connect(self.next_requested)
        layout.addWidget(self._btn_next)
        layout.addWidget(_divider())

        self._counter_lbl = QLabel("—/—")
        self._counter_lbl.setAlignment(Qt.AlignCenter)
        self._counter_lbl.setStyleSheet("font-size: 10px; color: black;")
        layout.addWidget(self._counter_lbl)
        layout.addWidget(_divider())

        self._undecided_glyph, self._undecided_count = self._add_status(
            layout, "○", "Undecided", "color: palette(mid); font-size: 18px;"
        )
        self._reviewed_glyph, self._reviewed_count = self._add_status(
            layout, "●", "Reviewed", "color: #45b85a; font-size: 16px;"
        )
        self._incomplete_glyph, self._incomplete_count = self._add_status(
            layout, "!", "Incomplete", "color: #e68619; font-weight: bold; font-size: 16px;"
        )

    def _button(self, icon_name: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setIcon(material_icon(icon_name, size=18, color="black"))
        button.setToolTip(tooltip)
        button.setFixedSize(44, 36)
        button.setAutoRaise(True)
        return button

    def _add_status(self, layout, glyph: str, name: str, style: str):
        group = QWidget()
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 1, 0, 3)
        group_layout.setSpacing(0)
        glyph_lbl = QLabel(glyph)
        glyph_lbl.setAlignment(Qt.AlignCenter)
        glyph_lbl.setStyleSheet(style)
        count_lbl = QLabel("0")
        count_lbl.setAlignment(Qt.AlignCenter)
        count_lbl.setStyleSheet("color: black;")
        group_layout.addWidget(glyph_lbl)
        group_layout.addWidget(count_lbl)
        group.setToolTip(f"0 {name.lower()} images")
        layout.addWidget(group)
        return glyph_lbl, count_lbl

    def set_counter(self, current: int, total: int) -> None:
        has_position = total > 0 and current >= 0
        text = f"{current + 1}/{total}" if has_position else "—/—"
        self._counter_lbl.setText(text)
        self._counter_lbl.setToolTip(
            f"Image {current + 1} of {total}"
            if has_position
            else f"{total} images loaded" if total > 0 else "No images loaded"
        )

    def set_counts(self, undecided: int, reviewed: int, incomplete: int) -> None:
        values = (
            (self._undecided_glyph, self._undecided_count, undecided, "undecided"),
            (self._reviewed_glyph, self._reviewed_count, reviewed, "reviewed"),
            (self._incomplete_glyph, self._incomplete_count, incomplete, "incomplete"),
        )
        for glyph, label, count, name in values:
            label.setText(str(count))
            glyph.parentWidget().setToolTip(f"{count} {name} images")

    def set_has_data(self, has_data: bool) -> None:
        """Grey out expand/prev/next until a dataset is actually loaded."""
        self._btn_expand.setEnabled(has_data)
        self._btn_prev.setEnabled(has_data)
        self._btn_next.setEnabled(has_data)


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
    collapsed_changed = Signal(bool)

    def __init__(
        self,
        dataset_model,
        inference_model=None,
        calibration_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        # Right border separating the panel from the tool palette
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("LeftPanel { border-right: 1px solid palette(mid); }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._expanded_view = QWidget()
        expanded_layout = QVBoxLayout(self._expanded_view)
        expanded_layout.setContentsMargins(0, 0, 0, 0)
        expanded_layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 4, 0)
        header_layout.setSpacing(2)
        self._title_lbl = QLabel("Dataset Navigator")
        self._title_lbl.setStyleSheet(
            "font-weight: bold; padding: 6px 8px 2px 8px; color: black;"
        )
        header_layout.addWidget(self._title_lbl, stretch=1)
        self._collapse_btn = QToolButton()
        self._collapse_btn.setIcon(
            material_icon("keyboard_double_arrow_left", size=18, color="black")
        )
        self._collapse_btn.setToolTip("Collapse Dataset Navigator")
        self._collapse_btn.setAutoRaise(True)
        self._collapse_btn.clicked.connect(lambda: self.set_collapsed(True))
        header_layout.addWidget(self._collapse_btn)
        expanded_layout.addWidget(header)

        self.navigator = DataNavigatorSection(
            dataset_model, inference_model, calibration_model
        )
        self.navigator.image_selected.connect(self.image_selected)
        self.navigator.prev_requested.connect(self.prev_requested)
        self.navigator.next_requested.connect(self.next_requested)
        self.navigator.annotation_selected.connect(self.annotation_selected)
        self.navigator.state_counts_changed.connect(self._on_state_counts_changed)

        expanded_layout.addWidget(self.navigator, stretch=1)

        self._collapsed_rail = _CollapsedNavigatorRail()
        self._collapsed_rail.expand_requested.connect(
            lambda: self.set_collapsed(False)
        )
        self._collapsed_rail.prev_requested.connect(self.prev_requested)
        self._collapsed_rail.next_requested.connect(self.next_requested)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._expanded_view)
        self._stack.addWidget(self._collapsed_rail)
        outer.addWidget(self._stack, stretch=1)
        self._collapsed = False
        self._on_state_counts_changed(
            int(self.navigator._lbl_count_undecided.text()),
            int(self.navigator._lbl_count_reviewed.text()),
            int(self.navigator._lbl_count_incomplete.text()),
        )

        self._dataset_model = dataset_model
        dataset_model.modelReset.connect(self._on_dataset_reset)
        self._on_dataset_reset()

    def set_collapsed(self, collapsed: bool) -> None:
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed
        self._stack.setCurrentWidget(
            self._collapsed_rail if collapsed else self._expanded_view
        )
        if collapsed:
            self.setFixedWidth(_COLLAPSED_WIDTH)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        else:
            self.setMinimumWidth(160)
            self.setMaximumWidth(16777215)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.collapsed_changed.emit(collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def _on_dataset_reset(self) -> None:
        """Collapsed with greyed-out controls until a dataset loads, then expanded."""
        has_data = self._dataset_model.rowCount() > 0
        self._collapsed_rail.set_has_data(has_data)
        if not has_data:
            self._collapsed_rail.set_counter(-1, 0)
        self.set_collapsed(not has_data)

    def _on_state_counts_changed(
        self, undecided: int, reviewed: int, incomplete: int
    ) -> None:
        self._collapsed_rail.set_counts(undecided, reviewed, incomplete)

    def select_row(self, row: int) -> None:
        """Silently highlight *row* in the navigator list."""
        self.navigator.select_row(row)

    def set_counter(self, current: int, total: int) -> None:
        """Update the image position counter in the navigator."""
        self.navigator.set_counter(current, total)
        self._collapsed_rail.set_counter(current, total)

    def navigator_adjacent_source_row(self, current_source_row: int, step: int) -> int:
        """Return the navigator-adjacent source row in current visible order."""
        return self.navigator.adjacent_source_row(current_source_row, step)

    def navigator_set_inference(self, row: int, score: float) -> None:
        self.navigator.set_row_inference(row, score)

    def navigator_refresh_inference(self) -> None:
        self.navigator.refresh_inference()

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
