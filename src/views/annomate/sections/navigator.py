from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
)

from ._shared import _dot, _COLOR_REVIEWED, _COLOR_IN_REVIEW

# ── Column-width constants ────────────────────────────────────────────────────
# Layout order:  [status col] [img id …stretch…] [annots] [decision]
#
# _STATUS_COL_W — fixed width of the colored status-dot column.
# _DOT_W        — diameter of the status dot (must be ≤ _STATUS_COL_W).
# _ANNOTS_W     — fixed width for the annotation count column.
# _DECISION_W   — fixed width for the Accept / Reject decision column.
# _IMG_MIN_W    — minimum width of the image-id text column (stretch=1).
# ─────────────────────────────────────────────────────────────────────────────
_STATUS_COL_W = 28
_DOT_W = 10
_ANNOTS_W = 46
_DECISION_W = 58
_IMG_MIN_W = 50


def _set_decision_label(lbl: QLabel, decision) -> None:
    if decision == "accept":
        lbl.setText("Accept")
        lbl.setStyleSheet("color: #4caf50; font-weight: bold;")
    elif decision == "reject":
        lbl.setText("Reject")
        lbl.setStyleSheet("color: #f44336; font-weight: bold;")
    else:
        lbl.setText("")
        lbl.setStyleSheet("")


class _NavigatorRow(QWidget):
    """Single image row: status dot | filename | annot count | decision."""

    row_clicked = Signal(int)

    def __init__(
        self,
        idx: int,
        filename_stem: str,
        dot_color: str,
        annot_count: int,
        decision,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._idx = idx
        self._original_palette = QPalette(self.palette())
        self.setCursor(Qt.PointingHandCursor)

        h = QHBoxLayout(self)
        h.setContentsMargins(6, 1, 6, 1)
        h.setSpacing(8)

        # Status dot column
        dot_col = QWidget()
        dot_col.setFixedWidth(_STATUS_COL_W)
        dot_h = QHBoxLayout(dot_col)
        dot_h.setContentsMargins(0, 0, 0, 0)
        self._dot = QLabel()
        self._dot.setFixedSize(_DOT_W, _DOT_W)
        self._dot.setStyleSheet(
            f"background-color: {dot_color}; border-radius: {_DOT_W // 2}px;"
        )
        dot_h.addWidget(self._dot, alignment=Qt.AlignCenter)
        h.addWidget(dot_col)

        # Image ID (filename stem)
        self._name_lbl = QLabel(filename_stem)
        self._name_lbl.setMinimumWidth(_IMG_MIN_W)
        h.addWidget(self._name_lbl, stretch=1)

        # Annotation count
        self._annot_lbl = QLabel(str(annot_count) if annot_count > 0 else "")
        self._annot_lbl.setFixedWidth(_ANNOTS_W)
        self._annot_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self._annot_lbl)

        # Decision
        self._decision_lbl = QLabel()
        self._decision_lbl.setFixedWidth(_DECISION_W)
        self._decision_lbl.setAlignment(Qt.AlignCenter)
        _set_decision_label(self._decision_lbl, decision)
        h.addWidget(self._decision_lbl)

    # ── Update helpers ────────────────────────────────────────────────────────

    def update_dot(self, color: str) -> None:
        self._dot.setStyleSheet(
            f"background-color: {color}; border-radius: {_DOT_W // 2}px;"
        )

    def update_annot(self, count: int) -> None:
        self._annot_lbl.setText(str(count) if count > 0 else "")

    def update_decision(self, decision) -> None:
        _set_decision_label(self._decision_lbl, decision)

    def set_selected(self, selected: bool) -> None:
        if selected:
            p = QPalette(self.palette())
            p.setColor(QPalette.Window, self.palette().color(QPalette.Highlight))
            p.setColor(
                QPalette.WindowText, self.palette().color(QPalette.HighlightedText)
            )
            self.setAutoFillBackground(True)
            self.setPalette(p)
        else:
            self.setAutoFillBackground(False)
            self.setPalette(self._original_palette)

    def mousePressEvent(self, event) -> None:
        self.row_clicked.emit(self._idx)
        super().mousePressEvent(event)


class DataNavigatorSection(QWidget):
    """Image list with proper column headers: Status | Img ID | Annots | Decision.

    The In Review / Reviewed legend lives inline next to the image count.
    Row selection highlights via QPalette.Highlight (OS-adaptive).

    Signals:
        image_selected (int): Emitted when the user clicks a row.
        prev_requested (): Emitted when Prev button is clicked.
        next_requested (): Emitted when Next button is clicked.
    """

    image_selected = Signal(int)
    prev_requested = Signal()
    next_requested = Signal()

    def __init__(
        self, dataset_model, inference_model=None, parent: QWidget = None
    ) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self._row_widgets: list = []
        self._selected_row: int = -1
        self._init_ui()
        self.dataset_model.modelReset.connect(self._rebuild_list)
        self.dataset_model.dataChanged.connect(self._on_data_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Top nav row: Prev / Next / count / legend ─────────────────────────
        nav_row = QWidget()
        nav_h = QHBoxLayout(nav_row)
        nav_h.setContentsMargins(0, 0, 0, 0)
        nav_h.setSpacing(4)

        self._btn_prev = QToolButton()
        self._btn_prev.setText("‹ Prev")
        self._btn_prev.setToolTip("Previous image")
        self._btn_prev.clicked.connect(self.prev_requested)
        self._btn_prev.setVisible(False)
        nav_h.addWidget(self._btn_prev)

        self._btn_next = QToolButton()
        self._btn_next.setText("Next ›")
        self._btn_next.setToolTip("Next image")
        self._btn_next.clicked.connect(self.next_requested)
        self._btn_next.setVisible(False)
        nav_h.addWidget(self._btn_next)

        self._lbl_counter = QLabel("No images loaded")
        nav_h.addWidget(self._lbl_counter)

        nav_h.addStretch()

        # Inline legend
        nav_h.addWidget(_dot(_COLOR_IN_REVIEW))
        nav_h.addWidget(QLabel("In Review"))
        nav_h.addSpacing(4)
        nav_h.addWidget(_dot(_COLOR_REVIEWED))
        nav_h.addWidget(QLabel("Reviewed"))

        layout.addWidget(nav_row)

        # ── Column header + separator ─────────────────────────────────────────
        self._header = QWidget()
        header_h = QHBoxLayout(self._header)
        header_h.setContentsMargins(6, 2, 6, 2)
        header_h.setSpacing(8)

        status_spacer = QWidget()
        status_spacer.setFixedWidth(_STATUS_COL_W)
        header_h.addWidget(status_spacer)

        lbl_img = QLabel("Img ID")
        lbl_img.setMinimumWidth(_IMG_MIN_W)
        lbl_img.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_img, stretch=1)

        lbl_annots = QLabel("Annots")
        lbl_annots.setFixedWidth(_ANNOTS_W)
        lbl_annots.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_annots.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_annots)

        lbl_decision = QLabel("Decision")
        lbl_decision.setFixedWidth(_DECISION_W)
        lbl_decision.setAlignment(Qt.AlignCenter)
        lbl_decision.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_decision)

        self._header.setVisible(False)
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self._sep = sep
        self._sep.setVisible(False)
        layout.addWidget(self._sep)

        # ── Scrollable row area ───────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setMinimumHeight(80)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll.setVisible(False)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.addStretch()

        self._scroll.setWidget(self._rows_container)
        layout.addWidget(self._scroll)

    # ── List management ───────────────────────────────────────────────────────

    def _rebuild_list(self) -> None:
        # Clear existing rows
        while self._rows_layout.count() > 1:  # keep the trailing stretch
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._row_widgets.clear()
        self._selected_row = -1

        total = self.dataset_model.rowCount()
        has_images = total > 0
        self._btn_prev.setVisible(has_images)
        self._btn_next.setVisible(has_images)
        self._header.setVisible(has_images)
        self._sep.setVisible(has_images)
        self._scroll.setVisible(has_images)

        if not has_images:
            self._lbl_counter.setText("No images loaded")
            return

        self._lbl_counter.setText(f"{total} image{'s' if total != 1 else ''} loaded")

        for row in range(total):
            filename = self.dataset_model.get_image_filename(row)
            dot_color = (
                _COLOR_REVIEWED
                if self.dataset_model.is_reviewed(row)
                else _COLOR_IN_REVIEW
            )
            annot_count = self.dataset_model.get_annotation_count(row)
            decision = self.dataset_model.get_review_decision(row)

            row_w = _NavigatorRow(
                row,
                Path(filename).stem,
                dot_color,
                annot_count,
                decision,
                self._rows_container,
            )
            row_w.row_clicked.connect(self._on_row_clicked)
            self._row_widgets.append(row_w)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row_w)

    def _on_data_changed(self, top_left, bottom_right, roles=None) -> None:
        for row in range(top_left.row(), bottom_right.row() + 1):
            if row < len(self._row_widgets):
                rw = self._row_widgets[row]
                color = (
                    _COLOR_REVIEWED
                    if self.dataset_model.is_reviewed(row)
                    else _COLOR_IN_REVIEW
                )
                rw.update_dot(color)
                rw.update_annot(self.dataset_model.get_annotation_count(row))
                rw.update_decision(self.dataset_model.get_review_decision(row))

    def _on_row_clicked(self, row: int) -> None:
        self._set_selected(row)
        self.image_selected.emit(row)

    def _set_selected(self, row: int) -> None:
        if self._selected_row >= 0 and self._selected_row < len(self._row_widgets):
            self._row_widgets[self._selected_row].set_selected(False)
        self._selected_row = row
        if row >= 0 and row < len(self._row_widgets):
            self._row_widgets[row].set_selected(True)

    # ── Public interface ──────────────────────────────────────────────────────

    def select_row(self, row: int) -> None:
        """Silently highlight *row* without emitting image_selected."""
        self._set_selected(row)
        if row >= 0 and row < len(self._row_widgets):
            self._scroll.ensureWidgetVisible(self._row_widgets[row])

    def set_counter(self, current: int, total: int) -> None:
        if total > 0:
            self._lbl_counter.setText(f"{current + 1} / {total}")

    def set_row_processed(self, row: int, processed: bool) -> None:
        """No-op — AI processing is shown in the status bar, not per row."""

    def refresh_all_processed(self) -> None:
        """No-op — AI processing is shown in the status bar, not per row."""
