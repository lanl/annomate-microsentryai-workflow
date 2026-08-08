"""
ActiveToolSection — "Active Tool" tab body for the right activity bar.

Layout:
  Common (always visible)
      Stroke width slider -- applies no matter which drawing tool is active
  Tool Options (visible while a tool is selected; body swaps per tool)
      SAM  -> model variant combo + status label
      everything else -> "No tool-specific settings" until it gets one

New tools (Rectangle, Circle, ...) register their settings widget via
add_tool_options() -- nothing else here needs to change as they're added.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from views.annomate.sections._collapsible import _CollapsibleSection

# Maps human-readable combo label -> internal variant key used by SAMStrategy
_SAM_VARIANT_MAP = {
    "SAM2.1 Tiny": "sam2_t.pt",
    "SAM2.1 Small": "sam2_s.pt",
    "SAM2.1 Base+": "sam2_b.pt",
    "SAM2.1 Large": "sam2_l.pt",
}


class ActiveToolSection(QWidget):
    """Settings for whichever drawing tool is currently selected.

    Signals:
        thickness_changed (float): Stroke width slider moved.
        sam_variant_changed (str): SAM model variant combo changed.
    """

    thickness_changed = Signal(float)
    sam_variant_changed = Signal(str)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        common = _CollapsibleSection("Common", expanded=True)
        common.body_layout().setContentsMargins(0, 0, 0, 4)
        common.body_layout().addWidget(self._build_thickness_row())
        layout.addWidget(common)

        self._tool_section = _CollapsibleSection("Tool Options", expanded=True)
        self._tool_section.body_layout().setContentsMargins(0, 0, 0, 4)
        self._tool_stack = QStackedWidget()
        self._tool_section.body_layout().addWidget(self._tool_stack)
        layout.addWidget(self._tool_section)
        self._tool_section.setVisible(False)

        self._tool_pages: dict[str, int] = {}
        self._empty_page = self._build_empty_page()
        self._tool_stack.addWidget(self._empty_page)

        layout.addStretch()

        self.add_tool_options("sam_bbox", self._build_sam_options())

    # ------------------------------------------------------------------ #
    # Common: stroke width
    # ------------------------------------------------------------------ #

    def _build_thickness_row(self) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)

        h.addWidget(QLabel("Stroke width"))

        self.slider_thickness = QSlider(Qt.Horizontal)
        self.slider_thickness.setRange(1, 40)
        self.slider_thickness.setValue(8)  # 8 x 0.25 = 2.00 px default
        self.slider_thickness.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_thickness.setTickInterval(4)
        h.addWidget(self.slider_thickness, stretch=1)

        self.lbl_thickness = QLabel("2.00 px")
        self.lbl_thickness.setFixedWidth(55)
        h.addWidget(self.lbl_thickness)

        self.slider_thickness.valueChanged.connect(self._on_thickness_slider_changed)
        return row

    def _on_thickness_slider_changed(self, value: int) -> None:
        thickness = value * 0.25
        self.lbl_thickness.setText(f"{thickness:.2f} px")
        self.thickness_changed.emit(thickness)

    def set_thickness(self, value: float) -> None:
        """Sync the slider/label to *value* without re-emitting thickness_changed."""
        self.slider_thickness.blockSignals(True)
        self.slider_thickness.setValue(int(value * 4))
        self.lbl_thickness.setText(f"{value:.2f} px")
        self.slider_thickness.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Tool-specific pages
    # ------------------------------------------------------------------ #

    def add_tool_options(self, tool_key: str, widget: QWidget) -> None:
        """Register *widget* as the Tool Options body shown when *tool_key* is active."""
        index = self._tool_stack.addWidget(widget)
        self._tool_pages[tool_key] = index

    def set_active_tool(self, tool_key: str) -> None:
        """Show the Tool Options page for *tool_key* ("" = no tool selected)."""
        self._tool_section.setVisible(bool(tool_key))
        if not tool_key:
            return
        index = self._tool_pages.get(tool_key)
        self._tool_stack.setCurrentWidget(
            self._tool_stack.widget(index) if index is not None else self._empty_page
        )

    def _build_empty_page(self) -> QWidget:
        lbl = QLabel("No tool-specific settings")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: grey;")
        lbl.setContentsMargins(0, 4, 0, 4)
        return lbl

    # ------------------------------------------------------------------ #
    # SAM options page
    # ------------------------------------------------------------------ #

    def _build_sam_options(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        variant_row = QHBoxLayout()
        variant_row.addWidget(QLabel("Variant:"))
        self.sam_variant_combo = QComboBox()
        self.sam_variant_combo.addItems(list(_SAM_VARIANT_MAP.keys()))
        self.sam_variant_combo.setToolTip(
            "SAM 2 model size — tiny is fastest, large is most accurate.\n"
            "Weights are downloaded to sam_weights/ on first use."
        )
        variant_row.addWidget(self.sam_variant_combo, stretch=1)
        v.addLayout(variant_row)

        self.sam_status_lbl = QLabel("Model: not loaded")
        self.sam_status_lbl.setStyleSheet("color: grey; font-style: italic;")
        v.addWidget(self.sam_status_lbl)

        self.sam_variant_combo.currentTextChanged.connect(self._on_sam_combo_changed)
        return page

    def _on_sam_combo_changed(self, display_name: str) -> None:
        self.sam_variant_changed.emit(_SAM_VARIANT_MAP.get(display_name, "sam2_t.pt"))

    def current_sam_variant(self) -> str:
        """Return the internal variant key for the currently selected SAM model."""
        return _SAM_VARIANT_MAP.get(self.sam_variant_combo.currentText(), "sam2_t.pt")

    def sam_variant_display_name(self) -> str:
        return self.sam_variant_combo.currentText()

    def set_sam_status(self, text: str, color: str = "grey", italic: bool = True) -> None:
        style = f"color: {color};"
        if italic:
            style += " font-style: italic;"
        self.sam_status_lbl.setText(text)
        self.sam_status_lbl.setStyleSheet(style)
