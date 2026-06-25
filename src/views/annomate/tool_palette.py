"""
ToolPalette — left tool column for the AnnoMate main window.

Layout (top to bottom):
  ⬠  Polygon tool
  ◢  Brush thickness popup
  ── divider ──
  ✦  SAM Segment tool
  ⚙  SAM Options popup
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QToolButton,
    QButtonGroup,
    QSizePolicy,
    QMenu,
    QWidgetAction,
    QSlider,
    QLabel,
    QHBoxLayout,
    QWidget,
    QComboBox,
)
from PySide6.QtGui import QFont

# Maps human-readable combo label → internal variant key used by SAMStrategy
_SAM_VARIANT_MAP = {
    "SAM2.1 Tiny": "sam2_t.pt",
    "SAM2.1 Small": "sam2_s.pt",
    "SAM2.1 Base+": "sam2_b.pt",
    "SAM2.1 Large": "sam2_l.pt",
}

_BTN_W = 44
_BTN_H = 40


class ToolPalette(QFrame):
    """Single-column tool panel.

    Signals:
        tool_selected (str): Emitted with the tool name when a button is pressed.
        thickness_changed (float): Emitted when the brush thickness slider moves.
        sam_variant_changed (str): Emitted when the SAM model variant combo changes.
    """

    tool_selected = Signal(str)
    thickness_changed = Signal(float)
    sam_variant_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setFixedWidth(56)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._btn_tool: dict = {}
        self._active_tool: str = ""
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignTop)

        font = QFont()
        font.setPointSize(18)

        # ------------------------------------------------------------------ #
        # Polygon tool
        # ------------------------------------------------------------------ #
        poly_btn = QToolButton()
        poly_btn.setText("⬠")
        poly_btn.setToolTip("Polygon (P)")
        poly_btn.setFixedSize(_BTN_W, _BTN_H)
        poly_btn.setCheckable(True)
        poly_btn.setFont(font)
        self._btn_tool[poly_btn] = "polygon"
        self._btn_group.addButton(poly_btn)
        layout.addWidget(poly_btn)

        # ------------------------------------------------------------------ #
        # Brush thickness popup
        # ------------------------------------------------------------------ #
        btn_thickness = QToolButton()
        btn_thickness.setText("◢")
        btn_thickness.setToolTip("Brush Thickness")
        btn_thickness.setFixedSize(_BTN_W, _BTN_H)
        btn_thickness.setFont(font)
        btn_thickness.setPopupMode(QToolButton.InstantPopup)

        thickness_menu = QMenu(self)
        thickness_action = QWidgetAction(self)
        t_container = QWidget()
        t_h = QHBoxLayout(t_container)
        t_h.setContentsMargins(8, 4, 8, 4)

        self.slider_thickness = QSlider(Qt.Horizontal)
        self.slider_thickness.setRange(1, 40)
        self.slider_thickness.setValue(8)  # 8 × 0.25 = 2.00 px default
        self.slider_thickness.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_thickness.setTickInterval(4)
        self.slider_thickness.setMinimumWidth(150)

        self.lbl_thickness = QLabel("2.00 px")
        self.lbl_thickness.setFixedWidth(55)

        t_h.addWidget(self.slider_thickness)
        t_h.addWidget(self.lbl_thickness)
        thickness_action.setDefaultWidget(t_container)
        thickness_menu.addAction(thickness_action)
        btn_thickness.setMenu(thickness_menu)
        layout.addWidget(btn_thickness)
        self.slider_thickness.valueChanged.connect(self._on_slider_changed)

        # ------------------------------------------------------------------ #
        # Divider
        # ------------------------------------------------------------------ #
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        # ------------------------------------------------------------------ #
        # SAM tool
        # ------------------------------------------------------------------ #
        sam_btn = QToolButton()
        sam_btn.setText("✦")
        sam_btn.setToolTip("SAM Segment (S)")
        sam_btn.setFixedSize(_BTN_W, _BTN_H)
        sam_btn.setCheckable(True)
        sam_btn.setFont(font)
        self._btn_tool[sam_btn] = "sam_bbox"
        self._btn_group.addButton(sam_btn)
        layout.addWidget(sam_btn)

        # ------------------------------------------------------------------ #
        # SAM options popup
        # ------------------------------------------------------------------ #
        btn_sam_opts = QToolButton()
        btn_sam_opts.setText("⚙︎")
        btn_sam_opts.setToolTip("SAM Options")
        btn_sam_opts.setFixedSize(_BTN_W, _BTN_H)
        btn_sam_opts.setFont(font)
        btn_sam_opts.setPopupMode(QToolButton.InstantPopup)

        sam_menu = QMenu(self)
        sam_action = QWidgetAction(self)
        sam_container = QWidget()
        sam_v = QVBoxLayout(sam_container)
        sam_v.setContentsMargins(8, 6, 8, 6)
        sam_v.setSpacing(4)

        variant_row = QHBoxLayout()
        variant_row.addWidget(QLabel("Variant:"))
        self.sam_variant_combo = QComboBox()
        self.sam_variant_combo.addItems(list(_SAM_VARIANT_MAP.keys()))
        self.sam_variant_combo.setToolTip(
            "SAM 2 model size — tiny is fastest, large is most accurate.\n"
            "Weights are downloaded to sam_weights/ on first use."
        )
        self.sam_variant_combo.setMinimumWidth(130)
        variant_row.addWidget(self.sam_variant_combo)
        sam_v.addLayout(variant_row)

        self.sam_status_lbl = QLabel("Model: not loaded")
        self.sam_status_lbl.setStyleSheet("color: grey; font-style: italic;")
        sam_v.addWidget(self.sam_status_lbl)

        sam_action.setDefaultWidget(sam_container)
        sam_menu.addAction(sam_action)
        btn_sam_opts.setMenu(sam_menu)
        layout.addWidget(btn_sam_opts)
        self.sam_variant_combo.currentTextChanged.connect(self._on_sam_combo_changed)

        self._btn_group.buttonClicked.connect(self._on_btn_clicked)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def current_sam_variant(self) -> str:
        """Return the internal variant key for the currently selected SAM model."""
        return _SAM_VARIANT_MAP.get(self.sam_variant_combo.currentText(), "sam2_t.pt")

    def deselect_all(self) -> None:
        """Uncheck all tool buttons (called when the canvas cancels a tool)."""
        self._active_tool = ""
        self._btn_group.setExclusive(False)
        for btn in self._btn_tool:
            btn.setChecked(False)
        self._btn_group.setExclusive(True)

    def toggle_polygon(self) -> None:
        """Toggle the polygon tool on/off."""
        self._toggle_tool("polygon")

    def toggle_sam(self) -> None:
        """Toggle the SAM segment tool on/off."""
        self._toggle_tool("sam_bbox")

    # ------------------------------------------------------------------ #
    # Internal slots
    # ------------------------------------------------------------------ #

    def _on_sam_combo_changed(self, display_name: str) -> None:
        self.sam_variant_changed.emit(_SAM_VARIANT_MAP.get(display_name, "sam2_t.pt"))

    def _on_btn_clicked(self, btn: QToolButton) -> None:
        tool_name = self._btn_tool.get(btn, "")
        if tool_name == self._active_tool:
            self.deselect_all()
            self.tool_selected.emit("")
        else:
            self._active_tool = tool_name
            self.tool_selected.emit(tool_name)

    def _on_slider_changed(self, value: int) -> None:
        thickness = value * 0.25
        self.lbl_thickness.setText(f"{thickness:.2f} px")
        self.thickness_changed.emit(thickness)

    def _toggle_tool(self, tool_name: str) -> None:
        for btn, name in self._btn_tool.items():
            if name == tool_name:
                if self._active_tool == tool_name:
                    self.deselect_all()
                    self.tool_selected.emit("")
                else:
                    self._active_tool = tool_name
                    self._btn_group.setExclusive(False)
                    btn.setChecked(True)
                    self._btn_group.setExclusive(True)
                    self.tool_selected.emit(tool_name)
                return
