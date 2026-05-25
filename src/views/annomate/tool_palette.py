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
)
from PySide6.QtGui import QFont

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

        self._btn_group.buttonClicked.connect(self._on_btn_clicked)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

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

    def _on_btn_clicked(self, btn: QToolButton) -> None:
        tool_name = self._btn_tool.get(btn, "")
        if tool_name == self._active_tool:
            self.deselect_all()
            self.tool_selected.emit("")
        else:
            self._active_tool = tool_name
            self.tool_selected.emit(tool_name)

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
