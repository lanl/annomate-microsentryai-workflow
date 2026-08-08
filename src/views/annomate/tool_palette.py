"""
ToolPalette — left tool column for the AnnoMate main window.

Layout (top to bottom):
  hexagon        Polygon tool
  ── divider ──
  auto_fix_high  SAM Segment tool

Tool-specific settings (brush thickness, SAM options, ...) live in the
"Active Tool" tab of the right activity bar, not here -- this column is
tool selection only.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QToolButton,
    QButtonGroup,
    QSizePolicy,
)

from views.icons import material_icon

_BTN_W = 44
_BTN_H = 40
_ICON_SIZE = 22


class ToolPalette(QFrame):
    """Single-column tool panel.

    Signals:
        tool_selected (str): Emitted with the tool name when a button is pressed.
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
        self._drawing_btns: list = []  # buttons disabled in image-level mode

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignTop)

        # ------------------------------------------------------------------ #
        # Polygon tool
        # ------------------------------------------------------------------ #
        poly_btn = QToolButton()
        poly_btn.setIcon(material_icon("hexagon", size=_ICON_SIZE, color="black"))
        poly_btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        poly_btn.setToolTip("Polygon (P)")
        poly_btn.setFixedSize(_BTN_W, _BTN_H)
        poly_btn.setCheckable(True)
        self._btn_tool[poly_btn] = "polygon"
        self._btn_group.addButton(poly_btn)
        layout.addWidget(poly_btn)
        self._drawing_btns.append(poly_btn)

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
        sam_btn.setIcon(material_icon("auto_fix_high", size=_ICON_SIZE, color="black"))
        sam_btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        sam_btn.setToolTip("SAM Segment (S)")
        sam_btn.setFixedSize(_BTN_W, _BTN_H)
        sam_btn.setCheckable(True)
        self._btn_tool[sam_btn] = "sam_bbox"
        self._btn_group.addButton(sam_btn)
        layout.addWidget(sam_btn)
        self._drawing_btns.append(sam_btn)

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

    def set_drawing_enabled(self, enabled: bool) -> None:
        """Enable or disable polygon/SAM drawing tools.

        Called when switching between pixel-level and image-level annotation modes.
        When disabled, deselects any active drawing tool so the canvas goes idle.
        """
        for btn in self._drawing_btns:
            btn.setEnabled(enabled)
        if not enabled:
            self.deselect_all()
            self.tool_selected.emit("")

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
