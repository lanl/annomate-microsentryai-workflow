"""
AnnoMateStatusBar — bottom status strip for the AnnoMate main window.

Shows live zoom %, image dimensions, active tool, and a right-aligned
task/ready indicator. No custom colors — system palette only.
"""

from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
)


class AnnoMateStatusBar(QWidget):
    """Fixed-height status bar with zoom, dimensions, tool, and task fields.

    Public methods are the only write path — no direct label access from outside.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(26)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        bar = QWidget()
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 0, 8, 0)
        h.setSpacing(0)

        self._lbl_zoom = QLabel("Zoom: 100%")
        h.addWidget(self._lbl_zoom)
        h.addWidget(self._pipe())

        self._lbl_dims = QLabel("— × — px")
        h.addWidget(self._lbl_dims)
        h.addWidget(self._pipe())

        self._lbl_tool = QLabel("Tool: None")
        h.addWidget(self._lbl_tool)

        self._lbl_tool_hint = QLabel("")
        self._lbl_tool_hint.setStyleSheet("color: grey; font-style: italic;")
        self._lbl_tool_hint.setVisible(False)
        h.addWidget(self._lbl_tool_hint)

        h.addWidget(self._pipe())
        self._lbl_class = QLabel("Class: —")
        h.addWidget(self._lbl_class)

        h.addStretch()

        self._lbl_loading = QLabel("Loading model…")
        self._lbl_loading.setVisible(False)
        h.addWidget(self._lbl_loading)
        h.addSpacing(6)

        self._progress_ai = QProgressBar()
        self._progress_ai.setFixedWidth(160)
        self._progress_ai.setMaximumHeight(16)
        self._progress_ai.setTextVisible(True)
        self._progress_ai.setFormat("Microsentry: %v / %m")
        self._progress_ai.setVisible(False)
        h.addWidget(self._progress_ai)
        h.addSpacing(6)

        root.addWidget(bar)

    @staticmethod
    def _pipe() -> QLabel:
        lbl = QLabel("  |  ")
        lbl.setEnabled(False)
        return lbl

    # ------------------------------------------------------------------ #
    # Public update API
    # ------------------------------------------------------------------ #

    def set_zoom(self, factor: float) -> None:
        self._lbl_zoom.setText(f"Zoom: {factor * 100:.0f}%")

    def set_dimensions(self, w: int, h: int) -> None:
        self._lbl_dims.setText(f"{w} × {h} px")

    _TOOL_HINTS = {
        "polygon": "double-click to close · Backspace to undo point · Esc to cancel",
        "sam_bbox": "draw bbox over object · Enter=accept · Esc=cancel",
    }

    _TOOL_DISPLAY = {
        "polygon": "Polygon",
        "sam_bbox": "SAM BBox",
    }

    def set_tool(self, name: str) -> None:
        display = self._TOOL_DISPLAY.get(name, name.capitalize() if name else "None")
        self._lbl_tool.setText(f"Tool: {display}")
        hint = self._TOOL_HINTS.get(name, "")
        self._lbl_tool_hint.setText(f"  —  {hint}" if hint else "")
        self._lbl_tool_hint.setVisible(bool(hint))

    def set_sam_hint(self, text: str) -> None:
        """Update the tool hint area with a dynamic SAM status message."""
        self._lbl_tool_hint.setText(f"  —  {text}" if text else "")
        self._lbl_tool_hint.setVisible(bool(text))

    def set_class(self, name: str) -> None:
        self._lbl_class.setText(f"Class: {name}" if name else "Class: —")

    def set_model_loading(self, loading: bool) -> None:
        self._lbl_loading.setVisible(loading)

    def set_inference_progress(self, done: int, total: int) -> None:
        self._progress_ai.setMaximum(total)
        self._progress_ai.setValue(done)
        self._progress_ai.setVisible(True)

    def clear_inference_progress(self) -> None:
        pass  # bar stays visible so the user can see all images were processed
