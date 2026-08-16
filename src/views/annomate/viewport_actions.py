from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QToolButton,
    QWidget,
)

from views.icons import material_icon

_ICON_SIZE = 20


class ViewportActionsBar(QFrame):
    """Floating bottom-center actions for canvas zoom/view controls."""

    _MARGIN = 12
    _BTN_SIZE = 32

    def __init__(
        self,
        canvas,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent or canvas)
        self._canvas = canvas
        self._has_image = False

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)
        self.setObjectName("viewportActionsBar")
        self.setStyleSheet(
            """
            QFrame#viewportActionsBar {
                background: palette(window);
                border: 1px solid palette(mid);
                border-radius: 8px;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        self._btn_zoom_in = self._make_button("zoom_in", "Zoom In")
        self._btn_zoom_in.clicked.connect(canvas.zoom_in)
        layout.addWidget(self._btn_zoom_in)

        self._btn_zoom_out = self._make_button("zoom_out", "Zoom Out")
        self._btn_zoom_out.clicked.connect(canvas.zoom_out)
        layout.addWidget(self._btn_zoom_out)

        self._btn_reset = self._make_button("fit_screen", "Reset View")
        self._btn_reset.clicked.connect(canvas.reset_view)
        layout.addWidget(self._btn_reset)

        self.adjustSize()
        already_loaded = hasattr(canvas, "is_image_loaded") and canvas.is_image_loaded()
        self.set_image_loaded(already_loaded)

    def _make_button(self, icon_name: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(material_icon(icon_name, size=_ICON_SIZE, color="black"))
        btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        btn.setToolTip(tooltip)
        btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def set_image_loaded(self, loaded: bool) -> None:
        self._has_image = loaded
        self._btn_zoom_in.setEnabled(loaded)
        self._btn_zoom_out.setEnabled(loaded)
        self._btn_reset.setEnabled(loaded)

    def reposition(self, canvas_size) -> None:
        self.adjustSize()
        x = (canvas_size.width() - self.width()) // 2
        y = canvas_size.height() - self.height() - self._MARGIN
        self.move(max(0, x), max(0, y))
        self._canvas.set_watermark_bar_y(max(0, y))
