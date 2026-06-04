from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
)


class _CollapsibleSection(QWidget):
    """Bold toggle-header + separator + collapsible body.

    Args:
        expandable: When True, the section uses an Expanding vertical size
            policy so it can grow inside a QSplitter. Default is Maximum
            (shrinks to content height).

    Signals:
        toggled (bool): Emitted after the body visibility changes.
            True = expanded, False = collapsed.
    """

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        parent: QWidget = None,
        expandable: bool = False,
        expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        v_policy = QSizePolicy.Expanding if expandable else QSizePolicy.Maximum
        self.setSizePolicy(QSizePolicy.Expanding, v_policy)
        self._title = title
        self._expanded = expanded

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        arrow = "▾" if expanded else "▸"
        self._toggle_btn = QPushButton(f"{arrow}  {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(expanded)
        self._toggle_btn.setStyleSheet(
            "text-align: left; font-weight: bold; padding: 5px 10px;"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        root.addWidget(self._toggle_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        self._body = QWidget()
        if expandable:
            self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 6, 8, 8)
        self._body_layout.setSpacing(4)
        self._body.setVisible(expanded)
        root.addWidget(self._body, stretch=1 if expandable else 0)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = checked
        self._body.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self._toggle_btn.setText(f"{arrow}  {self._title}")
        self.toggled.emit(checked)
