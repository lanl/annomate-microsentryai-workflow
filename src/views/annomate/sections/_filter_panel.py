from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from models.navigator_model import (
    DECISION_FILTER_OPTIONS,
    STATUS_FILTER_OPTIONS,
    NavigatorColumns,
)

_SORT_FIELDS = (
    (NavigatorColumns.IMG_ID, "Filename"),
    (NavigatorColumns.ANNOTS, "Annotations"),
    (NavigatorColumns.DECISION, "Decision"),
    (NavigatorColumns.SCORE, "Score"),
)


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; color: black;")
    return lbl


class _FilterPanel(QWidget):
    """Decision/Status filter checkboxes plus Sort-by options, hosted in a QMenu.

    Meant to be wrapped in a QWidgetAction (matching the checkbox-panel-in-a-
    menu pattern already used by ViewportActionsBar's settings/anomaly/crop
    menus in views/annomate/viewport_actions.py) so the menu stays open while
    checkboxes/radios are toggled.

    set_decision_filter/set_status_filter/set_sort_state are the programmatic
    sync entry points -- guarded by _syncing so pushing external state here
    doesn't re-emit the toggle signals this widget itself drives.

    Signals:
        decision_toggled (str, bool): "accept"/"reject", new checked state.
        status_toggled (str, bool): a STATUS_FILTER_OPTIONS key, new checked state.
        sort_field_clicked (int): a NavigatorColumns value.
        clear_filters_clicked (): "Clear filters" was clicked.
    """

    decision_toggled = Signal(str, bool)
    status_toggled = Signal(str, bool)
    sort_field_clicked = Signal(int)
    clear_filters_clicked = Signal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._decision_checks: dict[str, QCheckBox] = {}
        self._status_checks: dict[str, QCheckBox] = {}
        self._sort_radios: dict[int, QRadioButton] = {}
        self._sort_labels: dict[int, str] = {}

        self.setMinimumWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        layout.addWidget(_section_header("Decision"))
        for key, label in DECISION_FILTER_OPTIONS:
            chk = QCheckBox(label)
            chk.toggled.connect(
                lambda checked, k=key: self._on_decision_toggled(k, checked)
            )
            layout.addWidget(chk)
            self._decision_checks[key] = chk

        layout.addWidget(_divider())

        layout.addWidget(_section_header("Status"))
        for key, label in STATUS_FILTER_OPTIONS:
            chk = QCheckBox(label)
            chk.toggled.connect(
                lambda checked, k=key: self._on_status_toggled(k, checked)
            )
            layout.addWidget(chk)
            self._status_checks[key] = chk
        self._status_checks["conflicting"].setToolTip(
            "Accepted images that still have annotations -- a subset of Incomplete"
        )

        layout.addWidget(_divider())

        layout.addWidget(_section_header("Sort by"))
        sort_group = QButtonGroup(self)
        for column, label in _SORT_FIELDS:
            radio = QRadioButton(label)
            radio.setStyleSheet("color: black;")
            sort_group.addButton(radio)
            # clicked (not toggled) fires even when this field is already the
            # active one, which is what re-clicking needs to reverse sort order.
            radio.clicked.connect(
                lambda checked=False, col=column: self.sort_field_clicked.emit(col)
            )
            layout.addWidget(radio)
            self._sort_radios[column] = radio
            self._sort_labels[column] = label

        layout.addWidget(_divider())

        self._btn_clear = QPushButton("Clear filters")
        self._btn_clear.clicked.connect(self.clear_filters_clicked)
        layout.addWidget(self._btn_clear)

        self.set_sort_state(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

    def _on_decision_toggled(self, key: str, checked: bool) -> None:
        if self._syncing:
            return
        self.decision_toggled.emit(key, checked)

    def _on_status_toggled(self, key: str, checked: bool) -> None:
        if self._syncing:
            return
        self.status_toggled.emit(key, checked)

    def set_decision_filter(self, active: frozenset) -> None:
        self._syncing = True
        for key, chk in self._decision_checks.items():
            chk.setChecked(key in active)
        self._syncing = False

    def set_status_filter(self, active: frozenset) -> None:
        self._syncing = True
        for key, chk in self._status_checks.items():
            chk.setChecked(key in active)
        self._syncing = False

    def set_sort_state(self, column: int, order: Qt.SortOrder) -> None:
        arrow = "↑" if order == Qt.AscendingOrder else "↓"
        self._syncing = True
        for col, radio in self._sort_radios.items():
            label = self._sort_labels[col]
            if col == column:
                radio.setChecked(True)
                radio.setText(f"{label} {arrow}")
            else:
                radio.setText(label)
        self._syncing = False
