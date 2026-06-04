from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
)


class MetadataSection(QWidget):
    """Inspector name + image note fields for the currently displayed image.

    Writes to the model on edit (editingFinished / textChanged) and refreshes
    when the current row changes.  Signals are blocked during programmatic
    population to avoid spurious writes.

    A session inspector can be set via the "Set Inspector" button. Once set,
    it pre-fills the inspector field for any image that has no saved inspector.
    """

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._current_row: int = -1
        self._session_inspector: str = ""
        self._init_ui()
        dataset_model.modelReset.connect(self._reset_session_inspector)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Inspector"))

        inspector_row = QHBoxLayout()
        inspector_row.setSpacing(4)
        self._inspector_edit = QLineEdit()
        self._inspector_edit.setPlaceholderText("Inspector name…")
        self._inspector_edit.editingFinished.connect(self._store_inspector)
        inspector_row.addWidget(self._inspector_edit)

        self._set_inspector_btn = QPushButton("Set Inspector")
        self._set_inspector_btn.setFixedWidth(95)
        self._set_inspector_btn.clicked.connect(self._on_set_inspector)
        inspector_row.addWidget(self._set_inspector_btn)
        layout.addLayout(inspector_row)

        self._session_lbl = QLabel("Session Inspector: —")
        self._session_lbl.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(self._session_lbl)

        layout.addWidget(QLabel("Image note"))

        self._note_edit = QTextEdit()
        self._note_edit.setPlaceholderText("Add a note…")
        self._note_edit.setMaximumHeight(80)
        self._note_edit.textChanged.connect(self._store_note)
        layout.addWidget(self._note_edit)

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._load_fields()

    def _load_fields(self) -> None:
        self._inspector_edit.blockSignals(True)
        self._note_edit.blockSignals(True)

        if self._current_row >= 0:
            saved = self.dataset_model.get_inspector(self._current_row)
            if not saved and self._session_inspector:
                self._inspector_edit.setText(self._session_inspector)
                self.dataset_model.set_inspector(
                    self._current_row, self._session_inspector
                )
            else:
                self._inspector_edit.setText(saved)
            self._note_edit.setPlainText(self.dataset_model.get_note(self._current_row))
            self._inspector_edit.setEnabled(True)
            self._note_edit.setEnabled(True)
        else:
            self._inspector_edit.clear()
            self._note_edit.clear()
            self._inspector_edit.setEnabled(False)
            self._note_edit.setEnabled(False)

        self._inspector_edit.blockSignals(False)
        self._note_edit.blockSignals(False)

    def _reset_session_inspector(self) -> None:
        self._session_inspector = ""
        self._session_lbl.setText("Session Inspector: —")

    def _on_set_inspector(self) -> None:
        name = self._inspector_edit.text().strip()
        self._session_inspector = name
        self._session_lbl.setText(
            f"Session Inspector: {name}" if name else "Session Inspector: —"
        )
        if self._current_row >= 0:
            self.dataset_model.set_inspector(self._current_row, name)

    def _store_inspector(self) -> None:
        if self._current_row < 0:
            return
        self.dataset_model.set_inspector(
            self._current_row, self._inspector_edit.text().strip()
        )

    def _store_note(self) -> None:
        if self._current_row < 0:
            return
        self.dataset_model.set_note(
            self._current_row, self._note_edit.toPlainText().strip()
        )
