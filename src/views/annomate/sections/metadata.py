from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QTextEdit,
)


class MetadataSection(QWidget):
    """Inspector name + image note fields for the currently displayed image.

    Writes to the model on edit (editingFinished / textChanged) and refreshes
    when the current row changes.  Signals are blocked during programmatic
    population to avoid spurious writes.
    """

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._current_row: int = -1
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Inspector"))

        self._inspector_edit = QLineEdit()
        self._inspector_edit.setPlaceholderText("Inspector name…")
        self._inspector_edit.editingFinished.connect(self._store_inspector)
        layout.addWidget(self._inspector_edit)

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
            self._inspector_edit.setText(self.dataset_model.get_inspector(self._current_row))
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

    def _store_inspector(self) -> None:
        if self._current_row < 0:
            return
        self.dataset_model.set_inspector(self._current_row, self._inspector_edit.text().strip())

    def _store_note(self) -> None:
        if self._current_row < 0:
            return
        self.dataset_model.set_note(self._current_row, self._note_edit.toPlainText().strip())
