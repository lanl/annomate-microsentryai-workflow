from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
)


class _SetAllInspectorDialog(QDialog):
    """Dialog for bulk-assigning an inspector name across a filtered set of images."""

    _FILTER_ALL = "All Images"
    _FILTER_REVIEWED = "Reviewed"
    _FILTER_IN_REVIEW = "In Review"

    def __init__(self, dataset_model, prefill_name: str, parent=None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.setWindowTitle("Set All Inspectors")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setMinimumHeight(320)
        self._build_ui(prefill_name)
        self._refresh_list()
        self._update_ok_state()

    def _build_ui(self, prefill_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(QLabel("Inspector name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setText(prefill_name)
        self._name_edit.setPlaceholderText("Input Inspector Name")
        self._name_edit.textChanged.connect(self._update_ok_state)
        name_row.addWidget(self._name_edit, stretch=1)
        layout.addLayout(name_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        filter_row.addWidget(QLabel("Apply to:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(
            [self._FILTER_ALL, self._FILTER_REVIEWED, self._FILTER_IN_REVIEW]
        )
        self._filter_combo.currentIndexChanged.connect(self._refresh_list)
        filter_row.addWidget(self._filter_combo, stretch=1)
        layout.addLayout(filter_row)

        self._summary_lbl = QLabel("0 image(s) will be updated")
        layout.addWidget(self._summary_lbl)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Image", "Current Inspector"])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setEditTriggers(QTreeWidget.NoEditTriggers)
        self._tree.setSortingEnabled(False)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        self._tree.setColumnWidth(1, 140)
        layout.addWidget(self._tree, stretch=1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._ok_btn = self._buttons.button(QDialogButtonBox.Ok)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _get_filtered_rows(self) -> list[int]:
        model = self.dataset_model
        total = model.rowCount()
        sel = self._filter_combo.currentText()
        if sel == self._FILTER_REVIEWED:
            return [r for r in range(total) if model.is_reviewed(r)]
        if sel == self._FILTER_IN_REVIEW:
            return [r for r in range(total) if model.get_review_decision(r) is None]
        return list(range(total))

    def _refresh_list(self) -> None:
        self._tree.clear()
        rows = self._get_filtered_rows()
        for row in rows:
            stem = self.dataset_model.data(self.dataset_model.index(row, 0))
            inspector = self.dataset_model.get_inspector(row) or "—"
            self._tree.addTopLevelItem(QTreeWidgetItem([stem, inspector]))
        self._summary_lbl.setText(f"{len(rows)} image(s) will be updated")
        self._update_ok_state()

    def _update_ok_state(self) -> None:
        name_ok = bool(self._name_edit.text().strip())
        rows_ok = self._tree.topLevelItemCount() > 0
        self._ok_btn.setEnabled(name_ok and rows_ok)

    def get_result(self) -> tuple[str, list[int]]:
        """Return (inspector_name, filtered_row_indices) after dialog is accepted."""
        return self._name_edit.text().strip(), self._get_filtered_rows()


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
        self._set_inspector_btn.setToolTip(
            "Set as session inspector - auto-fills new images as you navigate"
        )
        self._set_inspector_btn.clicked.connect(self._on_set_inspector)
        inspector_row.addWidget(self._set_inspector_btn)

        self._set_all_btn = QPushButton("Set All")
        self._set_all_btn.setFixedWidth(60)
        self._set_all_btn.setToolTip(
            "Bulk-assign an inspector name to a filtered set of images"
        )
        self._set_all_btn.clicked.connect(self._on_set_all_inspectors)
        inspector_row.addWidget(self._set_all_btn)
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

    def _on_set_all_inspectors(self) -> None:
        prefill = self._inspector_edit.text().strip()
        dialog = _SetAllInspectorDialog(
            self.dataset_model, prefill_name=prefill, parent=self
        )
        if dialog.exec() != QDialog.Accepted:
            return
        name, rows = dialog.get_result()
        for row in rows:
            self.dataset_model.set_inspector(row, name)
        self._load_fields()

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
