"""ImportPublicDatasetDialog — imports one category of a public AD benchmark.

Reads from core.public_datasets directly (pure, read-only scanning until
Import is pressed) — no controller layer, since nothing here mutates
application state. The caller applies the result via
ProjectController.new_project_from_import() after this dialog accepts.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from core.public_datasets.build import build_dataset_from_category
from core.public_datasets.registry import PUBLIC_DATASET_ADAPTERS


class ImportPublicDatasetDialog(QDialog):
    """Dialog for importing one category of a public AD benchmark as a fresh dataset.

    Each import covers exactly one category — categories are disjoint image
    pools with their own class taxonomy and don't combine into one project.
    On accept, call get_result() to retrieve the DatasetState-ready dict
    built by build_dataset_from_category().
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Public Dataset")
        self.setModal(True)
        self.setMinimumWidth(480)

        self._result = None

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._format_combo = QComboBox()
        self._format_combo.addItems(sorted(PUBLIC_DATASET_ADAPTERS.keys()))
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        form.addRow("Format:", self._format_combo)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_root)
        path_row.addWidget(self._path_edit)
        path_row.addWidget(browse_btn)
        form.addRow("Dataset root:", path_row)

        self._category_combo = QComboBox()
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        form.addRow("Category:", self._category_combo)

        layout.addLayout(form)

        self._stats_view = QPlainTextEdit()
        self._stats_view.setReadOnly(True)
        self._stats_view.setFixedHeight(140)
        self._stats_view.setPlaceholderText(
            "Choose a dataset root and category to see statistics."
        )
        layout.addWidget(self._stats_view)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._import_btn = self._buttons.addButton(
            "Import", QDialogButtonBox.AcceptRole
        )
        self._import_btn.setEnabled(False)
        self._buttons.accepted.connect(self._on_import)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------ #
    # Result
    # ------------------------------------------------------------------ #

    def get_result(self) -> dict:
        """Return the built dataset dict. Only valid after Accepted."""
        return self._result

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _current_adapter(self):
        return PUBLIC_DATASET_ADAPTERS[self._format_combo.currentText()]

    def _reset_below_format(self) -> None:
        self._category_combo.clear()
        self._stats_view.clear()
        self._import_btn.setEnabled(False)

    def _on_format_changed(self, _text: str) -> None:
        self._reset_below_format()
        if self._path_edit.text():
            self._rescan_categories(self._path_edit.text())

    def _browse_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose Dataset Root", os.getcwd()
        )
        if not directory:
            return
        self._path_edit.setText(directory)
        self._rescan_categories(directory)

    def _rescan_categories(self, directory: str) -> None:
        self._reset_below_format()
        try:
            categories = self._current_adapter().detect_categories(directory)
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Public Dataset", f"Could not scan folder:\n{exc}"
            )
            return
        if not categories:
            QMessageBox.warning(
                self,
                "Import Public Dataset",
                f"No {self._current_adapter().format_name} categories found under:\n"
                f"{directory}",
            )
            return
        self._category_combo.addItems(categories)

    def _on_category_changed(self, category: str) -> None:
        self._stats_view.clear()
        self._import_btn.setEnabled(False)
        if not category or not self._path_edit.text():
            return
        try:
            stats = self._current_adapter().scan_category(
                self._path_edit.text(), category
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Public Dataset", f"Could not scan category:\n{exc}"
            )
            return
        self._stats_view.setPlainText(self._format_stats(stats))
        self._import_btn.setEnabled(True)

    def _format_stats(self, stats) -> str:
        lines = [
            f"Category: {stats.category_name}",
            f"Total images: {stats.total_images}",
            f"Normal (accept): {stats.normal_count}",
        ]
        if stats.defect_counts:
            lines.append("Defect classes:")
            for name, count in sorted(stats.defect_counts.items()):
                lines.append(f"  {name}: {count}")
        if stats.unlabeled_count:
            lines.append(
                f"Unlabeled (no ground truth available): {stats.unlabeled_count}"
            )
        return "\n".join(lines)

    def _on_import(self) -> None:
        root = self._path_edit.text()
        category = self._category_combo.currentText()
        if not root or not category:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._result = build_dataset_from_category(
                self._current_adapter(), root, category
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Public Dataset", f"Import failed:\n{exc}"
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.accept()
