import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QMessageBox, QTableView

from core.states.dataset_state import DatasetState
from models.classes_model import CLASS_NAME_ROLE, ClassColumns
from models.dataset_model import DatasetTableModel
from views.annomate.sections.classes import ClassesSection


@pytest.fixture
def classes_section(qtbot):
    model = DatasetTableModel(DatasetState())
    model.add_class("Beta", (20, 20, 20))
    model.add_class("alpha", (10, 10, 10))
    model.add_class("Gamma", (30, 30, 30))
    model.load_folder("/fake", ["one.jpg", "two.jpg"])
    model.add_annotation(0, "Beta", [(0, 0), (1, 0), (1, 1)])
    model.add_annotation(0, "Beta", [(0, 0), (2, 0), (2, 2)])
    model.add_annotation(0, "alpha", [(0, 0), (1, 0), (1, 1)])
    model.add_annotation(1, "Gamma", [(0, 0), (1, 0), (1, 1)])

    widget = ClassesSection(model)
    qtbot.addWidget(widget)
    widget.set_current_row(0)
    widget.show()
    qtbot.wait(50)
    return widget, model


def _proxy_index_for_class(widget, name: str, column: int):
    for row in range(widget._proxy.rowCount()):
        index = widget._proxy.index(row, column)
        if index.data(CLASS_NAME_ROLE) == name:
            return index
    raise AssertionError(f"Class not found in proxy: {name}")


def _click_index(qtbot, widget, proxy_index) -> None:
    rect = widget._table.visualRect(proxy_index)
    qtbot.mouseClick(widget._table.viewport(), Qt.LeftButton, pos=rect.center())


def test_clicking_sorted_row_emits_correct_class(classes_section, qtbot):
    """Verify that clicking a sorted row emits class_selected with the correct class name.

    Sorts alphabetically (alpha, beta, gamma) and clicks the 'gamma' row. Success means
    the class_selected signal emits 'gamma' and _selected_name is set to 'gamma'.
    """
    widget, _model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.AscendingOrder)
    index = _proxy_index_for_class(widget, "gamma", ClassColumns.CLASS)

    with qtbot.waitSignal(widget.class_selected, timeout=1000) as signal:
        _click_index(qtbot, widget, index)

    assert signal.args == ["gamma"]
    assert widget._selected_name == "gamma"


def test_adding_class_selects_new_class_under_active_sort(classes_section, qtbot):
    """Verify that adding a new class selects it and emits class_selected even when the proxy is sorted.

    With a descending sort active, adds a new class 'Delta' via the input field. Success
    means class_selected fires with 'delta', the class appears in the model, and
    _selected_name reflects the new class.
    """
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    widget._name_input.setText("Delta")

    with qtbot.waitSignal(widget.class_selected, timeout=1000) as signal:
        widget._add_class()

    assert signal.args == ["delta"]
    assert "delta" in model.get_class_names()
    assert widget._selected_name == "delta"


def test_deleting_class_after_sort_targets_correct_class(
    classes_section, qtbot, monkeypatch
):
    """Verify that clicking delete on a sorted row deletes the correct source class, not the proxy row class.

    Sorts descending (gamma, beta, alpha) and clicks delete for the 'alpha' row.
    Confirms the deletion by monkeypatching QMessageBox to return Yes. Success means
    'alpha' is removed but 'beta' and 'gamma' remain.
    """
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    index = _proxy_index_for_class(widget, "alpha", ClassColumns.DELETE)
    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )

    _click_index(qtbot, widget, index)

    assert "alpha" not in model.get_class_names()
    assert "beta" in model.get_class_names()
    assert "gamma" in model.get_class_names()


def test_deleting_class_with_annotations_can_be_cancelled(
    classes_section, qtbot, monkeypatch
):
    """Verify that cancelling the delete confirmation keeps the class and its annotations intact.

    'beta' has 2 annotations. Clicking delete and choosing No in the confirmation dialog
    should leave the class and all its annotations unchanged. Success means 'beta' is
    still in the model with its 2 annotations.
    """
    widget, model = classes_section
    index = _proxy_index_for_class(widget, "beta", ClassColumns.DELETE)
    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.No,
    )

    _click_index(qtbot, widget, index)

    assert "beta" in model.get_class_names()
    assert model.get_class_annotation_count("beta") == 2


def test_deleting_class_without_annotations_does_not_prompt(
    classes_section, qtbot, monkeypatch
):
    """Verify that deleting a class with no annotations skips the confirmation dialog.

    Adds an 'Empty' class with no annotations and deletes it. The QMessageBox should
    never be shown for annotation-free classes. Success means no assertion error is
    raised by the monkeypatched dialog and 'empty' is removed from the model.
    """
    widget, model = classes_section
    model.add_class("Empty", (1, 2, 3))
    widget._table_model.refresh_classes()
    index = _proxy_index_for_class(widget, "empty", ClassColumns.DELETE)

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("Delete confirmation should not be shown")

    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        fail_if_prompted,
    )

    _click_index(qtbot, widget, index)

    assert "empty" not in model.get_class_names()


def test_visibility_button_after_sort_targets_correct_class(classes_section, qtbot):
    """Verify that clicking the visibility button on a sorted row toggles only the targeted class.

    Sorts descending and clicks visibility for 'alpha'. Only alpha should become hidden;
    beta and gamma remain visible. Clicking again re-shows alpha. Success means only
    the targeted class visibility changes each time.
    """
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    index = _proxy_index_for_class(widget, "alpha", ClassColumns.VISIBILITY)

    _click_index(qtbot, widget, index)

    assert model.is_class_visible("alpha") is False
    assert model.is_class_visible("beta") is True
    assert model.is_class_visible("gamma") is True

    _click_index(qtbot, widget, index)

    assert model.is_class_visible("alpha") is True


def test_color_column_updates_correct_class_after_sort(
    classes_section, qtbot, monkeypatch
):
    """Verify that clicking the color column on a sorted row opens the color picker and updates the correct class.

    Sorts descending and clicks the color cell for 'beta'. Monkeypatches QColorDialog
    to return a specific color (101, 112, 123). Success means 'beta' receives the new
    color in the dataset model.
    """
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    index = _proxy_index_for_class(widget, "beta", ClassColumns.COLOR)

    monkeypatch.setattr(
        "views.annomate.sections.classes.QColorDialog.getColor",
        lambda *args, **kwargs: QColor(101, 112, 123),
    )

    _click_index(qtbot, widget, index)

    assert model.get_class_color("beta") == (101, 112, 123)


def test_classes_section_uses_table_view(classes_section):
    """Verify that ClassesSection uses a sortable QTableView with correct structural configuration.

    Checks that the internal table is a QTableView with sorting enabled, the color
    column has an empty header, and the initial sort indicator is on the CLASS column.
    Success means all structural assertions pass.
    """
    widget, _model = classes_section

    assert widget.findChild(QTableView) is widget._table
    assert widget._table.isSortingEnabled()
    assert widget._table_model.headerData(ClassColumns.COLOR, Qt.Horizontal) == ""
    assert widget._table.horizontalHeader().sortIndicatorSection() == ClassColumns.CLASS


def test_classes_table_expands_to_show_all_rows(classes_section, qtbot):
    """Verify that the classes table dynamically grows in height to show all rows without a scrollbar.

    Confirms the table has no vertical scrollbar and all rows are visible without
    scrolling. After adding a new class, the table should grow taller to accommodate
    the new row. Success means height increases and the last row remains visible.
    """
    widget, _model = classes_section

    assert widget._table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    last_index = widget._proxy.index(widget._proxy.rowCount() - 1, ClassColumns.CLASS)
    assert (
        widget._table.visualRect(last_index).bottom()
        < widget._table.viewport().height()
    )

    old_height = widget._table.height()
    widget._name_input.setText("Delta")
    widget._add_class()
    qtbot.wait(50)

    last_index = widget._proxy.index(widget._proxy.rowCount() - 1, ClassColumns.CLASS)
    assert widget._table.height() > old_height
    assert (
        widget._table.visualRect(last_index).bottom()
        < widget._table.viewport().height()
    )
