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
    widget, _model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.AscendingOrder)
    index = _proxy_index_for_class(widget, "Gamma", ClassColumns.CLASS)

    with qtbot.waitSignal(widget.class_selected, timeout=1000) as signal:
        _click_index(qtbot, widget, index)

    assert signal.args == ["Gamma"]
    assert widget._selected_name == "Gamma"


def test_adding_class_selects_new_class_under_active_sort(classes_section, qtbot):
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    widget._name_input.setText("Delta")

    with qtbot.waitSignal(widget.class_selected, timeout=1000) as signal:
        widget._add_class()

    assert signal.args == ["Delta"]
    assert "Delta" in model.get_class_names()
    assert widget._selected_name == "Delta"


def test_deleting_class_after_sort_targets_correct_class(
    classes_section, qtbot, monkeypatch
):
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    index = _proxy_index_for_class(widget, "alpha", ClassColumns.DELETE)
    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )

    _click_index(qtbot, widget, index)

    assert "alpha" not in model.get_class_names()
    assert "Beta" in model.get_class_names()
    assert "Gamma" in model.get_class_names()


def test_deleting_class_with_annotations_can_be_cancelled(
    classes_section, qtbot, monkeypatch
):
    widget, model = classes_section
    index = _proxy_index_for_class(widget, "Beta", ClassColumns.DELETE)
    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.No,
    )

    _click_index(qtbot, widget, index)

    assert "Beta" in model.get_class_names()
    assert model.get_class_annotation_count("Beta") == 2


def test_deleting_class_without_annotations_does_not_prompt(
    classes_section, qtbot, monkeypatch
):
    widget, model = classes_section
    model.add_class("Empty", (1, 2, 3))
    widget._table_model.refresh_classes()
    index = _proxy_index_for_class(widget, "Empty", ClassColumns.DELETE)

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("Delete confirmation should not be shown")

    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        fail_if_prompted,
    )

    _click_index(qtbot, widget, index)

    assert "Empty" not in model.get_class_names()


def test_visibility_button_after_sort_targets_correct_class(classes_section, qtbot):
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    index = _proxy_index_for_class(widget, "alpha", ClassColumns.VISIBILITY)

    _click_index(qtbot, widget, index)

    assert model.is_class_visible("alpha") is False
    assert model.is_class_visible("Beta") is True
    assert model.is_class_visible("Gamma") is True

    _click_index(qtbot, widget, index)

    assert model.is_class_visible("alpha") is True


def test_color_column_updates_correct_class_after_sort(
    classes_section, qtbot, monkeypatch
):
    widget, model = classes_section
    widget._proxy.sort(ClassColumns.CLASS, Qt.DescendingOrder)
    index = _proxy_index_for_class(widget, "Beta", ClassColumns.COLOR)

    monkeypatch.setattr(
        "views.annomate.sections.classes.QColorDialog.getColor",
        lambda *args, **kwargs: QColor(101, 112, 123),
    )

    _click_index(qtbot, widget, index)

    assert model.get_class_color("Beta") == (101, 112, 123)


def test_classes_section_uses_table_view(classes_section):
    widget, _model = classes_section

    assert widget.findChild(QTableView) is widget._table
    assert widget._table.isSortingEnabled()
    assert widget._table_model.headerData(ClassColumns.COLOR, Qt.Horizontal) == ""
    assert widget._table.horizontalHeader().sortIndicatorSection() == ClassColumns.CLASS


def test_classes_table_expands_to_show_all_rows(classes_section, qtbot):
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
