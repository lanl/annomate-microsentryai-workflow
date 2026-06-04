import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableView

from core.states.dataset_state import DatasetState
from models.annotations_model import ANNOTATION_INDEX_ROLE, AnnotationColumns
from models.dataset_model import DatasetTableModel
from views.annomate.sections.annotations import AnnotationsSection


@pytest.fixture
def annotations_section(qtbot):
    model = DatasetTableModel(DatasetState())
    model.add_class("crack", (255, 0, 0))
    model.add_class("scratch", (0, 255, 0))
    model.add_class("void", (0, 0, 255))
    model.load_folder("/fake", ["img.jpg"])
    model.add_annotation(0, "scratch", [(0, 0), (1, 0), (1, 1)])
    model.add_annotation(0, "crack", [(0, 0), (2, 0), (2, 2), (0, 2)])
    model.add_annotation(0, "void", [(0, 0), (3, 0), (3, 3)])

    widget = AnnotationsSection(model)
    qtbot.addWidget(widget)
    widget.set_current_row(0)
    widget.show()
    qtbot.wait(50)
    return widget, model


def _proxy_index_for_annotation(widget, annotation_idx: int, column: int):
    for row in range(widget._proxy.rowCount()):
        index = widget._proxy.index(row, column)
        if index.data(ANNOTATION_INDEX_ROLE) == annotation_idx:
            return index
    raise AssertionError(f"Annotation not found in proxy: {annotation_idx}")


def _click_index(qtbot, widget, proxy_index) -> None:
    rect = widget._table.visualRect(proxy_index)
    qtbot.mouseClick(widget._table.viewport(), Qt.LeftButton, pos=rect.center())


def test_annotations_section_uses_sortable_table_view(annotations_section):
    widget, _model = annotations_section

    assert widget.findChild(QTableView) is widget._table
    assert widget._table.isSortingEnabled()
    assert widget._table_model.headerData(AnnotationColumns.COLOR, Qt.Horizontal) == ""
    assert (
        widget._table_model.headerData(AnnotationColumns.VERTICES, Qt.Horizontal)
        == "Points"
    )
    assert (
        widget._table_model.headerData(AnnotationColumns.AREA, Qt.Horizontal)
        == "Area (px)"
    )
    assert (
        widget._table.horizontalHeader().sortIndicatorSection()
        == AnnotationColumns.CLASS
    )


def test_clicking_sorted_annotation_emits_source_index(annotations_section, qtbot):
    widget, _model = annotations_section
    widget._proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)
    index = _proxy_index_for_annotation(widget, 2, AnnotationColumns.VERTICES)

    with qtbot.waitSignal(widget.annotation_selected, timeout=1000) as signal:
        _click_index(qtbot, widget, index)

    assert signal.args == [2]
    assert widget._selected_idx == 2


def test_deleting_annotation_after_sort_targets_source_index(
    annotations_section, qtbot
):
    widget, model = annotations_section
    widget._proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)
    index = _proxy_index_for_annotation(widget, 0, AnnotationColumns.DELETE)

    _click_index(qtbot, widget, index)

    annos = model.get_annotations(0)
    assert len(annos) == 2
    assert [anno["category_name"] for anno in annos] == ["crack", "void"]


def test_visibility_button_after_sort_targets_source_index(annotations_section, qtbot):
    widget, model = annotations_section
    widget._proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)
    index = _proxy_index_for_annotation(widget, 0, AnnotationColumns.VISIBILITY)

    _click_index(qtbot, widget, index)

    assert model.get_annotations(0)[0]["visible"] is False
    assert model.get_annotations(0)[1].get("visible", True) is True
    assert model.get_annotations(0)[2].get("visible", True) is True

    index = _proxy_index_for_annotation(widget, 0, AnnotationColumns.VISIBILITY)
    _click_index(qtbot, widget, index)

    assert model.get_annotations(0)[0]["visible"] is True


def test_annotations_table_expands_to_show_all_rows(annotations_section, qtbot):
    widget, model = annotations_section

    assert widget._table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    last_index = widget._proxy.index(
        widget._proxy.rowCount() - 1, AnnotationColumns.CLASS
    )
    assert (
        widget._table.visualRect(last_index).bottom()
        < widget._table.viewport().height()
    )

    old_height = widget._table.height()
    model.add_annotation(0, "Crack", [(0, 0), (1, 0), (1, 1)])
    qtbot.wait(50)

    last_index = widget._proxy.index(
        widget._proxy.rowCount() - 1, AnnotationColumns.CLASS
    )
    assert widget._table.height() > old_height
    assert (
        widget._table.visualRect(last_index).bottom()
        < widget._table.viewport().height()
    )


def test_editing_class_column_updates_source_annotation(annotations_section):
    widget, model = annotations_section
    source_index = widget._table_model.index(0, AnnotationColumns.CLASS)

    assert widget._table_model.setData(source_index, "void", Qt.EditRole)

    assert model.get_annotations(0)[0]["category_name"] == "void"
