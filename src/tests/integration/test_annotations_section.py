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
    """Verify that AnnotationsSection uses a sortable QTableView with correct column configuration.

    Checks that the table view exists, sorting is enabled, the color column has no
    header text, the vertices column is labeled 'Points', area defaults to 'Area (px)',
    and the initial sort indicator is on the CLASS column. Success means all structural
    assertions pass.
    """
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
    """Verify that clicking a sorted row emits annotation_selected with the correct source annotation index.

    Sorts annotations alphabetically (crack, scratch, void) so the visual order changes.
    Clicks the row showing annotation index 2 (void). Success means the annotation_selected
    signal emits the source annotation index 2, not the proxy row number.
    """
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
    """Verify that clicking delete on a sorted row deletes the correct source annotation.

    Sorts alphabetically and clicks delete for the row showing annotation 0 (scratch).
    Success means the scratch annotation is removed from the model while crack and void
    remain, confirming the source index (not proxy row) was used for deletion.
    """
    widget, model = annotations_section
    widget._proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)
    index = _proxy_index_for_annotation(widget, 0, AnnotationColumns.DELETE)

    _click_index(qtbot, widget, index)

    annos = model.get_annotations(0)
    assert len(annos) == 2
    assert [anno["category_name"] for anno in annos] == ["crack", "void"]


def test_visibility_button_after_sort_targets_source_index(annotations_section, qtbot):
    """Verify that clicking the visibility button on a sorted row toggles the correct source annotation.

    Sorts alphabetically and clicks visibility for annotation 0 (scratch). Only the
    scratch annotation should become hidden; crack and void remain visible. Clicking
    again should re-show scratch. Success means only the targeted annotation's
    visibility changes.
    """
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
    """Verify that the annotations table dynamically grows in height to show all rows without a scrollbar.

    Confirms the table has no vertical scrollbar and all rows are visible without
    scrolling. After adding a new annotation, the table should grow taller to
    accommodate the new row. Success means the table height increases and the last
    row remains visible after the addition.
    """
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
    """Verify that editing the class column in the view updates the underlying dataset annotation.

    Uses setData on the table model's class column to change the first annotation's
    category to 'void'. Success means the dataset model reflects 'void' as the new
    category_name for that annotation.
    """
    widget, model = annotations_section
    source_index = widget._table_model.index(0, AnnotationColumns.CLASS)

    assert widget._table_model.setData(source_index, "void", Qt.EditRole)

    assert model.get_annotations(0)[0]["category_name"] == "void"
