import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt

from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.navigator_model import NavigatorColumns
from views.annomate.sections.navigator import DataNavigatorSection


@pytest.fixture
def navigator(qtbot, tmp_path):
    dataset_model = DatasetTableModel(DatasetState())
    inference_model = InferenceModel(InferenceState())
    dataset_model.load_folder(str(tmp_path), ["b.jpg", "a.jpg", "c.jpg"])
    widget = DataNavigatorSection(dataset_model, inference_model)
    qtbot.addWidget(widget)
    widget.resize(420, 180)
    widget.show()
    qtbot.wait(50)
    return widget, dataset_model, inference_model, tmp_path


def source_rows(widget):
    return [
        widget._proxy.mapToSource(
            widget._proxy.index(row, NavigatorColumns.IMG_ID)
        ).row()
        for row in range(widget._proxy.rowCount())
    ]


def click_header(qtbot, widget, section):
    header = widget._table.horizontalHeader()
    x = header.sectionViewportPosition(section) + header.sectionSize(section) // 2
    y = header.height() // 2
    qtbot.mouseClick(header.viewport(), Qt.LeftButton, pos=QPoint(x, y))
    qtbot.wait(20)


def test_header_click_sorts_and_reverses(navigator, qtbot):
    """Verify that clicking a column header sorts the navigator table and a second click reverses the order.

    Clicks the IMG_ID header twice. The first click should sort by image ID in one
    direction, the second should reverse it. Success means the two orderings together
    equal the expected ascending and descending source row sequences.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator

    click_header(qtbot, widget, NavigatorColumns.IMG_ID)
    first_order = source_rows(widget)
    click_header(qtbot, widget, NavigatorColumns.IMG_ID)
    second_order = source_rows(widget)

    assert {tuple(first_order), tuple(second_order)} == {(1, 0, 2), (2, 0, 1)}


def test_clicking_sorted_row_emits_source_row(navigator, qtbot):
    """Verify that clicking a row in the sorted navigator emits image_selected with the source row index.

    Adds two annotations to source row 2 ('c.jpg'), sorts descending by annotation
    count so 'c.jpg' rises to the top, then clicks the top row. Success means the
    image_selected signal emits the source row index 2, not the proxy row 0.
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.add_annotation(2, "Defect", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(2, "Defect", [(0, 0), (2, 0), (2, 2)])
    widget._table.sortByColumn(NavigatorColumns.ANNOTS, Qt.DescendingOrder)
    qtbot.wait(20)

    index = widget._proxy.index(0, NavigatorColumns.IMG_ID)
    point = widget._table.visualRect(index).center()

    with qtbot.waitSignal(widget.image_selected, timeout=1000) as blocker:
        qtbot.mouseClick(widget._table.viewport(), Qt.LeftButton, pos=point)

    assert blocker.args == [2]


def test_select_row_highlights_source_row_after_sort(navigator, qtbot):
    """Verify that select_row highlights the correct table row after a sort and supports adjacent navigation.

    After ascending sort by image ID (a=0, b=1, c=2), calls select_row with different
    source rows and confirms the final selection highlights source row 0. Also verifies
    adjacent_source_row correctly returns the adjacent source rows in the current sort order.
    Success means current selection is source row 0 with only one selected row, and
    adjacent rows are source rows 1 and 2.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget._table.sortByColumn(NavigatorColumns.IMG_ID, Qt.AscendingOrder)
    qtbot.wait(20)

    widget.select_row(0)
    widget.select_row(2)
    widget.select_row(0)

    selected_proxy_row = widget._table.currentIndex().row()
    selected_source_row = widget._proxy.mapToSource(
        widget._proxy.index(selected_proxy_row, NavigatorColumns.IMG_ID)
    ).row()
    assert selected_source_row == 0
    assert len(widget._table.selectionModel().selectedRows()) == 1
    assert widget.adjacent_source_row(0, -1) == 1
    assert widget.adjacent_source_row(0, 1) == 2


def test_microsentry_columns_and_score_resort(navigator, qtbot):
    """Verify that microsentry mode shows the score/class columns and re-sorts by score after inference.

    Initially SCORE and CLASS columns are hidden. After enabling microsentry mode they
    become visible. After storing inference results and calling set_row_inference, the
    table sorted by score descending should place the highest-scoring row (c.jpg,
    source row 2) at the top. Success means column visibility changes and sort order
    reflects inference scores.
    """
    widget, _dataset_model, inference_model, tmp_path = navigator
    assert widget._table.isColumnHidden(NavigatorColumns.SCORE)
    assert widget._table.isColumnHidden(NavigatorColumns.CLASS)

    widget.set_microsentry_mode(True)
    assert not widget._table.isColumnHidden(NavigatorColumns.SCORE)
    assert not widget._table.isColumnHidden(NavigatorColumns.CLASS)

    widget._table.sortByColumn(NavigatorColumns.SCORE, Qt.DescendingOrder)
    inference_model.set_score_map(
        str(tmp_path / "b.jpg"), 0.25, np.zeros((2, 2), dtype=np.float32)
    )
    widget.set_row_inference(0, 0.25, "NORMAL")
    inference_model.set_score_map(
        str(tmp_path / "c.jpg"), 0.95, np.zeros((2, 2), dtype=np.float32)
    )
    widget.set_row_inference(2, 0.95, "ANOMALY")
    qtbot.wait(20)

    assert source_rows(widget)[0] == 2


def test_column_menu_toggles_optional_columns(navigator, qtbot):
    """Verify that unchecking column menu actions hides the corresponding columns.

    Initially all standard columns (STATUS, IMG_ID, ANNOTS, DECISION) are visible.
    Unchecking ANNOTS and DECISION in the column actions should hide those columns
    while leaving STATUS and IMG_ID visible. Success means the hidden state of each
    column reflects the checked state of its action.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator

    assert not widget._table.isColumnHidden(NavigatorColumns.STATUS)
    assert not widget._table.isColumnHidden(NavigatorColumns.IMG_ID)
    assert not widget._table.isColumnHidden(NavigatorColumns.ANNOTS)
    assert not widget._table.isColumnHidden(NavigatorColumns.DECISION)

    widget._column_actions[NavigatorColumns.ANNOTS].setChecked(False)
    widget._column_actions[NavigatorColumns.DECISION].setChecked(False)
    qtbot.wait(20)

    assert widget._table.isColumnHidden(NavigatorColumns.ANNOTS)
    assert widget._table.isColumnHidden(NavigatorColumns.DECISION)
    assert not widget._table.isColumnHidden(NavigatorColumns.STATUS)
    assert not widget._table.isColumnHidden(NavigatorColumns.IMG_ID)


def test_microsentry_column_menu_state_is_respected(navigator, qtbot):
    """Verify that setting a microsentry column action to unchecked keeps it hidden even after enabling microsentry mode.

    The SCORE column action is checked by default but SCORE is hidden until microsentry
    mode activates. Unchecking SCORE's action before enabling microsentry mode should
    keep SCORE hidden even after microsentry activates. CLASS should still become
    visible. Success means SCORE stays hidden and CLASS becomes visible.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator

    assert widget._column_actions[NavigatorColumns.SCORE].isChecked()
    assert widget._table.isColumnHidden(NavigatorColumns.SCORE)

    widget._column_actions[NavigatorColumns.SCORE].setChecked(False)
    widget.set_microsentry_mode(True)
    qtbot.wait(20)

    assert widget._table.isColumnHidden(NavigatorColumns.SCORE)
    assert not widget._table.isColumnHidden(NavigatorColumns.CLASS)
