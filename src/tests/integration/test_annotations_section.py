import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from core.states.calibration_state import CalibrationState
from core.states.dataset_state import DatasetState
from models.annotations_model import AnnotationColumns
from models.calibration_model import CalibrationModel
from models.dataset_model import DatasetTableModel
from views.annomate.sections.annotations import _ICON_BTN_W, AnnotationsSection


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


def row_order_annotation_indices(widget):
    """Annotation indices in current on-screen (top-to-bottom) row order."""
    return [
        widget._rows_layout.itemAt(i).widget()._idx
        for i in range(widget._rows_layout.count())
    ]


def test_annotations_section_uses_plain_widget_rows_sorted_by_class(
    annotations_section,
):
    """Verify AnnotationsSection builds one plain-widget row per annotation, sorted by class.

    No QTableView/QAbstractItemView involved -- rows are alphabetically ordered
    by class name (crack, scratch, void) with no user-facing sort control.
    Success means the row order matches alphabetical class order and each row
    exposes the expected sub-widgets.
    """
    widget, _model = annotations_section

    assert len(widget._rows) == 3
    ordered = row_order_annotation_indices(widget)
    classes_in_order = [
        widget._rows[idx]._combo.currentText() for idx in ordered
    ]
    assert classes_in_order == ["crack", "scratch", "void"]


def test_clicking_row_emits_source_annotation_index(annotations_section, qtbot):
    """Verify that clicking a row emits annotation_selected with its source annotation index.

    Annotation index 2 ('void') sorts last alphabetically. Clicking its row
    must emit source annotation index 2, not its on-screen position.
    """
    widget, _model = annotations_section
    row = widget._rows[2]

    with qtbot.waitSignal(widget.annotation_selected, timeout=1000) as signal:
        qtbot.mouseClick(row, Qt.LeftButton)

    assert signal.args == [2]
    assert widget._selected_idx == 2
    assert row.styleSheet() != ""  # selected row gets a highlight style


def test_deleting_annotation_targets_source_index(annotations_section, qtbot):
    """Verify that clicking a row's delete button removes the correct source annotation.

    Annotation 0 ('scratch') sorts in the middle. Clicking its delete button
    must remove scratch specifically, leaving crack and void.
    """
    widget, model = annotations_section
    row = widget._rows[0]

    qtbot.mouseClick(row._delete_btn, Qt.LeftButton)

    annos = model.get_annotations(0)
    assert len(annos) == 2
    assert [anno["category_name"] for anno in annos] == ["crack", "void"]


def test_visibility_button_targets_source_index(annotations_section, qtbot):
    """Verify that clicking a row's eye button toggles the correct source annotation.

    Only annotation 0 ('scratch') should toggle; crack and void stay visible.
    Clicking again restores visibility.
    """
    widget, model = annotations_section
    row = widget._rows[0]

    qtbot.mouseClick(row._eye_btn, Qt.LeftButton)

    assert model.get_annotations(0)[0]["visible"] is False
    assert model.get_annotations(0)[1].get("visible", True) is True
    assert model.get_annotations(0)[2].get("visible", True) is True

    row = widget._rows[0]  # rows were rebuilt after the model reset
    qtbot.mouseClick(row._eye_btn, Qt.LeftButton)

    assert model.get_annotations(0)[0]["visible"] is True


def test_rows_rebuild_when_annotation_added(annotations_section, qtbot):
    """Verify a new annotation gets its own row without disturbing the others."""
    widget, model = annotations_section

    model.add_annotation(0, "crack", [(0, 0), (1, 0), (1, 1)])
    qtbot.wait(50)

    assert len(widget._rows) == 4
    assert 3 in widget._rows


def test_changing_class_via_combo_updates_source_annotation(annotations_section, qtbot):
    """Verify changing a row's class combo box updates the underlying annotation."""
    widget, model = annotations_section
    row = widget._rows[0]  # scratch

    pos = row._combo.findText("void")
    row._combo.setCurrentIndex(pos)
    row.class_changed.emit(0, "void")

    assert model.get_annotations(0)[0]["category_name"] == "void"


def test_header_labels_class_points_area_only(annotations_section):
    """Verify the compact labels Class/Pts/Area and blank action headers.

    The header must be visible once there are annotations, and its text
    should match the model's own header labels (so the "Area" unit stays in
    sync with calibration) rather than being hand-duplicated in the view.
    """
    widget, _model = annotations_section

    assert widget._header_row.isVisible() is True
    header_labels = widget._header_row.findChildren(QLabel)
    header_texts = [lbl.text() for lbl in header_labels]

    assert (
        widget._table_model.headerData(AnnotationColumns.CLASS, Qt.Horizontal)
        in header_texts
    )
    assert widget._vertices_header_lbl.text() == "Pts"
    assert widget._vertices_header_lbl.toolTip() == "Node count"
    assert widget._area_header_lbl.text() == "Area"
    assert widget._area_header_lbl.toolTip() == "Area (px)"
    # Swatch/eye/delete columns stay blank -- three empty-text spacer labels.
    assert header_texts.count("") == 3


def test_header_hidden_when_no_annotations(qtbot):
    """Verify the column header row is hidden (not just the rows) when there's nothing to label."""
    model = DatasetTableModel(DatasetState())
    model.add_class("crack", (255, 0, 0))
    model.load_folder("/fake", ["img.jpg"])

    widget = AnnotationsSection(model)
    qtbot.addWidget(widget)
    widget.set_current_row(0)
    widget.show()
    qtbot.wait(20)

    assert widget._header_row.isVisible() is False


def test_row_widths_match_header_so_combo_boxes_align(annotations_section):
    """Verify every row's Nodes/Area/eye/delete widths match the header's, so columns line up.

    This is what keeps the class combo box the same width on every row --
    once the other cells are pinned to fixed widths, the only stretchy
    element (the combo, stretch=1) fills identical leftover space everywhere,
    regardless of how long that row's class name, node count, or area text is.
    """
    widget, _model = annotations_section

    combo_widths = {row._combo.width() for row in widget._rows.values()}
    assert len(combo_widths) == 1  # every combo box ends up the same width

    for row in widget._rows.values():
        assert row.layout().itemAt(2).widget().width() == widget._vertices_col_w
        assert row.layout().itemAt(3).widget().width() == widget._area_col_w
        assert row._eye_btn.width() == _ICON_BTN_W
        assert row._delete_btn.width() == _ICON_BTN_W


def test_column_widths_derive_from_header_text_not_a_fixed_number(annotations_section):
    """Verify Nodes/Area column widths are computed from their own header label.

    Renaming a header (e.g. "Nodes" -> "N") should shrink its column
    automatically, freeing more space for the class combo box, instead of
    requiring a hand-tuned pixel constant that can clip when edited.
    """
    from views.annomate.sections.annotations import _header_label_width

    widget, _model = annotations_section

    assert widget._vertices_col_w == _header_label_width(
        widget._vertices_header_lbl.text()
    )
    assert widget._area_col_w == _header_label_width(widget._area_header_lbl.text())


def test_numeric_columns_expand_to_fit_largest_displayed_value(
    annotations_section, qtbot
):
    """A large area widens the shared Area header and every Area cell."""
    from views.annomate.sections.annotations import _cell_text_width

    widget, model = annotations_section
    model.add_annotation(
        0, "crack", [(0, 0), (123000, 0), (123000, 1), (0, 1)]
    )
    qtbot.wait(20)

    assert widget._area_col_w >= _cell_text_width("123000")
    for row in widget._rows.values():
        assert row.layout().itemAt(3).widget().width() == widget._area_col_w


def test_area_unit_stays_in_tooltip_after_calibration_change(qtbot):
    """The compact Area header retains unit context when calibration changes."""
    model = DatasetTableModel(DatasetState())
    model.add_class("crack", (255, 0, 0))
    model.load_folder("/fake", ["img.jpg"])
    model.add_annotation(0, "crack", [(0, 0), (1, 0), (1, 1)])
    calibration_model = CalibrationModel(CalibrationState())
    widget = AnnotationsSection(model, calibration_model)
    qtbot.addWidget(widget)
    widget.set_current_row(0)

    calibration_model.set_calib_points((0.0, 0.0), (100.0, 0.0))
    calibration_model.apply_calibration(5.0, "mm")
    qtbot.wait(20)

    assert widget._area_header_lbl.text() == "Area"
    assert widget._area_header_lbl.toolTip() == "Area (mm)"


def test_no_annotations_label_shown_when_empty(qtbot):
    """Verify the empty-state label shows when the current image has no annotations."""
    model = DatasetTableModel(DatasetState())
    model.add_class("crack", (255, 0, 0))
    model.load_folder("/fake", ["img.jpg"])

    widget = AnnotationsSection(model)
    qtbot.addWidget(widget)
    widget.set_current_row(0)
    widget.show()
    qtbot.wait(20)

    assert widget._empty_lbl.isVisible() is True
    assert len(widget._rows) == 0
