import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QMessageBox, QTableView

from core.states.dataset_state import DatasetState
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


def row_order_class_names(widget):
    """Class names in current on-screen (top-to-bottom) row order."""
    return [
        widget._rows_layout.itemAt(i).widget()._name
        for i in range(widget._rows_layout.count())
    ]


def test_classes_section_uses_plain_widget_rows_sorted_by_name(classes_section):
    """Verify ClassesSection builds one plain-widget row per class, sorted alphabetically.

    No QTableView/QAbstractItemView involved -- rows are alphabetically ordered
    by class name (alpha, beta, gamma) with no user-facing sort control.
    """
    widget, _model = classes_section

    assert widget.findChild(QTableView) is None
    assert len(widget._rows) == 3
    assert row_order_class_names(widget) == ["alpha", "beta", "gamma"]


def test_clicking_row_emits_class_selected(classes_section, qtbot):
    """Verify that clicking a row (outside its controls) emits class_selected with its name."""
    widget, _model = classes_section
    row = widget._rows["gamma"]

    with qtbot.waitSignal(widget.class_selected, timeout=1000) as signal:
        qtbot.mouseClick(row, Qt.LeftButton)

    assert signal.args == ["gamma"]
    assert widget._selected_name == "gamma"
    assert row.styleSheet() != ""  # selected row gets a highlight style


def test_clicking_swatch_changes_color_without_emitting_class_selected(
    classes_section, qtbot, monkeypatch
):
    """Verify clicking the color swatch opens the color picker without a row-click selection.

    The swatch is a nested clickable widget inside the row -- its own press
    must be consumed there (QToolButton always accepts its press), not
    propagate to the parent row's `activated` (which would emit
    class_selected). _change_color re-selects the class internally
    afterward, but silently (emit=False), so class_selected must not fire.
    """
    widget, model = classes_section
    row = widget._rows["beta"]
    monkeypatch.setattr(
        "views.annomate.sections.classes.QColorDialog.getColor",
        lambda *args, **kwargs: QColor(101, 112, 123),
    )

    received = []
    widget.class_selected.connect(received.append)
    qtbot.mouseClick(row._swatch, Qt.LeftButton)

    assert model.get_class_color("beta") == (101, 112, 123)
    assert received == []
    assert widget._selected_name == "beta"  # re-selected internally, just not emitted


def test_deleting_class_targets_correct_row(classes_section, qtbot, monkeypatch):
    """Verify clicking a row's delete button removes that class specifically."""
    widget, model = classes_section
    row = widget._rows["alpha"]
    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )

    qtbot.mouseClick(row._delete_btn, Qt.LeftButton)

    assert "alpha" not in model.get_class_names()
    assert "beta" in model.get_class_names()
    assert "gamma" in model.get_class_names()


def test_deleting_class_with_annotations_can_be_cancelled(
    classes_section, qtbot, monkeypatch
):
    """Verify cancelling the delete confirmation keeps the class and its annotations intact."""
    widget, model = classes_section
    row = widget._rows["beta"]
    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.No,
    )

    qtbot.mouseClick(row._delete_btn, Qt.LeftButton)

    assert "beta" in model.get_class_names()
    assert model.get_class_annotation_count("beta") == 2


def test_deleting_class_without_annotations_does_not_prompt(
    classes_section, qtbot, monkeypatch
):
    """Verify deleting a class with no annotations skips the confirmation dialog."""
    widget, model = classes_section
    model.add_class("Empty", (1, 2, 3))
    widget._table_model.refresh_classes()
    qtbot.wait(20)
    row = widget._rows["empty"]

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("Delete confirmation should not be shown")

    monkeypatch.setattr(
        "views.annomate.sections.classes.QMessageBox.question",
        fail_if_prompted,
    )

    qtbot.mouseClick(row._delete_btn, Qt.LeftButton)

    assert "empty" not in model.get_class_names()


def test_visibility_button_targets_correct_row(classes_section, qtbot):
    """Verify clicking a row's eye button toggles only that class's visibility."""
    widget, model = classes_section
    row = widget._rows["alpha"]

    qtbot.mouseClick(row._eye_btn, Qt.LeftButton)

    assert model.is_class_visible("alpha") is False
    assert model.is_class_visible("beta") is True
    assert model.is_class_visible("gamma") is True

    row = widget._rows["alpha"]  # rows were rebuilt after the model reset
    qtbot.mouseClick(row._eye_btn, Qt.LeftButton)

    assert model.is_class_visible("alpha") is True


def test_adding_class_selects_new_class(classes_section, qtbot):
    """Verify adding a class via the input field selects it and emits class_selected."""
    widget, model = classes_section
    widget._name_input.setText("Delta")

    with qtbot.waitSignal(widget.class_selected, timeout=1000) as signal:
        widget._add_class()

    assert signal.args == ["delta"]
    assert "delta" in model.get_class_names()
    assert widget._selected_name == "delta"


def test_header_labels_class_tot_only(classes_section):
    """Verify the compact Class/Tot header labels and blank action headers."""
    widget, _model = classes_section

    header_labels = widget._header_row.findChildren(QLabel)
    header_texts = [lbl.text() for lbl in header_labels]

    assert (
        widget._table_model.headerData(1, Qt.Horizontal) in header_texts
        or "Class" in header_texts
    )
    assert widget._total_header_lbl.text() == "Tot"
    assert widget._total_header_lbl.toolTip() == "Class count for the whole dataset"
    # Swatch/eye/delete columns stay blank -- three empty-text spacer labels.
    assert header_texts.count("") == 3


def test_count_column_width_to_widest_displayed_value(classes_section, qtbot):
    """Verify the Tot column shrinks-to-fit its widest value, not a fixed width.

    'beta' has total count 2 (single digit) initially. Adding enough beta
    annotations to reach a 3-digit total should widen both the header and
    every row's Tot column to match, instead of clipping or leaving unused
    fixed-width padding.
    """
    from views.annomate.sections.classes import _cell_text_width

    widget, model = classes_section
    narrow_total_w = widget._total_col_w

    for _ in range(100):
        model.add_annotation(1, "Beta", [(0, 0), (1, 0), (1, 1)])
    qtbot.wait(20)

    assert widget._total_col_w > narrow_total_w
    assert widget._total_col_w >= _cell_text_width("102")
    assert widget._total_header_lbl.width() == widget._total_col_w
    for row in widget._rows.values():
        assert row._total_lbl.width() == widget._total_col_w


def test_count_column_width_derives_from_widest_cell_or_header(classes_section):
    """Verify the Tot column width equals max(header width, widest cell width).

    Guards against reintroducing a hand-tuned fixed pixel constant that
    wastes space around small counts or clips large ones.
    """
    from views.annomate.sections.classes import (
        ClassColumns,
        _cell_text_width,
        _header_label_width,
    )

    widget, _model = classes_section

    widest_cell = max(
        _cell_text_width(
            str(
                widget._table_model.index(r, ClassColumns.TOTAL).data(Qt.DisplayRole)
                or "0"
            )
        )
        for r in range(widget._table_model.rowCount())
    )
    assert widget._total_col_w == max(
        _header_label_width(widget._total_header_lbl.text()), widest_cell
    )


def test_image_level_mode_hides_visibility_column(classes_section, qtbot):
    """Verify switching to Image Level mode hides the eye button.

    It stays visible in Pixel Level mode (the default) and hides once the
    user switches modes, matching the old table's setColumnHidden behavior.
    """
    widget, _model = classes_section
    row = widget._rows["alpha"]
    assert row._eye_btn.isVisible() is True
    assert widget._eye_header_spacer.isVisible() is True

    with qtbot.waitSignal(widget.annotation_mode_changed, timeout=1000) as signal:
        qtbot.mouseClick(widget._image_btn, Qt.LeftButton)

    assert signal.args == ["image_level"]
    assert widget._eye_header_spacer.isVisible() is False
    row = widget._rows["alpha"]  # rows rebuilt after mode-driven data change
    assert row._eye_btn.isVisible() is False
