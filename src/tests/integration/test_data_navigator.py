import numpy as np
import pytest
from PySide6.QtCore import Qt

from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.navigator_model import NavigatorColumns
from views.annomate.left_panel import LeftPanel
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


def first_card(widget):
    return widget._cards_layout.itemAt(0).widget()


def test_sorting_reorders_cards_and_reverses(navigator):
    """Verify that sorting the proxy model by image ID reorders the cards, and reversing flips the order.

    Success means ascending and descending sorts by IMG_ID together produce
    the two expected source-row orderings.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator

    widget._proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)
    first_order = source_rows(widget)
    widget._proxy.sort(NavigatorColumns.IMG_ID, Qt.DescendingOrder)
    second_order = source_rows(widget)

    assert {tuple(first_order), tuple(second_order)} == {(1, 0, 2), (2, 0, 1)}


def test_clicking_top_card_emits_source_row(navigator, qtbot):
    """Verify that clicking the visually-first card emits image_selected with its source row.

    Adds two annotations to source row 2 ('c.jpg'), sorts descending by
    annotation count so 'c.jpg' rises to the top, then clicks that card's
    header. Success means the image_selected signal emits source row 2.
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.add_annotation(2, "Defect", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(2, "Defect", [(0, 0), (2, 0), (2, 2)])
    widget._proxy.sort(NavigatorColumns.ANNOTS, Qt.DescendingOrder)
    qtbot.wait(20)

    card = first_card(widget)
    assert card.source_row() == 2

    with qtbot.waitSignal(widget.image_selected, timeout=1000) as blocker:
        qtbot.mouseClick(card._header, Qt.LeftButton)

    assert blocker.args == [2]


def test_select_row_expands_only_that_card_after_sort(navigator):
    """Verify that select_row expands the correct card after a sort and supports adjacent navigation.

    After ascending sort by image ID (a=0, b=1, c=2), calls select_row with
    different source rows and confirms only the final selection is expanded.
    Also verifies adjacent_source_row returns the adjacent source rows in the
    current sort order.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget._proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

    widget.select_row(0)
    widget.select_row(2)
    widget.select_row(0)

    expanded = [row for row, card in widget._cards.items() if card.is_expanded()]
    assert expanded == [0]
    assert widget.adjacent_source_row(0, -1) == 1
    assert widget.adjacent_source_row(0, 1) == 2


def test_microsentry_mode_shows_score_and_score_resorts(navigator, qtbot):
    """Verify that microsentry mode reveals the score label and re-sorts by score after inference.

    Initially the score label is hidden on every card. After enabling
    microsentry mode it becomes visible. After storing inference results and
    calling set_row_inference, sorting by score descending should place the
    highest-scoring row (c.jpg, source row 2) at the top.
    """
    widget, _dataset_model, inference_model, tmp_path = navigator
    assert not widget._cards[0]._score_lbl.isVisible()

    widget.set_microsentry_mode(True)
    assert widget._cards[0]._score_lbl.isVisible()

    inference_model.set_score_map(
        str(tmp_path / "b.jpg"), 0.25, np.zeros((2, 2), dtype=np.float32)
    )
    widget.set_row_inference(0, 0.25)
    inference_model.set_score_map(
        str(tmp_path / "c.jpg"), 0.95, np.zeros((2, 2), dtype=np.float32)
    )
    widget.set_row_inference(2, 0.95)
    widget._proxy.sort(NavigatorColumns.SCORE, Qt.DescendingOrder)
    qtbot.wait(20)

    assert source_rows(widget)[0] == 2


def test_clicking_chip_filters_and_second_click_returns_to_all(navigator, qtbot):
    """Verify clicking a status chip filters the list, and clicking it again clears the filter.

    Row 0 is accepted (reviewed), row 1 is rejected with no work
    (incomplete), row 2 is untouched (undecided). Clicking the "reviewed"
    chip should narrow the list to row 0; clicking it again should restore
    all three rows.
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(0, "accept")
    dataset_model.set_review_decision(1, "reject")
    qtbot.wait(20)

    widget._on_chip_clicked("reviewed")
    assert source_rows(widget) == [0]
    assert widget._filter_mode == "reviewed"

    widget._on_chip_clicked("reviewed")
    assert set(source_rows(widget)) == {0, 1, 2}
    assert widget._filter_mode == "all"


def test_overflow_menu_accept_filter(navigator, qtbot):
    """Verify the overflow menu's "Accept only" action filters to accepted rows."""
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(1, "accept")
    qtbot.wait(20)

    accept_action = next(
        a for a in widget._filter_group.actions() if a.data() == "accept"
    )
    accept_action.trigger()

    assert source_rows(widget) == [1]
    assert widget._filter_mode == "accept"


def test_sort_menu_same_field_reverses_different_field_resets_ascending(navigator):
    """Verify choosing the same sort field twice reverses order, a new field resets to ascending.

    The widget defaults to sorting by IMG_ID ascending, so picking IMG_ID
    from the sort menu once reverses it to descending; picking it again
    reverses back to ascending. Picking a different field resets to
    ascending on that field.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    assert widget._sort_column == NavigatorColumns.IMG_ID
    assert widget._sort_order == Qt.AscendingOrder
    ascending = source_rows(widget)

    widget._on_sort_field_chosen(NavigatorColumns.IMG_ID)
    descending = source_rows(widget)
    assert widget._sort_order == Qt.DescendingOrder
    assert "↓" in widget._btn_sort.text()
    assert ascending != descending

    widget._on_sort_field_chosen(NavigatorColumns.IMG_ID)
    assert widget._sort_order == Qt.AscendingOrder
    assert "↑" in widget._btn_sort.text()
    assert source_rows(widget) == ascending

    widget._on_sort_field_chosen(NavigatorColumns.ANNOTS)
    assert widget._sort_column == NavigatorColumns.ANNOTS
    assert widget._sort_order == Qt.AscendingOrder
    assert "↑" in widget._btn_sort.text()


def test_clicking_a_second_card_collapses_the_first_accordion_style(navigator, qtbot):
    """Verify only one card is ever expanded at a time.

    Clicking card B while card A is expanded must collapse A and expand B,
    and the shared Annotations/Metadata sections move along with the
    expansion into B's body.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget._proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

    widget.select_row(0)
    assert widget._cards[0].is_expanded() is True

    widget.select_row(1)
    assert widget._cards[0].is_expanded() is False
    assert widget._cards[1].is_expanded() is True
    assert widget.annotations.parent() is widget._cards[1].body_container()
    assert widget.metadata.parent() is widget._cards[1].body_container()


def test_clicking_expanded_card_collapses_then_navigation_reexpands(
    navigator, qtbot
):
    """A repeated card click collapses it; programmatic image navigation expands it."""
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget.select_row(0)
    card = widget._cards[0]
    assert card.is_expanded() is True

    qtbot.mouseClick(card._header, Qt.LeftButton)

    assert card.is_expanded() is False
    assert widget.annotations.parent() is widget._shared_slot
    assert widget.metadata.parent() is widget._shared_slot

    # Window navigation (including A/D) calls select_row for the destination.
    widget.select_row(0)

    assert card.is_expanded() is True
    assert widget.annotations.parent() is card.body_container()
    assert widget.metadata.parent() is card.body_container()


def test_typed_inspector_edit_persists_when_switching_cards_without_blur(
    navigator, qtbot
):
    """Verify an in-progress inspector edit is saved before the shared widget is reparented.

    The inspector field only commits on editingFinished (Enter/focus-loss).
    Typing into it and immediately switching to a different card -- without
    tabbing or clicking away first -- must still persist the typed value,
    validating the commit_pending_edits() call in _release_shared_sections.
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    widget.select_row(0)
    inspector_edit = widget.metadata._inspector_edit
    inspector_edit.setFocus()
    qtbot.keyClicks(inspector_edit, "mike")
    assert dataset_model.get_inspector(0) == ""  # not committed yet

    widget.select_row(1)

    assert dataset_model.get_inspector(0) == "mike"


def test_inspector_controls_use_compact_set_button_and_expandable_notes(navigator):
    """Inspector controls start compact and allow notes to expand on demand."""
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    metadata = widget.metadata

    assert metadata._set_inspector_btn.text() == "Set"
    assert metadata._set_inspector_btn.width() == 58
    compact_height = metadata._note_edit.height()
    assert compact_height == metadata._note_height_for_lines(2)

    metadata._expand_note_btn.setChecked(True)

    assert metadata._note_edit.height() == metadata._note_height_for_lines(7)
    assert metadata._note_edit.height() > compact_height
    assert metadata._expand_note_btn.toolTip() == "Collapse inspector notes"


def test_annotations_and_metadata_are_real_descendants_with_zero_images(qtbot):
    """Verify the shared Annotations/Metadata sections exist even with no images loaded.

    These widgets are constructed once up front and parked in a hidden
    holding slot until a card is expanded -- this is what keeps the guided
    tour's "navigator" step resolvable to a real widget before any dataset
    is loaded.
    """
    dataset_model = DatasetTableModel(DatasetState())
    widget = DataNavigatorSection(dataset_model)
    qtbot.addWidget(widget)

    assert widget.isAncestorOf(widget.annotations)
    assert widget.isAncestorOf(widget.metadata)


def test_left_panel_collapses_to_summary_rail_and_restores(qtbot):
    """The navigator swaps to a toolbar-width rail with synchronized summaries."""
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.load_folder("/fake", ["a.jpg", "b.jpg", "c.jpg"])
    panel = LeftPanel(dataset_model)
    qtbot.addWidget(panel)
    panel.resize(220, 600)
    panel.show()
    panel.set_counter(1, 3)

    panel.set_collapsed(True)

    assert panel.is_collapsed()
    assert panel.width() == 56
    assert panel._collapsed_rail._counter_lbl.text() == "2/3"
    assert panel._collapsed_rail._undecided_count.text() == "3"

    panel.set_collapsed(False)

    assert not panel.is_collapsed()
    assert panel.minimumWidth() == 160
    assert panel.maximumWidth() > 160


def test_collapsed_navigation_buttons_forward_requests(qtbot):
    """Collapsed previous/next controls use the normal navigation signals."""
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.load_folder("/fake", ["a.jpg"])
    panel = LeftPanel(dataset_model)
    qtbot.addWidget(panel)
    panel.set_collapsed(True)

    with qtbot.waitSignal(panel.prev_requested, timeout=1000):
        qtbot.mouseClick(panel._collapsed_rail._btn_prev, Qt.LeftButton)
    with qtbot.waitSignal(panel.next_requested, timeout=1000):
        qtbot.mouseClick(panel._collapsed_rail._btn_next, Qt.LeftButton)


def test_left_panel_defaults_to_collapsed_with_greyed_controls_when_no_data(qtbot):
    """No dataset loaded yet -- panel starts collapsed with disabled rail buttons."""
    dataset_model = DatasetTableModel(DatasetState())
    panel = LeftPanel(dataset_model)
    qtbot.addWidget(panel)

    assert panel.is_collapsed()
    assert not panel._collapsed_rail._btn_expand.isEnabled()
    assert not panel._collapsed_rail._btn_prev.isEnabled()
    assert not panel._collapsed_rail._btn_next.isEnabled()


def test_left_panel_expands_and_enables_controls_once_data_loads(qtbot):
    """Loading a dataset into an empty panel flips it to expanded with live controls."""
    dataset_model = DatasetTableModel(DatasetState())
    panel = LeftPanel(dataset_model)
    qtbot.addWidget(panel)
    assert panel.is_collapsed()

    dataset_model.load_folder("/fake", ["a.jpg"])

    assert not panel.is_collapsed()
    assert panel._collapsed_rail._btn_expand.isEnabled()
    assert panel._collapsed_rail._btn_prev.isEnabled()
    assert panel._collapsed_rail._btn_next.isEnabled()


def test_left_panel_defaults_to_expanded_when_data_already_present(qtbot):
    """Constructing the panel against an already-loaded dataset skips the collapsed default."""
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.load_folder("/fake", ["a.jpg"])
    panel = LeftPanel(dataset_model)
    qtbot.addWidget(panel)

    assert not panel.is_collapsed()
