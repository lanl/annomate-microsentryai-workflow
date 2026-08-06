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


def click_proxy_row(widget, qtbot, proxy_row):
    """Simulate a real click on the collapsed (delegate-painted) row at *proxy_row*."""
    index = widget._proxy.index(proxy_row, NavigatorColumns.IMG_ID)
    rect = widget._list.visualRect(index)
    qtbot.mouseClick(widget._list.viewport(), Qt.LeftButton, pos=rect.center())


def test_list_does_not_capture_window_navigation_hotkeys(navigator):
    """The virtualized view must leave A/D handling to AnnoMateWindow."""
    widget, _dataset_model, _inference_model, _tmp_path = navigator

    assert widget._list.focusPolicy() == Qt.NoFocus


def test_collapsed_delegate_exposes_card_and_icon_tooltips(navigator):
    """Painted rows retain the tooltip behavior of their real card widgets."""
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
    delegate = widget._delegate
    delegate._flyweight.set_source_row(0)
    delegate._flyweight.prepare_collapsed_render(400, delegate._collapsed_height)

    card = delegate._flyweight
    icon_pos = card._annot_icon_lbl.mapTo(
        card, card._annot_icon_lbl.rect().center()
    )

    assert delegate._tooltip_at(icon_pos) == "Has annotations"
    assert delegate._tooltip_at(card._header.rect().topLeft()) == card._header.toolTip()


def test_navigator_tooltips_use_selected_card_background(navigator):
    widget, _dataset_model, _inference_model, _tmp_path = navigator

    assert "QToolTip" in widget.styleSheet()
    assert "background-color: #d6d6d6" in widget.styleSheet()
    assert "}}" not in widget.styleSheet()
    tooltip_rule = widget.styleSheet().split("}", 1)[0]
    assert "border-radius" not in tooltip_rule


def test_navigator_header_hover_uses_selected_card_background(navigator):
    widget, _dataset_model, _inference_model, _tmp_path = navigator

    for button in (widget._btn_prev, widget._btn_next, widget._btn_filter):
        assert "QToolButton:hover" in button.styleSheet()
        assert "background-color: #d6d6d6" in button.styleSheet()
    assert "QFrame#navigatorFilterChip:hover" in widget.styleSheet()


def test_collapsed_delegate_caches_rows_and_invalidates_changed_row(navigator):
    """Scrolling reuses rendered rows while model edits refresh stale pixels."""
    widget, dataset_model, _inference_model, _tmp_path = navigator
    delegate = widget._delegate

    first = delegate._row_pixmap(0, 330, False, 1.0)
    second = delegate._row_pixmap(0, 330, False, 1.0)
    assert first.cacheKey() == second.cacheKey()

    dataset_model.add_annotation(0, "Defect", [(0, 0), (1, 0), (1, 1)])
    refreshed = delegate._row_pixmap(0, 330, False, 1.0)

    assert refreshed.cacheKey() != first.cacheKey()


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


def test_clicking_top_row_emits_source_row(navigator, qtbot):
    """Verify that clicking the visually-first row emits image_selected with its source row.

    Adds two annotations to source row 2 ('c.jpg'), sorts descending by
    annotation count so 'c.jpg' rises to the top, then clicks that row (it's
    delegate-painted, not a real widget, so the click goes through
    QListView's viewport like a real user click would).
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.add_annotation(2, "Defect", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(2, "Defect", [(0, 0), (2, 0), (2, 2)])
    widget._proxy.sort(NavigatorColumns.ANNOTS, Qt.DescendingOrder)
    qtbot.wait(20)

    index = widget._proxy.index(0, NavigatorColumns.IMG_ID)
    assert widget._source_row_from_proxy(index) == 2

    with qtbot.waitSignal(widget.image_selected, timeout=1000) as blocker:
        click_proxy_row(widget, qtbot, 0)

    assert blocker.args == [2]


def test_select_row_expands_only_that_row_after_sort(navigator):
    """Verify that select_row expands the correct row after a sort and supports adjacent navigation.

    After ascending sort by image ID (a=0, b=1, c=2), calls select_row with
    different source rows and confirms only the final selection ends up
    attached to the single shared expanded-card widget.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget._proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

    widget.select_row(0)
    widget.select_row(2)
    widget.select_row(0)

    assert widget._selected_row == 0
    assert widget._expanded_card.source_row() == 0
    assert widget._expanded_card.is_expanded() is True
    assert widget.adjacent_source_row(0, -1) == 1
    assert widget.adjacent_source_row(0, 1) == 2


def test_select_row_scrolls_expanded_row_to_top(navigator, qtbot):
    """Verify Prev/Next-style navigation (select_row) pins the active row to the top of the list.

    With three rows in a viewport too short to show them all at once,
    selecting the last row should scroll the list so that row's top edge
    sits at the very top of the visible area, matching the A/D keyboard
    navigation expectation that the current image's row stays anchored at
    the top instead of landing somewhere in the middle or bottom.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget._proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

    widget.select_row(2)
    qtbot.wait(20)

    assert widget._list.verticalScrollBar().maximum() > 0  # something to scroll past
    index = widget._proxy.index(
        widget._proxy_row_from_source(2), NavigatorColumns.IMG_ID
    )
    assert widget._list.visualRect(index).top() == 0


def test_microsentry_mode_shows_score_and_score_resorts(navigator, qtbot):
    """Verify that microsentry mode reveals the score label and re-sorts by score after inference.

    Initially the score label is hidden, both on the delegate's collapsed-row
    flyweight and on the expanded row's real widget. After enabling
    microsentry mode it becomes visible on both. After storing inference
    results and calling set_row_inference, sorting by score descending
    should place the highest-scoring row (c.jpg, source row 2) at the top.
    """
    widget, _dataset_model, inference_model, tmp_path = navigator
    flyweight = widget._delegate._flyweight
    assert flyweight._score_lbl.isVisibleTo(flyweight) is False

    widget.set_microsentry_mode(True)
    assert flyweight._score_lbl.isVisibleTo(flyweight) is True

    widget.select_row(0)
    assert widget._expanded_card._score_lbl.isVisible() is True

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
    assert widget._proxy.status_filter() == frozenset({"reviewed"})

    widget._on_chip_clicked("reviewed")
    assert set(source_rows(widget)) == {0, 1, 2}
    assert widget._proxy.status_filter() == frozenset()


def test_clicking_a_second_chip_adds_to_rather_than_replaces_the_first(
    navigator, qtbot
):
    """Chips are independent toggles now, not a mutually-exclusive radio group.

    Clicking "reviewed" then "incomplete" should leave both active
    simultaneously, matching the multi-select checkboxes they stay in sync
    with in the Filter menu.
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(0, "accept")  # reviewed
    dataset_model.set_review_decision(1, "reject")  # incomplete (no work)
    qtbot.wait(20)

    widget._on_chip_clicked("reviewed")
    widget._on_chip_clicked("incomplete")

    assert widget._proxy.status_filter() == frozenset({"reviewed", "incomplete"})
    assert set(source_rows(widget)) == {0, 1}


def test_filter_panel_decision_checkbox_filters_to_accepted_rows(navigator, qtbot):
    """Verify the Filter menu's Decision:Accept checkbox filters to accepted rows."""
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(1, "accept")
    qtbot.wait(20)

    widget._filter_panel._decision_checks["accept"].setChecked(True)

    assert source_rows(widget) == [1]
    assert widget._proxy.decision_filter() == frozenset({"accept"})


def test_expanded_card_never_becomes_a_top_level_window(navigator, qtbot):
    """Regression: the expanded-card widget must never flash open as a real
    top-level OS window. It's constructed with the list's viewport passed
    as its parent up front -- unlike the old per-row rebuild path, there's
    no point where a parentless card gets setVisible(True) first.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget.select_row(0)
    assert widget._expanded_card.isWindow() is False
    assert widget._expanded_card.windowHandle() is None


def test_filtering_out_the_expanded_row_collapses_it(navigator, qtbot):
    """Regression: with a fresh expanded-card widget built per selection
    instead of a per-row dict, filtering the currently expanded row out of
    view must explicitly release and delete it -- otherwise it would sit
    around attached to a row that no longer exists in the filtered list.
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(1, "accept")  # row 1 becomes "reviewed"
    qtbot.wait(20)
    widget.select_row(1)
    assert widget._expanded_card is not None
    assert widget._expanded_card.is_expanded() is True

    widget._filter_panel._status_checks["undecided"].setChecked(True)  # hides row 1

    assert widget._selected_row == -1
    assert widget._expanded_card is None


def test_loading_a_smaller_dataset_collapses_out_of_range_selection(navigator, qtbot):
    """Loading a dataset smaller than the previous one, while the now-out-of-
    range row is expanded, must release the expanded-card widget rather than
    leaving our reference pointing at one Qt may have already deleted.
    """
    widget, dataset_model, _inference_model, tmp_path = navigator
    widget.select_row(2)
    qtbot.wait(20)
    assert widget._expanded_card is not None
    assert widget._expanded_card.is_expanded() is True

    dataset_model.load_folder(str(tmp_path), ["d.jpg", "e.jpg"])
    qtbot.wait(20)

    assert widget.dataset_model.rowCount() == 2
    assert widget._selected_row == -1
    assert widget._expanded_card is None


def test_filter_panel_conflicting_checkbox_isolates_accept_conflict_rows(
    navigator, qtbot
):
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.add_annotation(0, "crack", [(0, 0), (1, 0), (1, 1)])
    dataset_model.set_review_decision(0, "accept")  # accept_conflict
    dataset_model.set_review_decision(1, "reject")  # reject_incomplete
    qtbot.wait(20)

    widget._filter_panel._status_checks["conflicting"].setChecked(True)

    assert source_rows(widget) == [0]


def test_chip_and_filter_panel_checkbox_stay_in_sync_bidirectionally(
    navigator, qtbot
):
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(0, "accept")
    qtbot.wait(20)

    widget._on_chip_clicked("reviewed")
    assert widget._filter_panel._status_checks["reviewed"].isChecked() is True

    widget._filter_panel._status_checks["reviewed"].setChecked(False)
    assert widget._proxy.status_filter() == frozenset()
    assert widget._filter_chips["reviewed"].styleSheet() == ""


def test_clear_filters_button_resets_everything_and_badge(navigator, qtbot):
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(0, "accept")
    qtbot.wait(20)

    widget._filter_panel._decision_checks["accept"].setChecked(True)
    widget._on_chip_clicked("incomplete")
    assert widget._btn_filter.text() != "Filter"

    widget._on_clear_filters_clicked()

    assert set(source_rows(widget)) == {0, 1, 2}
    assert widget._proxy.decision_filter() == frozenset()
    assert widget._proxy.status_filter() == frozenset()
    assert widget._btn_filter.text() == "Filter"
    assert widget._filter_panel._decision_checks["accept"].isChecked() is False


def test_filter_button_badge_shows_active_count(navigator, qtbot):
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.set_review_decision(0, "accept")
    qtbot.wait(20)

    widget._filter_panel._decision_checks["accept"].setChecked(True)
    assert widget._btn_filter.text() == "Filter (1)"

    widget._on_chip_clicked("incomplete")
    assert widget._btn_filter.text() == "Filter (2)"


def test_loading_new_dataset_clears_filters(navigator, qtbot):
    widget, dataset_model, _inference_model, tmp_path = navigator
    dataset_model.set_review_decision(0, "accept")
    qtbot.wait(20)
    widget._filter_panel._decision_checks["accept"].setChecked(True)
    assert widget._proxy.decision_filter() == frozenset({"accept"})

    dataset_model.load_folder(str(tmp_path), ["d.jpg"])

    assert widget._proxy.decision_filter() == frozenset()
    assert widget._btn_filter.text() == "Filter"


def test_filter_panel_class_checkbox_filters_by_annotation_class(navigator, qtbot):
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.add_class("crack", (255, 0, 0))
    dataset_model.add_class("scratch", (0, 255, 0))
    dataset_model.add_annotation(0, "crack", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(1, "scratch", [(0, 0), (1, 0), (1, 1)])
    qtbot.wait(20)

    widget._filter_panel._class_checks["crack"].setChecked(True)

    assert source_rows(widget) == [0]
    assert widget._proxy.class_filter() == frozenset({"crack"})
    assert set(source_rows(widget)) == {0}


def test_filter_panel_checkbox_labels_show_image_counts(navigator, qtbot):
    """Row 0 gets a crack annotation with no decision (undecided_work -- bucketed
    under Incomplete). Row 1 is accepted with no work (accept_clean -- Reviewed).
    Row 2 is untouched (Undecided). Each checkbox's label should reflect the
    number of images matching it, not annotation instances.
    """
    widget, dataset_model, _inference_model, _tmp_path = navigator
    dataset_model.add_class("crack", (255, 0, 0))
    dataset_model.add_annotation(0, "crack", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(0, "crack", [(0, 0), (2, 0), (2, 2)])
    dataset_model.set_review_decision(1, "accept")
    qtbot.wait(20)

    assert widget._filter_panel._class_checks["crack"].text() == "crack (1)"
    assert widget._filter_panel._decision_checks["accept"].text() == "Accept (1)"
    assert widget._filter_panel._status_checks["incomplete"].text() == "Incomplete (1)"
    assert widget._filter_panel._status_checks["reviewed"].text() == "Reviewed (1)"
    assert widget._filter_panel._status_checks["undecided"].text() == "Undecided (1)"


def test_filter_panel_shows_placeholder_when_no_classes_annotated_yet(navigator, qtbot):
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    qtbot.wait(20)

    assert widget._filter_panel._class_checks == {}


def test_filter_panel_class_options_update_as_annotations_are_added(navigator, qtbot):
    widget, dataset_model, _inference_model, _tmp_path = navigator
    qtbot.wait(20)
    assert "crack" not in widget._filter_panel._class_checks

    dataset_model.add_class("crack", (255, 0, 0))
    dataset_model.add_annotation(0, "crack", [(0, 0), (1, 0), (1, 1)])
    qtbot.wait(20)

    assert "crack" in widget._filter_panel._class_checks
    assert widget._filter_panel._class_checks["crack"].text() == "crack (1)"


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
    assert "↓" in widget._filter_panel._sort_radios[NavigatorColumns.IMG_ID].text()
    assert ascending != descending

    widget._on_sort_field_chosen(NavigatorColumns.IMG_ID)
    assert widget._sort_order == Qt.AscendingOrder
    assert "↑" in widget._filter_panel._sort_radios[NavigatorColumns.IMG_ID].text()
    assert source_rows(widget) == ascending

    widget._on_sort_field_chosen(NavigatorColumns.ANNOTS)
    assert widget._sort_column == NavigatorColumns.ANNOTS
    assert widget._sort_order == Qt.AscendingOrder
    assert "↑" in widget._filter_panel._sort_radios[NavigatorColumns.ANNOTS].text()


def test_selecting_a_second_row_collapses_the_first_accordion_style(navigator, qtbot):
    """Verify only one row is ever expanded at a time.

    Selecting row B while row A is expanded must retarget the single shared
    expanded-card widget at B, and the shared Annotations/Metadata sections
    move along with it.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget._proxy.sort(NavigatorColumns.IMG_ID, Qt.AscendingOrder)

    widget.select_row(0)
    assert widget._expanded_card.source_row() == 0
    assert widget._expanded_card.is_expanded() is True

    widget.select_row(1)
    assert widget._expanded_card.source_row() == 1
    assert widget._expanded_card.is_expanded() is True
    assert widget.annotations.parent() is widget._expanded_card.body_container()
    assert widget.metadata.parent() is widget._expanded_card.body_container()


def test_clicking_expanded_row_collapses_then_navigation_reexpands(
    navigator, qtbot
):
    """A repeated header click collapses the expanded row; programmatic image navigation re-expands it.

    Each expansion builds a fresh card (see module docstring in navigator.py
    -- collapsing deletes the old one rather than reusing it), so this
    checks widget._expanded_card's identity/state rather than holding onto
    one card instance across the collapse.
    """
    widget, _dataset_model, _inference_model, _tmp_path = navigator
    widget.select_row(0)
    card = widget._expanded_card
    assert card.is_expanded() is True

    qtbot.mouseClick(card._header, Qt.LeftButton)

    assert widget._expanded_card is None
    assert card.is_expanded() is False  # reset before its deferred deleteLater() runs
    assert widget.annotations.parent() is widget._shared_slot
    assert widget.metadata.parent() is widget._shared_slot

    # Window navigation (including A/D) calls select_row for the destination.
    widget.select_row(0)

    assert widget._expanded_card is not None
    assert widget._expanded_card.is_expanded() is True
    assert widget.annotations.parent() is widget._expanded_card.body_container()
    assert widget.metadata.parent() is widget._expanded_card.body_container()


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


def test_collapsed_rail_counter_resets_when_project_cleared(qtbot):
    """Starting a new project (dataset emptied) resets the collapsed counter to placeholders.

    Regression test: the collapsed rail's counter label used to only get
    updated from window.py's per-image load path, which never runs once the
    dataset goes back to zero rows -- leaving the previous project's stale
    "n/total" text on screen instead of resetting.
    """
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.load_folder("/fake", ["a.jpg", "b.jpg"])
    panel = LeftPanel(dataset_model)
    qtbot.addWidget(panel)
    panel.set_counter(1, 2)
    assert panel._collapsed_rail._counter_lbl.text() == "2/2"

    dataset_model.load_folder("/fake", [])  # simulates "New Project" clearing the dataset

    assert panel._collapsed_rail._counter_lbl.text() == "—/—"
    assert panel.is_collapsed()
