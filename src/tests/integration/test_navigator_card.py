import pytest
from PySide6.QtCore import Qt

from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from models.navigator_model import NavigatorTableModel
from views.annomate.sections._navigator_card import (
    _ClassPillTray,
    _NavigatorCard,
    _make_pill,
)
from views.annomate.sections._shared import _COLOR_SELECTED_BG


@pytest.fixture
def dataset_model(tmp_path):
    model = DatasetTableModel(DatasetState())
    model.load_folder(str(tmp_path), ["a.jpg"])
    model.add_class("scratch", (10, 20, 30))
    model.add_class("inclusion", (40, 50, 60))
    return model


@pytest.fixture
def nav_model(dataset_model):
    return NavigatorTableModel(dataset_model)


def pill_texts(tray):
    return [
        tray._layout.itemAt(i).widget().text()
        for i in range(tray._layout.count())
        if tray._layout.itemAt(i).widget() is not None
        and not tray._layout.itemAt(i).widget().isHidden()
    ]


def test_pill_tray_shows_all_pills_that_fit(qtbot):
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.resize(400, 20)

    tray.set_classes([("inclusion", (40, 50, 60)), ("scratch", (10, 20, 30))])

    assert pill_texts(tray) == ["inclusion", "scratch"]


def test_pill_tray_drops_overflow_pills_when_width_is_tight(qtbot):
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.set_classes([("inclusion", (40, 50, 60)), ("scratch", (10, 20, 30))])

    tray.resize(30, 20)  # too narrow for even one full pill in some cases; keep tiny
    tray._refit()  # resize() with an unchanged size may not fire resizeEvent

    assert len(pill_texts(tray)) <= 1


def test_pill_tray_sizehint_stays_pinned_regardless_of_pill_count(qtbot):
    """Regression: the tray's own sizeHint must never grow with its pills.

    It previously didn't override sizeHint(), so rebuilding pills in
    _refit() changed the tray's own size hint. That fed back into the
    parent layout's space negotiation, which resized the tray again and
    retriggered _refit() -- an oscillating layout loop that never settled
    for the multi-pill (packed-left, non-centered) case.
    """
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.resize(400, 20)

    empty_hint = tray.sizeHint().width()
    tray.set_classes([("inclusion", (40, 50, 60)), ("scratch", (10, 20, 30))])
    with_pills_hint = tray.sizeHint().width()

    assert empty_hint == 0
    assert with_pills_hint == 0


def test_navigator_card_height_stabilizes_with_multiple_classes(
    qtbot, nav_model, dataset_model
):
    """Regression: multiple class pills used to cause an unstable resize loop.

    Two distinct classes take the packed-left (non-centered) pill path,
    which was the one that never settled before the sizeHint fix. If the
    card's height is still changing after a couple of event-loop passes,
    the layout is thrashing.
    """
    dataset_model.add_annotation(0, "scratch", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(0, "inclusion", [(0, 0), (2, 0), (2, 2)])
    card = _NavigatorCard(0, nav_model)
    qtbot.addWidget(card)
    card.resize(300, card.sizeHint().height())
    card.show()
    qtbot.wait(50)

    first_height = card.height()
    qtbot.wait(100)
    second_height = card.height()

    assert first_height == second_height


def test_pill_tray_packs_single_pill_left_with_trailing_space(qtbot):
    """A lone pill sits flush left (next to the divider), not centered."""
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.resize(400, 20)

    tray.set_classes([("scratch", (10, 20, 30))])

    assert tray._layout.count() == 2
    assert tray._layout.itemAt(0).widget().text() == "scratch"
    assert tray._layout.itemAt(1).widget() is None  # trailing stretch


def test_pill_tray_packs_multiple_pills_left_with_trailing_space(qtbot):
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.resize(400, 20)

    tray.set_classes([("inclusion", (40, 50, 60)), ("scratch", (10, 20, 30))])

    assert tray._layout.count() == 3
    assert tray._layout.itemAt(0).widget().text() == "inclusion"
    assert tray._layout.itemAt(1).widget().text() == "scratch"
    assert tray._layout.itemAt(2).widget() is None  # trailing stretch


def test_pill_text_is_centered(qtbot):
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.resize(400, 20)

    tray.set_classes([("scratch", (10, 20, 30))])

    pill = tray._layout.itemAt(0).widget()
    assert pill.alignment() & Qt.AlignCenter


def test_pill_tray_empty_when_no_classes(qtbot):
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.resize(400, 20)

    tray.set_classes([])

    assert pill_texts(tray) == []


def test_pill_tray_reuse_across_rows_hides_and_reuses_stale_pill(qtbot):
    """Flyweight repainting must not create/destroy pills during scrolling."""
    tray = _ClassPillTray()
    qtbot.addWidget(tray)
    tray.resize(400, 20)

    tray.set_classes([("nick", (10, 20, 30))])
    old_pill = tray._layout.itemAt(0).widget()

    tray.set_classes([])  # simulates painting the next (class-less) row

    assert old_pill.isHidden()
    assert old_pill.parent() is tray

    tray.set_classes([("nick", (10, 20, 30))])

    assert tray._layout.itemAt(0).widget() is old_pill
    assert not old_pill.isHidden()


def test_pill_style_uses_class_color_border_and_black_text():
    pill = _make_pill("scratch", (10, 20, 30))
    style = pill.styleSheet()

    assert "rgb(10, 20, 30)" in style
    assert "color: black" in style


def test_navigator_card_dividers_hidden_without_classes(qtbot, nav_model):
    card = _NavigatorCard(0, nav_model)
    qtbot.addWidget(card)

    # isVisible() alone is unreliable pre-show(); isVisibleTo() checks the
    # widget's own flag regardless of whether the card itself is shown.
    assert card._pill_divider_left.isVisibleTo(card) is False
    assert card._pill_divider_right.isVisibleTo(card) is False


def test_collapsed_card_hover_matches_expanded_background(qtbot, nav_model):
    card = _NavigatorCard(0, nav_model)
    qtbot.addWidget(card)

    card.set_hovered(True)
    assert _COLOR_SELECTED_BG in card._header.styleSheet()

    card.set_hovered(False)
    assert _COLOR_SELECTED_BG not in card._header.styleSheet()


def test_navigator_card_shows_left_divider_with_classes_regardless_of_microsentry(
    qtbot, nav_model, dataset_model
):
    dataset_model.add_annotation(0, "scratch", [(0, 0), (1, 0), (1, 1)])
    card = _NavigatorCard(0, nav_model)
    qtbot.addWidget(card)

    assert card._pill_divider_left.isVisibleTo(card) is True
    # Score/right divider stay hidden outside MicroSentry mode.
    assert card._pill_divider_right.isVisibleTo(card) is False

    card.set_microsentry_mode(True)

    assert card._pill_divider_right.isVisibleTo(card) is True


def test_annot_badge_shows_polygon_icon_and_tooltip_in_pixel_mode(
    qtbot, nav_model, dataset_model
):
    dataset_model.add_annotation(0, "scratch", [(0, 0), (1, 0), (1, 1)])
    card = _NavigatorCard(0, nav_model)
    qtbot.addWidget(card)

    assert card._annot_icon_lbl.toolTip() == "Has annotations"
    assert card._annot_icon_lbl.isVisibleTo(card) is True


def test_annot_badge_shows_label_icon_and_tooltip_in_image_level_mode(
    qtbot, nav_model, dataset_model
):
    """Image-level mode's badge is a label/tag icon, not the pixel polygon icon.

    A pixel annotation alone (no image class tag) must NOT surface the
    badge -- image-level mode's "has work" signal is class tags, not
    polygons, matching class_entries()'s mode-aware source of truth. The
    annotation is added after the mode switch so the one-time
    pixel-to-image-tag migration doesn't pre-populate the tag.
    """
    dataset_model.set_annotation_mode("image_level")
    dataset_model.add_annotation(0, "scratch", [(0, 0), (1, 0), (1, 1)])
    card = _NavigatorCard(0, nav_model)
    qtbot.addWidget(card)

    assert card._annot_icon_lbl.isVisibleTo(card) is False

    dataset_model.set_image_classes(0, ["scratch"])
    card.refresh()

    assert card._annot_icon_lbl.toolTip() == "Has class tags"
    assert card._annot_icon_lbl.isVisibleTo(card) is True


def test_collapsed_render_refits_pills_after_row_layout_changes(
    qtbot, nav_model, dataset_model
):
    """A flyweight row must fit pills using its current, not previous, geometry."""
    dataset_model.add_annotation(0, "scratch", [(0, 0), (1, 0), (1, 1)])
    card = _NavigatorCard(0, nav_model, microsentry_mode=True)
    qtbot.addWidget(card)

    # Simulate stale geometry inherited from a previously painted, cramped row.
    card._pill_tray.resize(1, 20)
    card.refresh()
    assert pill_texts(card._pill_tray) == []

    card.prepare_collapsed_render(400, card.sizeHint().height())

    assert pill_texts(card._pill_tray) == ["scratch"]
    pill = card._pill_tray._layout.itemAt(0).widget()
    assert pill.width() <= card._pill_tray.width()
    assert pill.height() <= card._pill_tray.height()
