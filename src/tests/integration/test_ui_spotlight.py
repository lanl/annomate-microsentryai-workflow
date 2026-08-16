from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QPushButton, QScrollArea, QVBoxLayout, QWidget

import views.annomate.ui_spotlight as ui_spotlight
from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from views.annomate import right_panel as right_panel_module
from views.annomate.left_panel import LeftPanel
from views.annomate.right_panel import RightPanel
from views.annomate.sections._collapsible import _CollapsibleSection
from views.annomate.ui_spotlight import _SpotlightOverlay, spotlight_widget


def test_spotlight_widget_returns_false_for_none():
    """Verify spotlight_widget(None) does nothing and returns False.

    Success means calling it with no target simply reports failure rather
    than raising.
    """
    assert spotlight_widget(None) is False


def test_spotlight_widget_returns_false_for_invisible_widget(qtbot):
    """Verify spotlight_widget returns False for a widget that isn't shown.

    A widget that was never .show()'d reports isVisible() == False, so
    success means spotlight_widget declines to spotlight it.
    """
    widget = QWidget()
    qtbot.addWidget(widget)

    assert spotlight_widget(widget) is False


def test_spotlight_widget_returns_true_for_visible_widget(qtbot):
    """Verify spotlight_widget returns True synchronously for a visible target.

    The overlay itself is created on a deferred QTimer.singleShot(0, ...),
    but the True/False result is returned immediately based on visibility.
    Success means True comes back without waiting for the event loop.
    """
    parent = QWidget()
    target = QPushButton(parent)
    qtbot.addWidget(parent)
    parent.show()

    assert spotlight_widget(target) is True
    qtbot.wait(50)  # let the deferred reveal/overlay run before the test ends


def test_spotlight_widget_creates_overlay_after_deferred_reveal(qtbot, monkeypatch):
    """Verify the deferred reveal constructs a _SpotlightOverlay for the target.

    _SpotlightOverlay is monkeypatched to a recording stub so this only
    checks *that* it's constructed, and with what arguments, not its
    internal painting/animation behavior. Success means it's called once
    with (target, target.window()) after the event loop processes the
    singleShot(0, ...) callback.
    """
    calls = []
    monkeypatch.setattr(
        ui_spotlight,
        "_SpotlightOverlay",
        lambda target, window: calls.append((target, window)),
    )
    parent = QWidget()
    target = QPushButton(parent)
    qtbot.addWidget(parent)
    parent.show()

    spotlight_widget(target)
    qtbot.wait(50)

    assert len(calls) == 1
    assert calls[0] == (target, target.window())


def test_spotlight_widget_expands_collapsed_collapsible_ancestor(qtbot):
    """Verify a collapsed _CollapsibleSection ancestor is expanded before spotlighting.

    The target lives inside a collapsed section, so it isn't visible until
    the section is expanded. Success means set_expanded(True) is applied
    (section._expanded becomes True) and the widget is found visible.
    """
    root = QWidget()
    root_layout = QVBoxLayout(root)
    section = _CollapsibleSection("Section", expanded=False)
    root_layout.addWidget(section)
    target = QPushButton()
    section.body_layout().addWidget(target)
    qtbot.addWidget(root)
    root.show()
    assert not section._expanded

    result = spotlight_widget(target)

    assert section._expanded is True
    assert result is True
    qtbot.wait(50)  # let the deferred reveal run while `target` is still in scope


def test_spotlight_widget_scrolls_target_into_view(qtbot, monkeypatch):
    """Verify a target inside a QScrollArea is scrolled into view before spotlighting.

    Success means ensureWidgetVisible is called on the ancestor QScrollArea
    with (target, 40, 40), matching spotlight_widget's reveal margins.
    """
    calls = []
    monkeypatch.setattr(
        QScrollArea,
        "ensureWidgetVisible",
        lambda self, widget, xmargin=0, ymargin=0: calls.append(
            (self, widget, xmargin, ymargin)
        ),
    )

    scroll = QScrollArea()
    container = QWidget()
    layout = QVBoxLayout(container)
    target = QPushButton()
    layout.addWidget(target)
    scroll.setWidget(container)
    scroll.setWidgetResizable(True)
    qtbot.addWidget(scroll)
    scroll.show()

    spotlight_widget(target)
    qtbot.wait(50)

    assert len(calls) == 1
    assert calls[0] == (scroll, target, 40, 40)


def test_spotlight_widget_opens_right_panel_tab_for_target(qtbot, monkeypatch):
    """Verify a target inside a non-active RightPanel tab has its tab opened.

    RightPanel starts collapsed on the "classes" tab by default (no prior
    settings). Success means spotlighting a widget that lives on the
    "active_tool" page switches to that tab and un-collapses the panel.
    """
    monkeypatch.setattr(right_panel_module, "_SETTINGS_ORG", "AnnoMateTestOrg")
    monkeypatch.setattr(right_panel_module, "_SETTINGS_APP", "UiSpotlightTest")
    settings = QSettings("AnnoMateTestOrg", "UiSpotlightTest")
    settings.clear()

    dataset_model = DatasetTableModel(DatasetState())
    panel = RightPanel(dataset_model)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.is_collapsed() is True
    target = panel.active_tool.slider_thickness

    result = spotlight_widget(target)

    assert panel.is_collapsed() is False
    assert result is True
    qtbot.wait(50)  # let the deferred reveal run while `target` is still in scope
    settings.clear()


def test_spotlight_widget_uncollapses_left_panel_for_target(qtbot):
    """Verify a target inside a collapsed LeftPanel has the panel expanded.

    LeftPanel starts collapsed whenever no dataset is loaded. Success means
    spotlighting a widget inside the navigator expands the panel first.
    """
    dataset_model = DatasetTableModel(DatasetState())
    panel = LeftPanel(dataset_model)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.is_collapsed() is True
    target = panel.navigator

    result = spotlight_widget(target)

    assert panel.is_collapsed() is False
    assert result is True
    qtbot.wait(50)  # let the deferred reveal run while `target` is still in scope


class TestSpotlightOverlay:
    def test_overlay_geometry_tracks_target_position(self, qtbot):
        """Verify the overlay's geometry updates as the target widget moves.

        The reposition timer runs every _REPOSITION_INTERVAL_MS. Success
        means after the target moves and the timer fires again, the
        overlay's geometry follows it (mapped rect padded by _MARGIN).
        """
        window = QWidget()
        target = QPushButton(window)
        target.move(0, 0)
        target.resize(40, 20)
        qtbot.addWidget(window)
        window.show()

        overlay = _SpotlightOverlay(target, window)
        try:
            target.move(50, 60)
            qtbot.wait(ui_spotlight._REPOSITION_INTERVAL_MS * 3)

            top_left = target.mapTo(window, target.rect().topLeft())
            expected_x = top_left.x() - ui_spotlight._MARGIN
            expected_y = top_left.y() - ui_spotlight._MARGIN
            assert overlay.geometry().topLeft().x() == expected_x
            assert overlay.geometry().topLeft().y() == expected_y
        finally:
            overlay._finish()
            qtbot.wait(10)  # flush the deleteLater() scheduled by _finish()

    def test_overlay_finishes_when_target_becomes_invisible(self, qtbot):
        """Verify the overlay closes itself once its target is hidden.

        Success means after hiding the target and letting the reposition
        timer fire again, the overlay is no longer visible.
        """
        window = QWidget()
        target = QPushButton(window)
        qtbot.addWidget(window)
        window.show()

        overlay = _SpotlightOverlay(target, window)
        target.hide()
        qtbot.wait(ui_spotlight._REPOSITION_INTERVAL_MS * 3)

        try:
            assert not overlay.isVisible()
        except RuntimeError:
            pass  # already deleted, which also satisfies "no longer shown"
        qtbot.wait(10)  # flush any deleteLater() the self-finish scheduled

    def test_overlay_self_cleans_up_after_pulse_duration(self, qtbot, monkeypatch):
        """Verify the overlay closes and deletes itself once its pulses finish.

        Pulse timing is monkeypatched down to a few milliseconds so the test
        doesn't have to wait out the real ~2.85s duration. Success means the
        overlay becomes invisible on its own with no manual _finish() call.
        """
        monkeypatch.setattr(ui_spotlight, "_PULSE_DURATION_MS", 20)
        monkeypatch.setattr(ui_spotlight, "_PULSE_COUNT", 1)

        window = QWidget()
        target = QPushButton(window)
        qtbot.addWidget(window)
        window.show()

        overlay = _SpotlightOverlay(target, window)

        def _is_gone():
            try:
                return not overlay.isVisible()
            except RuntimeError:
                return True

        qtbot.waitUntil(_is_gone, timeout=2000)
        qtbot.wait(10)  # flush the deleteLater() scheduled by _finish()
