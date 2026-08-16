import pytest
from PySide6.QtCore import QSettings, Qt

from controllers.io_controller import IOController
from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from views.annomate.tour.manager import TourManager
from views.annomate.tour.steps import build_default_steps
from views.annomate.window import AnnoMateWindow


@pytest.fixture
def isolated_settings():
    """A QSettings store scoped away from the developer's real app settings."""
    settings = QSettings(
        QSettings.IniFormat, QSettings.UserScope, "AnnoMateTourTest", "AnnoMateTourTest"
    )
    settings.clear()
    yield settings
    settings.clear()


def _build_window(qtbot, monkeypatch, settings):
    # These tests only exercise tour UI wiring. Without this, AnnoMateWindow's
    # startup autoload would spin up a real background QThread whenever this
    # machine has SAM weights cached on disk — one that can outlive the window
    # (nothing here calls shutdown()) and later deliver a signal into an
    # already-deleted window, crashing the process. Also makes these tests'
    # behavior depend on local disk state, which they shouldn't.
    monkeypatch.setattr(
        "controllers.sam_controller.SAMController.try_autoload",
        lambda self, variant: False,
    )
    dataset_model = DatasetTableModel(DatasetState())
    io_controller = IOController(dataset_model)
    win = AnnoMateWindow(dataset_model, io_controller)
    win._tour_manager = TourManager(win, settings=settings, parent=win)
    qtbot.addWidget(win)
    win.resize(1200, 800)
    return win


@pytest.fixture
def main_window(qtbot, monkeypatch, isolated_settings):
    """A real AnnoMateWindow with an isolated tour settings store.

    The tour no longer auto-launches on show — tests drive TourManager
    explicitly for deterministic behavior.
    """
    win = _build_window(qtbot, monkeypatch, isolated_settings)
    win.show()
    qtbot.waitExposed(win)
    return win


class TestTourManager:
    def test_fresh_settings_should_run(self, main_window):
        """Verify a brand-new settings store means the tour has not been seen yet.

        Success means should_run() is True before start()/skip() are ever called.
        """
        assert main_window._tour_manager.should_run() is True

    def test_skip_marks_completed_and_tears_down_overlay(self, main_window):
        """Verify Skip immediately dismisses the tour and persists completion.

        Skipping mid-tour (not on the last step) must still mark the tour as
        seen and release the overlay widget, satisfying "skippable at any point."
        """
        manager = main_window._tour_manager
        manager.start()
        assert manager.is_active() is True

        manager.skip()

        assert manager.is_active() is False
        assert manager.should_run() is False

    def test_each_step_target_resolves_to_a_real_descendant_widget(self, main_window):
        """Verify every non-welcome/outro step's target resolves to a widget in the window.

        Success means calling step.target(main_window) returns a non-None
        widget that main_window.isAncestorOf(...) confirms is a descendant.
        """
        for step in build_default_steps():
            if step.target is None:
                assert step.key in ("welcome", "outro")
                continue
            widget = step.target(main_window)
            assert widget is not None
            assert main_window.isAncestorOf(widget)

    def test_escape_key_skips_tour(self, main_window, qtbot):
        """Verify pressing Escape while the tour is showing skips it.

        Success means should_run() becomes False after the key press.
        """
        manager = main_window._tour_manager
        manager.start()

        qtbot.keyClick(manager._overlay, Qt.Key_Escape)

        assert manager.is_active() is False
        assert manager.should_run() is False

    def test_next_navigates_through_all_steps_then_completes(self, main_window):
        """Verify repeatedly advancing Next walks every step and finishes the tour.

        Success means the tour is still active after every step except the
        last, and becomes inactive (completed) exactly on the final Next.
        """
        manager = main_window._tour_manager
        manager.start()
        total = len(manager._steps)

        for _ in range(total - 1):
            manager._on_next()
            assert manager.is_active() is True

        manager._on_next()

        assert manager.is_active() is False
        assert manager.should_run() is False

    def test_back_unavailable_on_first_step_then_available_after_advancing(
        self, main_window
    ):
        """Verify Back only takes effect once the user has moved past step one.

        Success means _on_back() is a no-op while on the first step (index
        stays 0), but moves the index back by one after advancing forward.
        """
        manager = main_window._tour_manager
        manager.start()
        assert manager._index == 0

        manager._on_back()
        assert manager._index == 0

        manager._on_next()
        assert manager._index == 1

        manager._on_back()
        assert manager._index == 0

    def test_start_tour_force_replays_after_completion(self, main_window):
        """Verify start_tour(force=True) can replay the tour even once already seen.

        Success means start_tour(force=True) reactivates the tour regardless
        of should_run()'s value.
        """
        main_window._tour_manager.skip()
        assert main_window._tour_manager.should_run() is False

        main_window.start_tour(force=True)

        assert main_window._tour_manager.is_active() is True

    def test_reposition_matches_overlay_geometry_to_window(self, main_window):
        """Verify reposition() keeps the overlay's geometry matched to the window after a resize.

        Success means the overlay's geometry equals the window's rect after
        resizing and calling reposition() explicitly.
        """
        manager = main_window._tour_manager
        manager.start()

        main_window.resize(900, 600)
        manager.reposition()

        assert manager._overlay.geometry() == main_window.rect()
