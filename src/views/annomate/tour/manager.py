"""TourManager — orchestrates the first-run guided tour.

Owns the step list, current step index, the lazily-built TourOverlay, and
the QSettings flag that records whether the tour has been completed/skipped.
"""

import logging
from typing import Optional

from PySide6.QtCore import QObject, QSettings

from .overlay import TourOverlay
from .steps import TourStep, build_default_steps

logger = logging.getLogger(__name__)

_TOUR_COMPLETED_KEY = "onboarding/tour_completed"


class TourManager(QObject):
    """Drives the guided tour for a single AnnoMateWindow instance.

    Args:
        main_window: The AnnoMateWindow the tour highlights widgets on.
        settings: Optional QSettings override (tests inject an isolated
            instance instead of the real "LANL"/"AnnoMateMicroSentryAI" store).
        steps: Optional step list override (defaults to build_default_steps()).
        parent: Optional QObject parent.
    """

    def __init__(
        self,
        main_window,
        settings: Optional[QSettings] = None,
        steps: Optional[list[TourStep]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent or main_window)
        self._main_window = main_window
        self._settings = (
            settings
            if settings is not None
            else QSettings("LANL", "AnnoMateMicroSentryAI")
        )
        self._steps = steps if steps is not None else build_default_steps()
        self._index = -1
        self._overlay: Optional[TourOverlay] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def should_run(self) -> bool:
        """True if the tour has never been completed or skipped before."""
        return not self._settings.value(_TOUR_COMPLETED_KEY, False, type=bool)

    def is_active(self) -> bool:
        return self._overlay is not None

    def start(self) -> None:
        """Show the tour from the first step, building the overlay if needed."""
        if not self._steps:
            return
        if self._overlay is None:
            self._overlay = TourOverlay(self._main_window)
            self._overlay.next_requested.connect(self._on_next)
            self._overlay.back_requested.connect(self._on_back)
            self._overlay.skip_requested.connect(self.skip)
        self._index = 0
        self._show_current()
        logger.info("Guided tour started (%d steps)", len(self._steps))

    def skip(self) -> None:
        """Dismiss the tour immediately and mark it complete."""
        self._complete()

    def reposition(self) -> None:
        """Re-sync the overlay geometry after the main window resizes. No-op if inactive."""
        if self._overlay is not None:
            self._overlay.reposition(self._main_window)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _on_next(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._complete()
        else:
            self._index += 1
            self._show_current()

    def _on_back(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def _show_current(self) -> None:
        step = self._steps[self._index]
        self._overlay.show_step(
            step, self._index, len(self._steps), can_go_back=self._index > 0
        )

    def _complete(self) -> None:
        self._settings.setValue(_TOUR_COMPLETED_KEY, True)
        if self._overlay is not None:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None
        self._index = -1
        logger.info("Guided tour completed/skipped")
