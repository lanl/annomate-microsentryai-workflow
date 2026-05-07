"""
AutosaveManager — QTimer-based autosave signal emitter.

Emits save_requested with the current project directory when the timer
fires. The actual write is handled by a ProjectController slot so that
Qt threading rules are respected (everything on the main thread).
"""

import logging

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger("AnnoMate.AutosaveManager")

_DEFAULT_INTERVAL_MINUTES = 5


class AutosaveManager(QObject):
    """Periodic autosave trigger using QTimer.

    Emits save_requested(project_dir) on each timer tick. The connected
    slot performs the actual disk write. The timer only runs when a project
    directory has been set via set_project_dir().

    Signals:
        save_requested (str): Emitted with the project directory path when
            the timer fires and a project is open.
    """

    save_requested = Signal(str)

    def __init__(
        self,
        interval_minutes: int = _DEFAULT_INTERVAL_MINUTES,
        parent: QObject = None,
    ) -> None:
        super().__init__(parent)
        self._project_dir: str = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self.set_interval(interval_minutes)

    def set_project_dir(self, project_dir: str) -> None:
        """Set the current project directory and start the timer if not running.

        Passing an empty string stops the timer.
        """
        self._project_dir = project_dir
        if project_dir:
            if not self._timer.isActive():
                self._timer.start()
                logger.debug("Autosave timer started (%d ms).", self._timer.interval())
        else:
            self.stop()

    def set_interval(self, minutes: int) -> None:
        """Update the autosave interval in minutes. Takes effect immediately."""
        ms = max(1, minutes) * 60 * 1000
        was_active = self._timer.isActive()
        self._timer.setInterval(ms)
        if was_active:
            self._timer.start()

    def start(self) -> None:
        """Start (or restart) the autosave timer."""
        if self._project_dir:
            self._timer.start()

    def stop(self) -> None:
        """Stop the autosave timer."""
        self._timer.stop()

    def is_active(self) -> bool:
        """Return True if the timer is currently running."""
        return self._timer.isActive()

    def _on_timer(self) -> None:
        if self._project_dir:
            logger.debug("Autosave tick — emitting save_requested.")
            self.save_requested.emit(self._project_dir)
