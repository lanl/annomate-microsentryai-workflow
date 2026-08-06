from unittest.mock import MagicMock

from PySide6.QtCore import Qt

from main import _configure_theme
from views.annomate.sections._shared import TOOLTIP_STYLESHEET


def test_tooltip_stylesheet_uses_navigator_selected_row_background():
    """The app-wide tooltip look is the same one the navigator originated.

    Locks in the specific values (light grey background, black text, no
    rounded corners) so a future edit to this constant is a deliberate,
    visible diff rather than an accidental drift.
    """
    assert "QToolTip" in TOOLTIP_STYLESHEET
    assert "background-color: #d6d6d6" in TOOLTIP_STYLESHEET
    assert "color: black" in TOOLTIP_STYLESHEET
    assert "border-radius" not in TOOLTIP_STYLESHEET


def test_configure_theme_forces_fusion_light_and_tooltip_stylesheet():
    """_configure_theme wires Fusion + light scheme + the shared tooltip QSS.

    Uses a mock instead of a real QApplication -- setStyle()/setStyleSheet()
    mutate the process-wide Qt application singleton, which would leak into
    every other test in the session if exercised for real here.
    """
    app = MagicMock()

    _configure_theme(app)

    app.setStyle.assert_called_once_with("Fusion")
    app.styleHints.return_value.setColorScheme.assert_called_once_with(
        Qt.ColorScheme.Light
    )
    app.setStyleSheet.assert_called_once_with(TOOLTIP_STYLESHEET)
