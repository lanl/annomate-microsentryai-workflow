"""Static data describing the first-run guided tour's steps."""

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtWidgets import QWidget

TargetResolver = Callable[[QWidget], Optional[QWidget]]


@dataclass(frozen=True)
class TourStep:
    """One stop on the guided tour.

    Args:
        key: Stable identifier for logging/tests.
        title: Callout heading.
        body: Callout description text.
        target: Callable that resolves the widget to highlight, given the
            AnnoMateWindow instance. None means a centered card with no
            spotlight (used for the welcome/outro steps).
        placement: Preferred callout placement hint. Currently informational —
            the overlay applies its own auto-placement heuristic regardless.
    """

    key: str
    title: str
    body: str
    target: Optional[TargetResolver] = None
    placement: str = "auto"


def build_default_steps() -> list[TourStep]:
    """Return the tour steps in presentation order."""
    return [
        TourStep(
            key="welcome",
            title="Welcome to AnnoMate & MicroSentryAI",
            body=(
                "This quick tour highlights where the major features live. "
                "You can skip at any time, and replay it later from "
                "Help → Show Welcome Tour."
            ),
            placement="center",
        ),
        TourStep(
            key="canvas",
            title="Main Canvas",
            body=(
                "Once you open an image folder, your images appear here. "
                "Scroll to zoom, right-click and drag to pan."
            ),
            target=lambda w: w.canvas,
        ),
        TourStep(
            key="tool_palette",
            title="Annotation Tools",
            body=(
                "Pick your drawing tool here: Polygon (P) to draw manually, "
                "or SAM Segment (S) to let AI segment an object from a box "
                "you draw."
            ),
            target=lambda w: w.tool_palette,
            placement="right",
        ),
        TourStep(
            key="viewport_actions",
            title="Viewport Controls",
            body=(
                "Zoom and view controls live in this floating bar."
            ),
            target=lambda w: w.viewport_actions,
            placement="above",
        ),
        TourStep(
            key="navigator",
            title="Dataset Navigator",
            body=(
                "Browse every loaded image here. Colored status dots show "
                "which images are reviewed, pending, or incomplete. Click a "
                "card to expand it and see its annotations, inspector, and "
                "notes inline."
            ),
            target=lambda w: w.left_panel.navigator_header(),
            placement="right",
        ),
        TourStep(
            key="classes",
            title="Annotation Classes",
            body=(
                "Define your defect classes and pick a color for each, "
                "e.g. 'crack' or 'scratch'."
            ),
            target=lambda w: w.right_panel.classes_header(),
            placement="left",
        ),
        TourStep(
            key="status_bar",
            title="Status Bar",
            body=(
                "Keep an eye on zoom level, the active tool/class, and "
                "MicroSentryAI batch progress here."
            ),
            target=lambda w: w.status_bar,
            placement="above",
        ),
        TourStep(
            key="microsentry",
            title="MicroSentryAI",
            body=(
                "Load a custom MicroSentryAI model here to get AI-powered "
                "defect heatmaps and suggested polygons across your whole "
                "dataset."
            ),
            target=lambda w: w.right_panel.microsentry_header(),
            placement="left",
        ),
        TourStep(
            key="outro",
            title="You're all set",
            body=(
                "The File and Data menus above hold project save/open and "
                "export options. Replay this tour anytime from "
                "Help → Show Welcome Tour."
            ),
            placement="center",
        ),
    ]
