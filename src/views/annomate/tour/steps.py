"""Static data describing the first-run guided tour's steps."""

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtWidgets import QWidget

TargetResolver = Callable[[QWidget], Optional[QWidget]]
StepHook = Callable[[QWidget], None]


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
        on_enter: Optional callable run (given the AnnoMateWindow instance)
            right before this step is shown. Used to force open UI state
            that would otherwise hide the target (e.g. an auto-collapsed
            panel).
        on_exit: Optional callable run right after this step is left,
            typically to restore whatever on_enter changed.
    """

    key: str
    title: str
    body: str
    target: Optional[TargetResolver] = None
    placement: str = "auto"
    on_enter: Optional[StepHook] = None
    on_exit: Optional[StepHook] = None


def _left_panel_hooks(collapsed: bool) -> tuple[StepHook, StepHook]:
    """Force the left panel open (or closed) for a step, then restore its prior state.

    The left panel auto-collapses to a narrow icon rail whenever no dataset
    is loaded (see LeftPanel._on_dataset_reset) -- exactly the state at
    first launch, when the tour auto-starts. Without this, a step targeting
    something inside the expanded view is hidden and the overlay silently
    falls back to a centered card with no spotlight -- and conversely, a
    step targeting the collapsed rail needs the panel forced *closed*
    instead, in case it's already expanded. Each call returns its own
    independent hook pair, so consecutive steps with different needs don't
    fight over shared state.
    """
    prior_collapsed: list[bool] = []

    def on_enter(w: QWidget) -> None:
        prior_collapsed.append(w.left_panel.is_collapsed())
        w.left_panel.set_collapsed(collapsed)

    def on_exit(w: QWidget) -> None:
        if prior_collapsed:
            w.left_panel.set_collapsed(prior_collapsed.pop())

    return on_enter, on_exit


def _overlay_section_hooks(
    section_getter: Callable[[QWidget], QWidget],
) -> tuple[StepHook, StepHook]:
    """Open the View Overlays tab and expand one specific section for a step.

    Center Crop, Grid, and Anomaly Constraints all start collapsed and live
    inside the right panel's "View Overlays" tab, which itself starts
    collapsed until a project is open -- the same hidden-target problem the
    navigator has. show_tab() opens the tab without persisting the change,
    so restore_last_state() on exit reliably puts the real, previously-saved
    tab/collapsed state back (see RightPanel.show_tab's docstring).
    """
    prior_expanded: list[bool] = []

    def on_enter(w: QWidget) -> None:
        w.right_panel.show_tab("overlays")
        section = section_getter(w)
        prior_expanded.append(section.is_expanded())
        section.set_expanded(True)

    def on_exit(w: QWidget) -> None:
        section = section_getter(w)
        if prior_expanded:
            section.set_expanded(prior_expanded.pop())
        w.right_panel.restore_last_state()

    return on_enter, on_exit


def build_default_steps() -> list[TourStep]:
    """Return the tour steps in presentation order."""
    navigator_on_enter, navigator_on_exit = _left_panel_hooks(collapsed=False)
    markers_on_enter, markers_on_exit = _left_panel_hooks(collapsed=True)
    center_crop_on_enter, center_crop_on_exit = _overlay_section_hooks(
        lambda w: w.right_panel.center_crop_section()
    )
    grid_on_enter, grid_on_exit = _overlay_section_hooks(
        lambda w: w.right_panel.grid_section()
    )
    anomaly_on_enter, anomaly_on_exit = _overlay_section_hooks(
        lambda w: w.right_panel.anomaly_constraints_section()
    )

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
            key="project_start",
            title="Start a Project",
            body=(
                "New here? Use Open Project to reopen a saved .annoproj file, "
                "or Open Image Folder to start annotating a folder of images. "
                "You can always start fresh from File → New Project."
            ),
            target=lambda w: w.start_screen(),
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
                "Browse every loaded image here. Click a card to expand it "
                "and see its annotations, inspector, and notes inline, and "
                "use the Filter button to narrow the list by status, "
                "decision, or class, or change the sort order."
            ),
            target=lambda w: w.left_panel,
            placement="right",
            on_enter=navigator_on_enter,
            on_exit=navigator_on_exit,
        ),
        TourStep(
            key="navigator_markers",
            title="Review Status Markers",
            body=(
                "Collapse the navigator (the arrow at its top) to save "
                "canvas space, and you'll still see this at a glance on the "
                "icon rail: the counter tracks your position (e.g. \"1/3\") "
                "as you step through images with Prev/Next or the A/D keys, "
                "and each image is one of three statuses with a live count "
                "below it -- ○ Undecided has no Accept/Reject decision yet; "
                "● Reviewed is accepted, or rejected with a polygon or class "
                "tag attached; ! Incomplete needs a second look -- rejected "
                "without supporting evidence, accepted but still has "
                "annotations, or has annotation work with no decision. Set "
                "the decision itself with the Accept/Reject bar in the "
                "canvas's top-right corner."
            ),
            target=lambda w: w.left_panel.collapsed_rail(),
            placement="right",
            on_enter=markers_on_enter,
            on_exit=markers_on_exit,
        ),
        TourStep(
            key="active_tool",
            title="Active Tool Settings",
            body=(
                "Adjust line thickness and per-tool settings here -- this "
                "panel updates to match whichever annotation tool you have "
                "selected, e.g. picking a SAM model variant."
            ),
            target=lambda w: w.right_panel.active_tool_header(),
            placement="left",
        ),
        TourStep(
            key="classes",
            title="Dataset Setup",
            body=(
                "Define your defect classes and pick a color for each, "
                "e.g. 'crack' or 'scratch'."
            ),
            target=lambda w: w.right_panel.classes_header(),
            placement="left",
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
            key="view_overlays",
            title="View Overlays",
            body=(
                "This tab holds display overlays that guide inspection "
                "without changing your data: Center Crop, Grid, and Anomaly "
                "Constraints. Let's look at each."
            ),
            target=lambda w: w.right_panel.overlays_header(),
            placement="left",
        ),
        TourStep(
            key="center_crop",
            title="Center Crop",
            body=(
                "Overlay a rectangle or circle centered on the image, with "
                "adjustable size, opacity, and border color. Use Calibrate "
                "Center to line it up against a saved reference image -- "
                "Accept saves the match, or Import loads one from a PNG "
                "template."
            ),
            target=lambda w: w.right_panel.center_crop_section(),
            placement="left",
            on_enter=center_crop_on_enter,
            on_exit=center_crop_on_exit,
        ),
        TourStep(
            key="grid",
            title="Grid",
            body=(
                "Calibrate a pixel-to-real-world ratio here -- enter it "
                "directly, or click two known points on the image and "
                "enter their real distance. Once calibrated, Enable Grid "
                "overlays a scaled grid with adjustable spacing, opacity, "
                "and color."
            ),
            target=lambda w: w.right_panel.grid_section(),
            placement="left",
            on_enter=grid_on_enter,
            on_exit=grid_on_exit,
        ),
        TourStep(
            key="anomaly_constraints",
            title="Anomaly Constraints",
            body=(
                "Flag annotations that break your limits: Area Threshold "
                "highlights any annotation larger than a max area, and "
                "Proximity Threshold highlights annotations that sit closer "
                "together than a minimum distance, measured center-to-center "
                "or edge-to-edge. Violation counts and colors update live."
            ),
            target=lambda w: w.right_panel.anomaly_constraints_section(),
            placement="left",
            on_enter=anomaly_on_enter,
            on_exit=anomaly_on_exit,
        ),
        TourStep(
            key="image_adjustments",
            title="Image Adjustments",
            body=(
                "Preview HSV and brightness/contrast adjustments on the "
                "canvas here -- these only affect what you see, not your "
                "source images."
            ),
            target=lambda w: w.right_panel.image_adjustments_header(),
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
