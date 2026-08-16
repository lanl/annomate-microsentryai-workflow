"""Static help/documentation entries searched by the Help menu's search dialog.

To add a new help topic: append a HelpEntry to HELP_ENTRIES below. Nothing
else needs to change — the search dialog and ranking in help_search.py both
read from this list directly. See docs_index.py for the entries generated
from docs/*.md, appended to HELP_ENTRIES at the bottom of this file.
"""

from core.logic.help_entry import HelpEntry


HELP_ENTRIES = [
    HelpEntry(
        title="Creating, opening, and saving a project",
        category="Projects",
        keywords=[
            "new project",
            "open project",
            "save",
            "saving",
            "save as",
            "annoproj",
            "file menu",
            "store",
            "keep",
            "preserve",
            "write",
            "export",
            "backup",
            "save my work",
            "keep my work",
            "bring in an existing project",
            "load a project",
            "import a project",
            "existing project",
            "autosave",
        ],
        description="Start a new project, open an existing .annoproj file, or save your current work.",
        full_text=(
            "Use File > New Project to start fresh, or File > Open Project… to load "
            "an existing .annoproj file. Use File > Save Project (Ctrl+S) to save "
            "changes, and File > Save Project As… (Ctrl+Shift+S) to save under a new "
            "name and folder — this is also how you save a project for the first "
            "time. Once a project has been saved at least once, AnnoMate autosaves "
            "periodically in the background; a status bar message confirms each "
            "autosave."
        ),
    ),
    HelpEntry(
        title="Opening an image folder",
        category="Projects",
        keywords=[
            "open image folder",
            "load images",
            "dataset",
            "folder",
            "import images",
            "bring in images",
            "access a folder",
            "browse for images",
        ],
        description="Load a folder of images as the active dataset to annotate.",
        full_text=(
            "Use File > Open Image Folder… (or the 'Open Image Folder' button on "
            "the start screen) and choose a directory. All supported images in "
            "that folder are loaded into the navigator on the right. If you already "
            "have annotations and just need to point at a different copy of the "
            "same images, use File > Relocate Images… instead — it keeps existing "
            "annotations and only updates the file paths."
        ),
    ),
    HelpEntry(
        title="Relocating images without losing annotations",
        category="Projects",
        keywords=[
            "relocate images",
            "move dataset",
            "orphaned annotations",
            "missing images",
        ],
        description="Point an existing project at a new copy of its images while keeping annotations intact.",
        full_text=(
            "File > Relocate Images… rescans a new folder and matches it against "
            "the images already referenced by the project, keeping annotations "
            "attached wherever a filename matches. If some annotated images can't "
            "be found in the new folder, AnnoMate warns you that those annotations "
            "will be dropped the next time you save."
        ),
    ),
    HelpEntry(
        title="Drawing polygon annotations",
        category="Annotation Tools",
        keywords=[
            "polygon tool",
            "draw annotation",
            "hotkey p",
            "brush thickness",
            "Polygon (P)",
        ],
        description="Manually trace a region by clicking points to build a polygon outline.",
        full_text=(
            "Select the polygon tool (⬠ in the left tool palette, or press 'P') "
            "and click to place points around the region you want to annotate. "
            "New polygons are assigned to the currently active annotation class, "
            "shown in the right panel — pick a class before you start drawing, or "
            "AnnoMate will prompt you to. Use the thickness slider in the tool "
            "palette to adjust the outline width, which can also be changed per "
            "polygon after it's drawn by selecting it first."
        ),
        object_name="toolPolygonButton",
    ),
    HelpEntry(
        title="Adjusting brush thickness",
        category="Annotation Tools",
        keywords=[
            "brush thickness",
            "outline width",
            "polygon thickness",
            "line thickness",
            "active tool",
        ],
        description="Set the default outline width new polygons are drawn with.",
        full_text=(
            "Open the right panel's Active Tool tab and drag the Line Width "
            "slider in the Common section to change the thickness used for new "
            "polygons — this section is always visible, no matter which "
            "drawing tool is selected. Select an existing polygon first if you "
            "want to change its outline width after the fact instead."
        ),
        object_name="activeToolHeader",
    ),
    HelpEntry(
        title="Segmenting with the SAM tool",
        category="Annotation Tools",
        keywords=[
            "sam",
            "segment anything",
            "sam bbox",
            "hotkey s",
            "ai segmentation",
            "auto segment",
            "SAM Segment (S)",
        ],
        description="Draw a bounding box and let the Segment Anything Model (SAM) generate the polygon for you.",
        full_text=(
            "Select the SAM tool (✦ in the tool palette, or press 'S') and drag a "
            "box around the object of interest. AnnoMate runs the Segment Anything "
            "Model on that region and shows a ghost outline with a confidence "
            "score; press Enter to accept it as a polygon on the active class, or "
            "Esc to cancel. Switch model variants in the right panel's Active "
            "Tool tab, under Tool Options — the first use of a variant loads its "
            "weights, which can take a few seconds and is shown there in a "
            "status label."
        ),
        object_name="toolSamButton",
    ),
    HelpEntry(
        title="Choosing a SAM model variant",
        category="Annotation Tools",
        keywords=[
            "sam options",
            "sam variant",
            "sam2",
            "model size",
            "tiny",
            "small",
            "base",
            "large",
            "active tool",
            "tool options",
        ],
        description="Switch between SAM2 model sizes — tiny is fastest, large is most accurate.",
        full_text=(
            "With the SAM tool selected, open the right panel's Active Tool tab "
            "and pick a variant from the dropdown in the Tool Options section "
            "(only shown while a drawing tool is active). The first use of a "
            "variant downloads and loads its weights, which can take a few "
            "seconds; progress is shown in the status label underneath."
        ),
        object_name=None,
    ),
    HelpEntry(
        title="Calibrating pixel-to-real-world distance",
        category="Annotation Tools",
        keywords=[
            "calibrate",
            "calibration",
            "ruler",
            "hotkey c",
            "measure scale",
            "units",
            "grid",
            "view overlays",
        ],
        description="Set a pixel-to-distance scale by clicking two points a known real-world distance apart.",
        full_text=(
            "Click 'Click two points…' in the Grid section of the right "
            "panel's View Overlays tab (or press 'C') and click two points on "
            "the image that correspond to a known real-world distance — for "
            "example, the two ends of a ruler visible in the shot. A dialog "
            "then asks for that real distance and its unit (e.g. '5mm', "
            "'100um', '0.5in'). Alternatively, enter a pixel:real-world ratio "
            "directly, or Import a saved ratio from a .txt file. Once set, the "
            "measure tool, the grid overlay, and Anomaly Constraints all use "
            "this scale."
        ),
        object_name="gridHeader",
    ),
    HelpEntry(
        title="Measuring distances on an image",
        category="Annotation Tools",
        keywords=[
            "measure tool",
            "hotkey m",
            "distance",
            "ruler measurement",
            "Measure Distance (M)",
        ],
        description="Measure real-world distance between two points once the image has been calibrated.",
        full_text=(
            "Select the measure tool (in the left tool palette, or press 'M') "
            "and click two points on the image. The distance is reported in "
            "real-world units using the scale set by calibration — calibrate "
            "the image first if you haven't already, otherwise the measurement "
            "has no unit to convert to."
        ),
        object_name="toolMeasureButton",
    ),
    HelpEntry(
        title="Zooming and resetting the canvas view",
        category="Annotation Tools",
        keywords=[
            "zoom in",
            "zoom out",
            "reset view",
            "fit view",
            "Zoom In",
            "Zoom Out",
            "Reset View",
        ],
        description="Zoom in/out on the canvas or reset it back to fit the image.",
        full_text=(
            "Use the +, -, and ⊙ buttons at the left of the viewport actions bar "
            "(under the canvas) to zoom in, zoom out, or reset the view back to "
            "fit the current image."
        ),
        object_name="viewportActionsBar",
    ),
    HelpEntry(
        title="Setting a center crop template",
        category="Annotation Tools",
        keywords=[
            "center template",
            "center crop",
            "center calibration",
            "auto-center",
            "Center Crop",
            "view overlays",
        ],
        description="Teach AnnoMate where the frame's center point is so future images auto-crop consistently.",
        full_text=(
            "The Center Crop section of the right panel's View Overlays tab "
            "overlays a rectangle or circle centered on the image, with "
            "adjustable size, opacity, and border color. Use Calibrate Center "
            "to mark the true center of a recurring frame or fixture — the "
            "project must be saved first, since the template is stored "
            "alongside the project — then Accept to save the match, or Import "
            "to load one from a saved PNG template. Once saved, AnnoMate "
            "automatically matches this template against every subsequent "
            "image and re-centers the crop overlay to line up, without "
            "needing to recalibrate each image by hand. Use 'Clear Center "
            "Template' to remove it."
        ),
        object_name="centerCropHeader",
    ),
    HelpEntry(
        title="Flagging anomalies with constraint checks",
        category="Annotation Tools",
        keywords=[
            "anomaly constraints",
            "area threshold",
            "proximity threshold",
            "distance threshold",
            "flag defects",
            "qc checks",
            "Anomaly Constraints",
            "view overlays",
        ],
        description="Automatically highlight annotations that exceed an area threshold or sit too close together.",
        full_text=(
            "The Anomaly Constraints section of the right panel's View "
            "Overlays tab flags annotations that break your limits: Area "
            "Threshold highlights any annotation larger than a max area, and "
            "Proximity Threshold highlights annotations that sit closer "
            "together than a minimum distance, measured center-to-center or "
            "edge-to-edge. Violations are outlined directly on the canvas in "
            "the colors chosen here, and violation counts update live as you "
            "annotate. Thresholds and distances use the image's calibrated "
            "real-world units when one has been set, otherwise pixels."
        ),
        object_name="anomalyConstraintsHeader",
    ),
    HelpEntry(
        title="Deleting or editing annotations",
        category="Annotation Tools",
        keywords=[
            "delete annotation",
            "edit polygon",
            "select annotation",
            "remove polygon",
            "dataset navigator",
        ],
        description="Select a polygon on the canvas or in an expanded navigator card to edit, resize, or delete it.",
        full_text=(
            "Click a polygon on the canvas to select it — its thickness "
            "slider updates to match. You can also expand an image's card in "
            "the Dataset Navigator (left panel) to see and select from its "
            "annotation list inline. Drag a selected polygon's points to "
            "reshape it, adjust the thickness slider to change its outline "
            "width, or press Delete to remove the selected annotation "
            "entirely."
        ),
        object_name=None,
    ),
    HelpEntry(
        title="Importing and exporting annotation classes",
        category="Data Menu",
        keywords=[
            "import classes",
            "export classes",
            "data menu",
            "class list",
            "categories",
        ],
        description="Load a set of annotation class names from a text file, or export the current class list.",
        full_text=(
            "Data > Import Annotation Classes… reads class names from a plain text "
            "file (one per line) and adds them to the project's class list, making "
            "them available for annotation. Data > Export Annotation Classes writes "
            "the project's current class list back out, which is useful for "
            "reusing the same class set across projects."
        ),
    ),
    HelpEntry(
        title="Exporting binary masks",
        category="Data Menu",
        keywords=["export masks", "binary masks", "ground truth", "data menu"],
        description="Export each image's annotations as binary mask images for training or evaluation.",
        full_text=(
            "Data > Export Binary Masks… asks for an output folder and writes one "
            "binary mask per annotated image (grouped under a 'ground_truth' "
            "subfolder), suitable as ground-truth input to a training pipeline."
        ),
    ),
    HelpEntry(
        title="Exporting annotations to CSV",
        category="Data Menu",
        keywords=["export csv", "metadata csv", "data menu", "spreadsheet"],
        description="Export annotation metadata for all images to a single CSV file.",
        full_text=(
            "Data > Export CSV… writes a metadata.csv file summarizing the "
            "annotations across every image in the dataset — useful for reviewing "
            "counts, classes, and measurements outside of AnnoMate."
        ),
    ),
    HelpEntry(
        title="Exporting a training dataset structure",
        category="Data Menu",
        keywords=[
            "export train structure",
            "pixel-level train structure",
            "image-level train structure",
            "training data",
            "dataset export",
            "data menu",
        ],
        description="Export images and annotations into a folder layout ready for model training.",
        full_text=(
            "Data > Export Pixel-Level Train Structure… and Data > Export "
            "Image-Level Train Structure… each ask for an output folder and a "
            "dataset name, then write images and their annotations into a "
            "standard training-ready folder structure under that name, matching "
            "the pixel-level (polygon) or image-level (class tag) annotation "
            "style respectively."
        ),
    ),
    HelpEntry(
        title="Exporting COCO JSON annotations",
        category="Data Menu",
        keywords=["export coco", "coco json", "coco format", "data menu"],
        description="Export the current dataset's annotations in the COCO JSON format.",
        full_text=(
            "Data > Export COCO JSON… writes a single .json file containing "
            "every image's polygon annotations in the standard COCO format, "
            "for use with tools and training pipelines that expect COCO input."
        ),
    ),
    HelpEntry(
        title="Exporting a project template",
        category="Data Menu",
        keywords=[
            "export project template",
            "project template",
            "template",
            "reuse settings",
            "data menu",
        ],
        description="Export the current project's settings (classes, calibration, center template) as a reusable template.",
        full_text=(
            "Data > Export Project Template… saves the current project's classes, "
            "calibration, and center-template settings (without its images or "
            "annotations) so they can be reused as the starting point for a new "
            "project. The project must be saved first; if it hasn't been, "
            "AnnoMate prompts you to save it before exporting the template."
        ),
    ),
    HelpEntry(
        title="Loading an AI model for MicroSentryAI",
        category="MicroSentryAI",
        keywords=[
            "load model",
            "load previous model",
            "inference model",
            "pt",
            "pth",
            "microsentry panel",
            "Load New",
            "Load Previous",
        ],
        description="Load a PyTorch model to enable AI-assisted scoring, heatmaps, and segmentation.",
        full_text=(
            "In the right panel's Microsentry section, use 'Load New' to browse "
            "for a .pt or .pth model file, or 'Load Previous' to reload the model "
            "path saved with the current project. Once loaded, AnnoMate runs "
            "inference across the dataset in the background and shows progress in "
            "the status bar; results power the heatmap overlay and AI-suggested "
            "segmentation polygons."
        ),
        object_name="microsentryLoadNewButton",
    ),
    HelpEntry(
        title="Understanding the heatmap overlay and AI segmentation",
        category="MicroSentryAI",
        keywords=[
            "heatmap",
            "score map",
            "seg_pct",
            "ai polygons",
            "accept ai polygon",
            "microsentry",
            "Accept AI Polygons",
            "Segmentation",
        ],
        description="View model confidence as a heatmap and accept AI-suggested polygons onto your annotation classes.",
        full_text=(
            "Once a model is loaded, the Microsentry panel's Heatmap and "
            "Segmentation toggles let you overlay the model's score map (a "
            "semi-transparent heatmap) and candidate polygons traced from that "
            "score map using a threshold and simplification setting. Click a "
            "suggested AI polygon to accept it into a chosen class with the "
            "small popup that appears, or use 'Accept AI Polygons' in the right "
            "panel to accept every current suggestion onto the active class at "
            "once."
        ),
        object_name="microsentryHeatmapButton",
    ),
    HelpEntry(
        title="Accepting or rejecting an image review",
        category="Review",
        keywords=[
            "accept reject",
            "review bar",
            "review decision",
            "qc",
            "Accept",
            "Reject",
        ],
        description="Mark each image as accepted or rejected using the floating review bar over the canvas.",
        full_text=(
            "The floating Accept/Reject bar in the top-right of the canvas lets "
            "you record a review decision per image — click Accept or Reject to "
            "set it, or click the active button again to clear the decision. Drag "
            "the ⋮ handle to reposition the bar if it's covering something you "
            "need to see. Decisions are saved with the project. The bar only "
            "appears once an image is loaded."
        ),
        object_name="reviewAcceptButton",
    ),
    HelpEntry(
        title="Browsing the dataset navigator panel",
        category="Right Panel",
        keywords=[
            "dataset navigator",
            "navigator panel",
            "prev next",
            "image list",
            "filter",
            "sort",
            "Previous image",
            "Next image",
            "left panel",
        ],
        description="Step through images with Prev/Next, or jump to one directly from the navigator list.",
        full_text=(
            "The Dataset Navigator on the left lists every image in the loaded "
            "folder. Use the Prev (A) / Next (D) buttons or click a card to "
            "jump to that image, and click a card to expand it and see its "
            "annotations, inspector, and notes inline. The three status chips "
            "(Undecided / Reviewed / Incomplete) show live counts and can be "
            "clicked to filter the list; use the Filter button for more "
            "options, including filtering by class and choosing the sort "
            "order. Collapse the panel (the arrow at its top) to save canvas "
            "space — the counter and status counts stay visible on the "
            "collapsed icon rail."
        ),
        object_name="leftPanelNavigator",
    ),
    HelpEntry(
        title="Managing annotation classes in the panel",
        category="Right Panel",
        keywords=[
            "annotation classes panel",
            "add class",
            "class list",
            "active class",
            "Add Class",
            "dataset setup",
        ],
        description="Add new annotation classes and pick the active class for new polygons.",
        full_text=(
            "The Annotation Classes section, under the right panel's Dataset "
            "Setup tab, lists every class in the project. Type a name and "
            "click 'Add Class' to create a new one, or click a class in the "
            "list to make it the active class that new polygons (drawn or "
            "SAM-segmented) are assigned to."
        ),
        object_name="classesHeader",
    ),
    HelpEntry(
        title="Switching between pixel-level and image-level annotation modes",
        category="Right Panel",
        keywords=[
            "image level mode",
            "pixel level mode",
            "annotation mode",
            "class tag",
            "tag image",
            "mode toggle",
            "Pixel Level",
            "Image Level",
        ],
        description="Choose whether a rejected image needs polygon annotations or whole-image class tags to count as reviewed.",
        full_text=(
            "The Pixel Level / Image Level toggle above the class list sets how "
            "review completeness is judged for rejected images. In Pixel Level "
            "mode (the default), a rejected image needs at least one polygon "
            "annotation; the polygon and SAM drawing tools are enabled. In "
            "Image Level mode, a rejected image instead needs at least one "
            "class tag — click a class's tag cell in the table to toggle it on "
            "or off for the current image — and the drawing tools are disabled "
            "so nothing gets accidentally drawn. A class with pixel-level "
            "annotations already on it can't be deleted from Image Level mode "
            "without switching back to Pixel Level first."
        ),
        object_name="classesModeToggle",
    ),
    HelpEntry(
        title="Setting an inspector name and image notes",
        category="Right Panel",
        keywords=[
            "inspector",
            "image note",
            "set inspector",
            "session inspector",
            "Set Inspector",
            "dataset navigator",
        ],
        description="Record who inspected each image and add a free-text note, expanded inline in the navigator.",
        full_text=(
            "Expand an image's card in the Dataset Navigator (left panel) to "
            "reveal its Inspector and Image note fields. Click 'Set' next to "
            "the inspector field to save the current name as the session "
            "inspector, which then pre-fills for any image that doesn't "
            "already have one saved."
        ),
        object_name=None,
    ),
    HelpEntry(
        title="Bulk-assigning an inspector with Set All",
        category="Right Panel",
        keywords=[
            "set all",
            "bulk inspector",
            "assign inspector",
            "all images",
            "Set All",
        ],
        description="Assign one inspector name across many images at once, optionally filtered to reviewed or in-review images.",
        full_text=(
            "Click 'Set All' next to the inspector field to open a dialog that "
            "assigns one inspector name to a whole set of images in one step. "
            "Choose which images to apply it to — All Images, Reviewed only, or "
            "In Review only — and the list updates to show exactly which images "
            "will change and their current inspector, before you confirm."
        ),
        object_name="metadataSetAllButton",
    ),
    HelpEntry(
        title="Keyboard shortcuts",
        category="Reference",
        keywords=["hotkeys", "shortcuts", "keyboard"],
        description="Quick reference for the annotation hotkeys and file shortcuts.",
        full_text=(
            "Annotation hotkeys: A = previous image, D = next image, "
            "P = polygon tool, S = SAM segment tool, C = toggle Grid "
            "calibration, M = measure tool, Delete = delete the selected "
            "annotation. (A/D are disabled while center-crop calibration is "
            "in progress.) File shortcuts: Ctrl+N = New Project, Ctrl+O = "
            "Open Project, Ctrl+S = Save Project, Ctrl+Shift+S = Save Project "
            "As, Ctrl+Q = Exit."
        ),
    ),
    HelpEntry(
        title="Taking the guided welcome tour",
        category="Reference",
        keywords=[
            "welcome tour",
            "guided tour",
            "onboarding",
            "walkthrough",
            "Show Welcome Tour",
        ],
        description="Replay the guided walkthrough of AnnoMate's main panels and tools.",
        full_text=(
            "AnnoMate shows a short guided tour highlighting the main tools and "
            "panels the first time it's used. Choose Help > Show Welcome Tour "
            "to replay it at any time — for example, after onboarding a new "
            "inspector."
        ),
    ),
    HelpEntry(
        title="Tracking session time on a project",
        category="Reference",
        keywords=[
            "session time",
            "time tracking",
            "elapsed time",
            "status bar",
        ],
        description="See how long the current project has been open in this session, shown in the status bar.",
        full_text=(
            "Once a project is open, the status bar shows the elapsed time for "
            "the current session, updating roughly once a minute. It resets "
            "when a project is closed and starts again the next time one is "
            "opened."
        ),
    ),
    HelpEntry(
        title="Searching the Help documentation",
        category="Reference",
        keywords=[
            "help search",
            "documentation",
            "docs",
            "guide",
            "manual",
            "find a topic",
            "search help",
            "user manual",
            "how do i",
        ],
        description="Search the full user manual and quick UI help from the Help menu — no need to open the docs folder separately.",
        full_text=(
            "There are two ways to search: type directly into the search box at "
            "the top of the Help menu — matching File/Data commands are listed "
            "right there so you can jump straight to one, or (if nothing "
            "matches) a 'Search Help for …' entry opens the full search window. "
            "Or open Help > Search Help… (or press F1) directly. "
            "Either way, type a keyword, a partial word, or a natural-language "
            "question — the search tolerates typos and related terms, not just "
            "exact matches, and covers both quick per-control help and the "
            "complete written user manual (Getting Started, AnnoMate, "
            "MicroSentryAI, and Validation guides), so you rarely need to open "
            "those files directly. Click a result to read its full instructions "
            "in the panel on the right; if it names a specific button or panel, "
            "a 'Show Me in the App' button appears to highlight it live."
        ),
    ),
]

# Adds one HelpEntry per section/subsection of docs/*.md (the user manual),
# so a search can also answer a question or point to a spot in the written
# guides, not just the quick UI-focused entries above. See docs_index.py.
from core.logic.docs_index import load_doc_entries as _load_doc_entries  # noqa: E402

HELP_ENTRIES = HELP_ENTRIES + _load_doc_entries()
