# Developer Guide — AnnoMate & MicroSentryAI

This guide is written for someone who has completed a college-level Python course and wants to understand, maintain, or extend this project. It covers how the code is organized, how all the pieces connect, and what rules to follow when making changes.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Setting Up Your Development Environment](#2-setting-up-your-development-environment)
3. [Project Directory Structure](#3-project-directory-structure)
4. [The Big Idea: MVC Architecture](#4-the-big-idea-mvc-architecture)
5. [Layer 1 — Core (Pure Python)](#5-layer-1--core-pure-python)
6. [Layer 2 — Models (Qt Adapters)](#6-layer-2--models-qt-adapters)
7. [Layer 3 — Controllers (Business Logic)](#7-layer-3--controllers-business-logic)
8. [Layer 4 — Views (The GUI)](#8-layer-4--views-the-gui)
9. [How the GUI Is Assembled](#9-how-the-gui-is-assembled)
10. [Signals and Slots: How Parts Communicate](#10-signals-and-slots-how-parts-communicate)
11. [Key Data Flows: Tracing an Action End-to-End](#11-key-data-flows-tracing-an-action-end-to-end)
12. [The AI Features: SAM2 and MicroSentryAI](#12-the-ai-features-sam2-and-microsentry-ai)
13. [Project Files: Saving and Loading](#13-project-files-saving-and-loading)
14. [Running the Tests](#14-running-the-tests)
15. [How to Add a New Feature](#15-how-to-add-a-new-feature)
16. [Common Pitfalls and Rules to Remember](#16-common-pitfalls-and-rules-to-remember)

---

## 1. What This Project Does

AnnoMate & MicroSentryAI is a desktop application for annotating images with polygon shapes and running AI-assisted anomaly detection. Here is what a user actually does with it:

1. **Load a folder of images** (e.g., photos of industrial parts on a tray).
2. **Define annotation classes** (e.g., "crack", "scratch", "corrosion") and pick a color for each.
3. **Draw polygon annotations** on images manually, or use SAM2 (Segment Anything Model 2) to auto-segment an object from a drawn bounding box.
4. **Load a MicroSentryAI model** (a `.pt` file trained with Anomalib) to get a heatmap of anomalous regions overlaid on each image.
5. **Accept AI-suggested polygons** into the annotation set.
6. **Mark images as Accept or Reject** during review.
7. **Export** annotations as COCO JSON, polygon overlay images, binary mask PNGs, or a CSV summary.
8. **Save and reopen projects** (`.annoproj` files) that preserve all annotations and settings.
9. **Validate model performance** by comparing predicted masks against ground truth masks in a separate validation window.

---

## 2. Setting Up Your Development Environment

### Prerequisites

- **Python 3.10** (strongly recommended — ML libraries pin to this)
- **Anaconda or Miniconda** for environment management

### Installation

First, clone the repository:

```bash
git clone <repository-url>
cd annomate-microsentryai-workflow
```

Then create and activate the environment for your hardware. Pick **one** of the following — each command creates a fully configured environment in a single step:

**macOS (Apple Silicon — creates `annomate-mac`):**
```bash
conda env create --file environment-mac.yml
conda activate annomate-mac
```

**Windows/Linux with NVIDIA GPU (creates `annomate-cuda`):**
```bash
conda env create --file environment-cuda.yml
conda activate annomate-cuda
```

**Windows/Linux without a GPU, or CPU-only (creates `annomate-cpu`):**
```bash
conda env create --file environment-cpu.yml
conda activate annomate-cpu
```

Each yml file specifies its own environment name, Python version (3.10), and all required pip packages including PyTorch, Anomalib, SAM2, and PySide6. You do **not** need to create a base environment manually first — `conda env create` handles everything.

### Running the Application

Make sure your environment is activated, then run from the project root:

```bash
python src/main.py
```

### Pre-commit Hooks (if contributing)

```bash
pre-commit install
```

This sets up automatic linting checks before each `git commit`.

---

## 3. Project Directory Structure

```
annomate-microsentryai-workflow/
├── docs/                       # Documentation files
├── logos/                      # Application logo assets
├── sam_weights/                # Auto-created on first SAM use (weights downloaded here)
├── environment*.yml            # Conda dependency files (cpu / mac / cuda)
├── requirements.txt
└── src/                        # All Python source code lives here
    ├── main.py                 # Entry point — the only place all layers are wired together
    │
    ├── core/                   # Pure Python, ZERO Qt imports
    │   ├── logic/
    │   │   ├── mask_comparator.py      # Computes IoU and precision/recall between masks
    │   │   └── comparison_logger.py    # Logs mask comparison results
    │   ├── persistence/
    │   │   └── project_io.py           # Reads/writes .annoproj and COCO JSON files
    │   ├── states/
    │   │   ├── dataset_state.py        # Raw data: images, annotations, classes, metadata
    │   │   ├── inference_state.py      # Per-image AI score maps
    │   │   └── validation_state.py     # Paths and results for the validation workflow
    │   └── utils/
    │       ├── constants.py            # Default colors, app name, version
    │       ├── geometry.py             # Polygon math (area, bbox, simplify)
    │       └── logger.py               # Logging setup
    │
    ├── models/                 # Qt adapter classes — wrap core states with Qt signals
    │   ├── dataset_model.py            # DatasetTableModel (wraps DatasetState)
    │   ├── inference_model.py          # InferenceModel (wraps InferenceState)
    │   └── validation_model.py         # ValidationModel (wraps ValidationState)
    │
    ├── controllers/            # Business logic — headless QObject services
    │   ├── io_controller.py            # Load folders, export data, import JSON
    │   ├── inference_controller.py     # Load AI models, run batch inference
    │   ├── sam_controller.py           # Load SAM2 model, run segmentation
    │   ├── validation_controller.py    # Run mask comparison evaluation
    │   ├── project_controller.py       # Save/open/autosave projects
    │   └── autosave.py                 # Timer that triggers periodic saves
    │
    ├── ai_strategies/          # Qt-free AI backend wrappers
    │   ├── interface.py                # Abstract base class for strategies
    │   ├── anomalib_strategy.py        # Wraps Anomalib for inference scoring
    │   └── sam_strategy.py             # Wraps Meta SAM2 for bbox segmentation
    │
    ├── views/                  # All Qt widgets — the visible application
    │   ├── app_window.py               # Top-level QMainWindow shell + menu bar
    │   ├── annomate/
    │   │   ├── window.py               # Main annotation workspace coordinator
    │   │   ├── image_label.py          # The drawing canvas (zoom, pan, polygons)
    │   │   ├── right_panel.py          # Scrollable right panel shell
    │   │   ├── tool_palette.py         # Left tool column (polygon, SAM, thickness)
    │   │   ├── status_bar.py           # Bottom status strip
    │   │   ├── _splitter.py            # Custom styled splitter widget
    │   │   ├── microsentry_overlay.py  # Heatmap rendering helper
    │   │   └── sections/
    │   │       ├── _collapsible.py     # Reusable collapsible section header
    │   │       ├── _shared.py          # Shared helpers (dot widget, colors)
    │   │       ├── navigator.py        # Image list with status dots
    │   │       ├── classes.py          # Annotation class list with add/delete
    │   │       ├── annotations.py      # Per-image annotation list
    │   │       ├── metadata.py         # Inspector name and image note fields
    │   │       └── microsentry.py      # MicroSentryAI controls and sliders
    │   └── validation/
    │       └── window.py               # Standalone validation comparison window
    │
    ├── MVC.md                  # Architecture rules document (authoritative)
    └── tests/                  # Test suite
        ├── unit/
        ├── integration/
        └── model/
```

---

## 4. The Big Idea: MVC Architecture

### What is MVC?

MVC stands for **Model-View-Controller**. It is a way of organizing code so that three concerns are kept separate:

- **Model** — where the data lives
- **View** — what the user sees and clicks
- **Controller** — the logic that connects them and handles operations

This project adds a **Core** layer underneath the model, giving it four layers in total.

### Why does this matter?

Without this separation, it is very common to end up with one giant "God file" that reads files, draws buttons, computes math, and opens dialogs all in the same function. That is nearly impossible to test or change safely.

With MVC:
- You can test all the annotation logic without ever opening a window.
- You can change how something looks without touching the business logic.
- You can swap out the AI backend without changing the GUI.

### The Four Layers at a Glance

```
┌─────────────────────────────────────────────┐
│  VIEWS  — Qt Widgets. What the user sees.   │
│  Reads from models. Calls controllers.      │
└─────────────────────────────┬───────────────┘
                              │ calls / reads
┌─────────────────────────────▼───────────────┐
│  CONTROLLERS — Headless QObject services.   │
│  Business logic, file I/O, AI operations.  │
│  Emits signals back to views when done.     │
└─────────────────────────────┬───────────────┘
                              │ reads/mutates
┌─────────────────────────────▼───────────────┐
│  MODELS — Qt adapter classes.               │
│  Wrap core states. Emit Qt signals on       │
│  every change so views can react.           │
└─────────────────────────────┬───────────────┘
                              │ wraps
┌─────────────────────────────▼───────────────┐
│  CORE — Pure Python. Zero Qt imports.       │
│  States (data containers), persistence,     │
│  math utilities. Fully unit-testable.       │
└─────────────────────────────────────────────┘
```

The **most important rule** is that data only flows downward through the layers when writing, and upward through signals when notifying. No layer ever skips a layer — the view does not touch the core state directly.

---

## 5. Layer 1 — Core (Pure Python)

The `core/` folder contains everything that has nothing to do with Qt or the GUI. You can import any file in this folder from a plain Python script and it will work.

### States (`core/states/`)

A **state** is just a plain Python class that holds data in dictionaries and lists. Think of it like a structured notebook. There is no logic for drawing, signaling, or file I/O — just data and simple CRUD operations.

#### `DatasetState` — the most important state

This holds everything about the current image set and its annotations:

```python
# What it stores:
self.image_dir = ""           # path to the loaded folder
self.image_files = []         # list of filenames like ["img1.jpg", "img2.jpg"]
self.annotations = {}         # { "img1.jpg": [ { "category_name": "crack", "polygon": [(x,y), ...] } ] }
self.inspectors = {}          # { "img1.jpg": "Alice" }
self.notes = {}               # { "img1.jpg": "Needs second look" }
self.review_decisions = {}    # { "img1.jpg": "accept" }
self.class_names = []         # ["crack", "scratch"]
self.class_colors = {}        # { "crack": (255, 0, 0), "scratch": (0, 200, 0) }
```

**Key design choice:** class names and colors are **not** cleared when a new folder is loaded. Only per-folder data (annotations, notes, etc.) is cleared. This means your classes survive across multiple folder loads in the same session.

The state also provides simple methods for CRUD (Create, Read, Update, Delete) operations:

```python
state.add_annotation("img1.jpg", "crack", [(10, 20), (30, 40), (50, 20)])
state.delete_annotation("img1.jpg", 0)  # removes annotation at index 0
state.add_class("crack", (255, 0, 0))
```

#### `InferenceState`

Stores the anomaly score maps returned by the MicroSentryAI model. A score map is a 2D NumPy array (same size as the image) where higher values mean more anomalous. These are stored as a dictionary keyed by image path:

```python
self.score_maps = {}  # { "/path/to/img.jpg": np.ndarray of shape (H, W) }
```

#### `ValidationState`

Stores paths configured for the validation workflow (ground truth folder, predictions folder) and the results of the last evaluation run.

### Persistence (`core/persistence/`)

`project_io.py` handles reading and writing `.annoproj` project files. An `.annoproj` file is actually a folder that contains:

- `project.annoproj` — a JSON file with metadata and settings
- `annotations.coco.json` — all polygon annotations in COCO format
- `scoremaps.npz` — compressed NumPy arrays of the AI score maps
- `autosave/` — a copy of the above, written every 5 minutes

`ProjectIO` is a plain Python class (no Qt) with two key methods:
- `save_project(...)` — writes all three files
- `load_project(path)` — reads the `.annoproj` JSON and returns a dict
- `apply_project_to_states(...)` — takes that dict and populates the state objects

These are separated so the `ProjectController` can inspect the data (e.g., check for warnings) before applying it.

### Utilities (`core/utils/`)

- `constants.py` — `DEFAULT_CLASS_COLORS` (a list of 7 RGB tuples cycled when adding new classes), app name, version.
- `geometry.py` — `polygon_area()`, `polygon_bbox()`, `simplify_polygon()`. All operate on plain Python lists of `(x, y)` tuples.
- `logger.py` — calls `logging.basicConfig(...)` to set up Python's standard logging module. All other files use `logging.getLogger(__name__)`.

---

## 6. Layer 2 — Models (Qt Adapters)

Qt has a concept called **Model/View** built into the framework. A `QAbstractTableModel` is a class that knows how to feed data to Qt list/table widgets. However, it also has a signal called `dataChanged` that it emits whenever data is modified — this is how views know to repaint themselves.

The models in this project each wrap one core state object and do two things:

1. Implement the Qt model interface (`rowCount`, `columnCount`, `data`, `headerData`)
2. Expose a **command API** — methods that mutate the state and then emit `dataChanged` or `modelReset`

### `DatasetTableModel`

This is the main model. It wraps `DatasetState`. Every time anything changes (annotation added, class deleted, note written), the model emits `dataChanged` so all connected views can update.

The key pattern is in every mutation method:

```python
def add_annotation(self, row, category, polygon, thickness=2.0):
    # 1. Validate
    if not (0 <= row < self.rowCount()):
        return
    # 2. Mutate the state
    self.state.add_annotation(filename, category, polygon, thickness)
    # 3. Tell Qt "something changed on this row"
    self._emit_row(row)
```

`_emit_row(row)` is a helper that calls `self.dataChanged.emit(...)` for the entire row. Any widget that has connected to `dataChanged` will then update itself.

**The separation rule:** Views call model methods. Models talk to states. Views never access `model.state` directly. The model provides a typed query API:

```python
# Views use these:
model.get_annotations(row)       # returns a list of dicts
model.get_class_names()          # returns a list of strings
model.get_image_path(row)        # returns a string
model.is_reviewed(row)           # returns a bool
```

---

## 7. Layer 3 — Controllers (Business Logic)

Controllers are where operations actually happen. They are `QObject` subclasses (which lets them use Qt signals) but they have **zero widget imports**. No `QLabel`, no `QDialog`, no `QColor` — nothing that needs a screen.

### `IOController`

Handles all file I/O for the dataset:

| Method | What it does |
|---|---|
| `load_folder(directory)` | Scans for image files (png/jpg/bmp/tif), calls `model.load_folder()` |
| `load_image_for_display(row)` | Reads image from disk using OpenCV, returns a BGR NumPy array |
| `export_polygons_and_data(out_dir)` | Saves overlay JPEG images + a JSON data file |
| `export_binary_masks(out_dir)` | Saves black-and-white PNG masks (white = annotated region) |
| `export_csv(out_path)` | Saves a CSV with one row per image: tray, filename, inspector, note, classes |
| `import_data_json(path)` | Loads annotations from a custom AnnoMate JSON or a COCO JSON file |

**Important:** `IOController` does not open dialogs. When something goes wrong, it raises a Python exception. The view layer catches the exception and shows the `QMessageBox` to the user. This is by design.

### `InferenceController`

Manages the MicroSentryAI anomaly detection pipeline:

**Model loading:**
```python
controller.load_model("/path/to/model.pt")  # returns the model name string
controller.has_model()                       # returns True/False
```

**Batch inference** runs on a background thread (`InferenceWorker`) so the UI does not freeze:
```python
controller.start_batch_inference(list_of_image_paths)
# The controller then emits signals as results come in:
# controller.result_ready(path, score_map)
# controller.progress(count_done)
# controller.batch_done()
```

**Visualization helpers** — pure Python functions that run on the main thread. These take a score map and produce heatmap overlays or polygon contours:
- `compute_heatmap(...)` — applies Gaussian blur, normalizes, applies the jet colormap, alpha-composites onto the image
- `compute_segmentation(...)` — thresholds the score map at a percentile, finds contours with OpenCV, simplifies them with Douglas-Peucker

### `SAMController`

Manages the SAM2 (Segment Anything Model 2) pipeline. SAM is a large neural network — loading it can take 30+ seconds — so it uses **two background threads**:

- `_ModelLoadWorker` — downloads weights (first run only) and initializes the model
- `SAMWorker` — runs a single bounding-box prediction

```python
# On app startup, SAMController checks if weights are already on disk:
controller.try_autoload("sam2_t.pt")   # returns True if started

# When the user activates the SAM tool:
controller.ensure_loaded_async()        # starts background load if not already loaded
# → emits loading_done or loading_failed when complete

# When the user draws a bounding box on the canvas:
controller.run_inference(bgr_image, (x1, y1, x2, y2))
# → emits result_ready(polygon_points, confidence) when done
```

### `ProjectController`

Owns the save/open/autosave lifecycle. It connects to `dataset_model.dataChanged` and `dataset_model.modelReset` to track whether there are unsaved changes (the "dirty flag").

Key properties:
- `is_dirty` — True if there are unsaved changes
- `has_project` — True if a project directory has been set
- `project_name` — the current project's name

Key methods:
- `new_project()` — clears all state
- `open_project(path)` — loads a `.annoproj` file, returns `(project_data, warnings)`
- `save_project()` — writes to the current project directory
- `save_project_as(dir, name)` — saves to a new location

The **autosave** system (`AutosaveManager`) runs a `QTimer` that fires every 5 minutes. When it fires, `ProjectController._do_autosave()` writes a snapshot into a `autosave/` subfolder. Score maps are skipped in autosaves to keep them fast.

---

## 8. Layer 4 — Views (The GUI)

The views are all `QWidget` subclasses. They display data, accept user input, and express user intent through signals. **They never store the real application data** — they always read from the model.

### Two Windows

The application has two separate windows:

| Window | Class | Purpose |
|---|---|---|
| Main window | `AppWindow` → `AnnoMateWindow` | Image annotation workspace |
| Validation window | `ValidationWindow` | Mask comparison / model evaluation |

`AppWindow` contains `AnnoMateWindow` as its central widget. `ValidationWindow` is a separate floating window that opens when the user chooses "Open Validation…" from the menu.

### `AppWindow` — The Shell

`AppWindow` is a `QMainWindow`. Its responsibilities are narrow and specific:

1. Hold the menu bar (File, Data, Validation, Microsentry menus)
2. Own all `QFileDialog` and `QMessageBox` calls — every dialog in the application lives here
3. Route menu actions to the correct controller methods
4. Show status bar messages for project save/autosave events
5. Guard against closing with unsaved changes (`closeEvent`)

**The golden rule for `AppWindow`:** It is the only place that should ever call `QFileDialog.getOpenFileName(...)` or `QMessageBox.critical(...)`. If you find yourself wanting to open a dialog from a child widget, bubble a signal up to `AppWindow` instead.

---

## 9. How the GUI Is Assembled

### AnnoMateWindow Layout

`AnnoMateWindow._init_ui()` builds the following structure entirely in code:

```
AnnoMateWindow (QWidget)
└── QVBoxLayout
    ├── workspace (QWidget) — stretch=1, fills all available space
    │   └── QHBoxLayout
    │       ├── ToolPalette (QFrame, 56px wide, fixed)
    │       └── StyledSplitter (Qt.Horizontal, resizable)
    │           ├── ImageLabel (canvas, expands to fill)
    │           │   ├── _ZoomToolbar (floating overlay, bottom-left)
    │           │   ├── _ReviewBar (floating overlay, top-right)
    │           │   └── _AIAcceptPopup (floating overlay, appears near AI polygon)
    │           └── RightPanel (min 160px wide)
    └── AnnoMateStatusBar (QWidget, fixed 26px height)
```

The "floating overlays" are `QFrame` widgets that are parented to the `ImageLabel` canvas widget. This means they sit on top of the canvas visually. Their positions are recalculated every time the canvas is resized, which is detected by `AnnoMateWindow.eventFilter(...)`.

### RightPanel Layout

The right panel has a vertical `StyledSplitter` inside it, and the bottom half is wrapped in a `QScrollArea`:

```
RightPanel (QWidget)
└── QVBoxLayout
    ├── _CollapsibleSection("Microsentry")  ← hidden until user enables it
    └── StyledSplitter (Qt.Vertical, resizable)
        ├── _CollapsibleSection("Dataset Navigator")  ← contains DataNavigatorSection
        └── QScrollArea
            └── bottom_content (QWidget)
                ├── _CollapsibleSection("Annotation Classes")  ← ClassesSection
                ├── _CollapsibleSection("Current Image Annotations")  ← AnnotationsSection
                └── _CollapsibleSection("Inspector/Notes")  ← MetadataSection
```

### The `_CollapsibleSection` Pattern

Almost every section in the right panel is wrapped in a `_CollapsibleSection`. This is a reusable widget pattern that provides a **bold toggle button as a header** and a **collapsible body**:

```
▾  Dataset Navigator         ← clicking this toggles the body
─────────────────────────
[body content here]
```

When the button is clicked, `_on_toggle(checked)` runs:
- Shows or hides the `_body` widget
- Changes the arrow from ▾ (expanded) to ▸ (collapsed)
- Emits the `toggled(bool)` signal so the parent can react (e.g., resize the splitter)

### The Canvas: `ImageLabel`

`ImageLabel` is a `QLabel` subclass — the most complex widget in the project. It handles:

**Zoom and pan:**
- Mouse wheel scrolling zooms in/out around the cursor position
- Middle-mouse button drag pans the image
- The `_zoom_toolbar` (the +/−/⊙ buttons) calls `zoom_in()`, `zoom_out()`, `reset_view()`

**Polygon drawing:**
- Left-click adds a vertex to the in-progress polygon
- Double-click closes the polygon and emits `polygonFinished(pts)`
- Backspace removes the last vertex
- Escape cancels the polygon and emits `toolCanceled()`

**Editing:**
- Clicking near an existing polygon vertex grabs it for dragging
- Clicking inside an existing polygon body drags the whole polygon
- When the drag is released, `polygonEdited(idx, pts)` is emitted

**SAM bounding box mode:**
- Left-click-drag draws a bounding box rubber band
- On release, `samBboxDrawn(x1, y1, x2, y2)` is emitted in original image coordinates

All coordinates emitted by signals are in the **original image coordinate space**, not the scaled/displayed space. The canvas internally converts between display and image coordinates using `_zoom` (a float) and `_offset` (a QPointF describing pan).

### The Tool Palette

`ToolPalette` is a narrow (56px) vertical column on the left side with two primary tools:

- **⬠ Polygon** — toggleable, activates polygon drawing on the canvas
- **✦ SAM Segment** — toggleable, activates the SAM bounding box tool

Both tools use a `QButtonGroup` set to exclusive mode, meaning only one can be active at a time. Clicking an already-active tool deactivates it (toggling off).

There are also two popup menu buttons (not checkable):
- **◢ Brush Thickness** — opens a popup with a slider for line thickness
- **⚙ SAM Options** — opens a popup with a combo box for selecting the SAM model size

### The Status Bar

`AnnoMateStatusBar` is a 26px-tall strip at the bottom that shows:

- **Zoom level** — e.g., "Zoom: 150%"
- **Image dimensions** — e.g., "1920 × 1080 px"
- **Active tool** — e.g., "Tool: Polygon" with a hint like "double-click to close · Esc to cancel"
- **Active class** — e.g., "Class: crack"
- **Inference progress** — a progress bar showing MicroSentryAI batch completion

---

## 10. Signals and Slots: How Parts Communicate

Qt signals and slots are the primary way different parts of the application communicate without being directly coupled.

### What is a Signal?

A signal is like a notification you can emit (broadcast). Any number of functions can "subscribe" (connect) to a signal, and they will be called when the signal fires.

```python
# Defining a signal on a class:
class MyWidget(QWidget):
    something_happened = Signal(str)  # this signal carries a string

# Emitting it:
self.something_happened.emit("hello")

# Connecting to it elsewhere:
my_widget.something_happened.connect(self.on_something)

def on_something(self, message: str):
    print(message)  # called whenever the signal fires
```

### The Signal Map for AnnoMateWindow

Here is every major signal connection in the main workspace, showing what triggers what:

| Event | Signal | Connected to |
|---|---|---|
| User clicks image in navigator | `right_panel.image_selected(int)` | `AnnoMateWindow._load_row` |
| User draws polygon | `canvas.polygonFinished(list)` | `AnnoMateWindow._on_polygon_finished` |
| Polygon stored in model | `dataset_model.dataChanged` | `AnnoMateWindow._on_dataset_data_changed` |
| User clicks class row | `right_panel.class_selected(str)` | `AnnoMateWindow._set_active_class` |
| User clicks annotation row | `right_panel.annotation_selected(int)` | `AnnoMateWindow._on_annotation_selected` |
| User changes zoom | `canvas.zoom_changed(float)` | `status_bar.set_zoom` |
| User selects polygon tool | `tool_palette.tool_selected(str)` | `AnnoMateWindow._on_tool_selected` |
| User draws SAM bbox | `canvas.samBboxDrawn(...)` | `AnnoMateWindow._on_sam_bbox_drawn` |
| SAM inference done | `sam_controller.result_ready(list, float)` | `AnnoMateWindow._on_sam_result_ready` |
| Inference batch completes | `inference_controller.batch_done()` | `AnnoMateWindow._on_inference_batch_done` |
| User clicks Accept/Reject | `review_bar.decision_changed(str)` | `AnnoMateWindow._on_review_decision` |

### Debouncing

Some signals fire very rapidly — for example, a slider value changes on every pixel the user moves it. Running an expensive operation (like re-rendering the heatmap) on every single slider event would make the UI stutter.

The solution is a **debounce timer**. In `MicrosentrySection`, every slider connects to `self._debounce.start()` instead of directly emitting `settings_changed`. The debounce is a `QTimer` set to fire once after 200ms. If the slider moves again within those 200ms, the timer resets. Only when the user stops moving the slider for 200ms does the `settings_changed` signal actually fire.

```python
# The debounce timer setup:
self._debounce = QTimer(self)
self._debounce.setSingleShot(True)  # fires once, not repeatedly
self._debounce.setInterval(200)     # 200 milliseconds
self._debounce.timeout.connect(self.settings_changed)

# Sliders connect to start the timer, not directly to settings_changed:
self._alpha.valueChanged.connect(lambda v: self._debounce.start())
```

---

## 11. Key Data Flows: Tracing an Action End-to-End

Understanding the full path an action takes helps you know where to make a change.

### Flow 1: User Draws a Polygon

```
1. User clicks on the canvas multiple times, then double-clicks to close.

2. ImageLabel detects the double-click and emits:
   canvas.polygonFinished(pts)  — pts is a list of (x, y) in image coords

3. AnnoMateWindow._on_polygon_finished(pts) is called:
   - Looks up the active class name (_active_class)
   - Calls: dataset_model.add_annotation(current_row, class_name, pts, thickness)

4. DatasetTableModel.add_annotation() is called:
   - Validates the row is in range
   - Calls: self.state.add_annotation(filename, category, polygon, thickness)
   - Calls: self._emit_row(row)  which fires dataChanged

5. dataChanged fires. All connected slots are called:
   - AnnoMateWindow._on_dataset_data_changed() → calls _refresh_canvas_render()
   - DataNavigatorSection._on_data_changed() → updates annotation count in the row widget
   - ClassesSection._on_data_changed() → updates annotation counts on each class row

6. _refresh_canvas_render() reads the current annotations back from the model
   and calls canvas.set_overlays(...) to repaint the polygons on screen.
```

### Flow 2: User Opens an Image Folder

```
1. User clicks File → "Open Image Folder…"

2. AppWindow._open_image_folder() calls QFileDialog.getExistingDirectory(...)

3. AppWindow calls: io_controller.load_folder(directory)

4. IOController.load_folder():
   - Scans the directory for image files
   - Calls: model.load_folder(directory, sorted_file_list)

5. DatasetTableModel.load_folder():
   - Calls beginResetModel()      ← tells Qt "the whole table is changing"
   - Clears and repopulates state
   - Calls endResetModel()        ← tells Qt "the table is ready again"

6. modelReset fires. Connected slots are called:
   - AnnoMateWindow._on_model_reset() → loads the first image automatically
   - DataNavigatorSection._rebuild_list() → destroys old row widgets, builds new ones
   - ClassesSection._rebuild_classes() → rebuilds the class list

7. _on_model_reset() calls _load_row(0), which:
   - Reads the image from disk via io_controller.load_image_for_display(0)
   - Sets it on the canvas: canvas.set_image(bgr)
   - Updates the review bar, counter, and right panel for row 0
```

### Flow 3: MicroSentryAI Runs on All Images

```
1. User loads a model: AppWindow._open_load_model() → io dialog → model path string

2. AnnoMateWindow._load_model_from_path(path):
   - Shows "Loading model…" in status bar
   - Calls: inference_controller.load_model(path)   ← blocking, runs on main thread
   - Hides "Loading model…"
   - Calls: right_panel.set_model_loaded(name, path)
   - Calls: _start_pending_inference()

3. _start_pending_inference():
   - Builds a list of unprocessed image paths
   - Calls: inference_controller.start_batch_inference(paths)

4. InferenceController.start_batch_inference():
   - Creates InferenceWorker(strategy, paths) — a QThread
   - Connects worker.resultReady → controller.result_ready
   - Connects worker.progress → controller.progress
   - Connects worker.finished → controller.batch_done
   - Calls worker.start()    ← runs in a background thread

5. Background thread processes each image:
   - strategy.predict(path) returns (_, score_map)
   - worker.resultReady.emit(path, score_map)
   - worker.progress.emit(count)

6. On the main thread, connected slots run:
   - inference_model.set_score_map(path, score_map)
   - status_bar.set_inference_progress(done, total)
   - If the current image was just processed: _refresh_canvas_render()

7. _refresh_canvas_render():
   - Gets the score map from inference_model
   - Applies Gaussian smoothing
   - Calls canvas.set_heatmap_layer(score_map, alpha, heat_min)
   - Computes AI polygon contours if segmentation is enabled
   - Calls canvas.set_ai_overlays(contours)
```

---

## 12. The AI Features: SAM2 and MicroSentryAI

### AI Strategy Pattern

Both AI systems follow the same structural pattern. There is an abstract base class (`ai_strategies/interface.py`) and two concrete implementations:

- `AnomalibStrategy` — wraps Anomalib for anomaly scoring
- `SAMStrategy` — wraps Meta SAM2 for bounding-box segmentation

The strategy classes have **zero Qt imports**. They are instantiated and used by their respective controllers. This means the AI backends can be swapped out, tested in isolation, or used from a plain Python script without any GUI.

### MicroSentryAI (Anomalib)

The `AnomalibStrategy` loads a `.pt` model file trained with the Anomalib library. Calling `strategy.predict(image_path)` returns a score map — a 2D NumPy float array aligned with the image where higher values indicate more anomalous pixels.

The `InferenceController` orchestrates the full pipeline:
1. Loads the model from a `.pt` file
2. Runs all images in a background thread (so the UI stays responsive)
3. Stores score maps in `InferenceModel`
4. Provides visualization helpers (heatmap rendering, polygon extraction)

### SAM2 (Segment Anything)

SAM2 takes an image and a bounding box and returns a segmentation mask. The `SAMStrategy` wraps this with:

- Automatic weight downloading on first use (weights go to `sam_weights/` in the project root)
- Four model size variants: tiny, small, base+, large
- Auto-load on startup if weights are already on disk

The `SAMController` manages two background threads:
- One for loading the model (avoids freezing the UI during the 30+ second load)
- One for each inference call (avoids freezing during prediction)

The user workflow:
1. Click the ✦ SAM tool in the tool palette
2. The SAM model begins loading in the background (status bar shows progress)
3. Once loaded, draw a bounding box around an object on the canvas
4. A "ghost" polygon appears showing the predicted boundary
5. Press Enter to accept it as a real annotation, or Esc to cancel

---

## 13. Project Files: Saving and Loading

A saved project is a **folder** with this structure:

```
my_project/
├── my_project.annoproj          # JSON metadata file
├── annotations.coco.json        # All annotations in COCO format
├── scoremaps.npz                # Compressed AI score maps (NumPy)
└── autosave/
    ├── my_project.autosave.annoproj
    └── annotations.coco.json
```

### The `.annoproj` File

This is a JSON file with:

```json
{
  "schema_version": "1.0",
  "project_name": "my_project",
  "created_at": "2025-01-01T00:00:00Z",
  "dataset": {
    "image_dir": "/absolute/path/to/images"
  },
  "inference": {
    "model_path": "/path/to/model.pt"
  }
}
```

Note that the `.annoproj` file stores only the **path** to the image directory, not the images themselves. If you move your images, use `File → Relocate Images…` to update the path.

### The COCO JSON

All actual annotations are stored separately in COCO format. COCO (Common Objects in Context) is an industry-standard format widely used in machine learning:

```json
{
  "images": [ {"id": 1, "file_name": "img1.jpg"} ],
  "categories": [ {"id": 1, "name": "crack"} ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[10, 20, 30, 40, 50, 20]],  // flat x,y pairs
      "area": 200.0,
      "bbox": [10, 20, 40, 20]
    }
  ]
}
```

### Orphaned Annotations

When you open a project but the image files have changed (e.g., images were renamed or deleted), some annotations may reference filenames that no longer exist in the folder. These are called **orphaned annotations**. The application warns you about them before saving, and they are silently dropped from the COCO file when you save.

---

## 14. Running the Tests

Tests are located in `src/tests/` and divided into three categories:

- `unit/` — test individual functions in isolation (no Qt needed)
- `model/` — test the model layer (requires a `QApplication`)
- `integration/` — test multi-layer interactions

```bash
# From the project root:
cd src
pytest tests/

# Run only unit tests:
pytest tests/unit/

# Run with verbose output:
pytest -v tests/
```

Because `core/` has zero Qt imports, unit tests for states, geometry, and persistence can run as plain Python without any special setup.

Model and integration tests typically require a `QApplication` object to exist. The `conftest.py` in `src/tests/` sets this up automatically.

---

## 15. How to Add a New Feature

This section walks through the steps for three common types of changes.

### Adding a New Annotation Property

Say you want to add a "confidence score" field to each annotation.

**Step 1: Core State** — add the field in `DatasetState.add_annotation()` and store it in the dict:
```python
# In dataset_state.py:
def add_annotation(self, image_name, category, polygon, thickness=2.0, confidence=1.0):
    self.annotations.setdefault(image_name, []).append({
        "category_name": category,
        "polygon": polygon,
        "thickness": thickness,
        "confidence": confidence,   # NEW
    })
```

**Step 2: Model** — update `DatasetTableModel.add_annotation()` to accept and pass through the new parameter.

**Step 3: View** — update wherever annotations are displayed (e.g., `AnnotationsSection`) to show the new field. Add a column to `_AnnotationRow` if needed.

**Step 4: Persistence** — update `ProjectIO.save_project()` (COCO export) and `apply_project_to_states()` (import) to read/write the new field.

### Adding a New Export Format

**Step 1: Controller** — add a new method to `IOController`:
```python
def export_xml(self, out_path: str) -> str:
    state = self.model.state
    if not state.image_files:
        raise RuntimeError("No images loaded.")
    # ... write XML ...
    return f"XML saved to:\n{out_path}"
```

**Step 2: View** — add a menu item in `AppWindow._build_menu()` and a handler:
```python
add(data_menu, "Export XML…", "", self._export_xml)

def _export_xml(self) -> None:
    out_path, _ = QFileDialog.getSaveFileName(self, "Save XML", "output.xml", "XML (*.xml)")
    if not out_path:
        return
    try:
        msg = self.io_controller.export_xml(out_path)
        QMessageBox.information(self, "Export", msg)
    except Exception as exc:
        QMessageBox.critical(self, "Export Error", str(exc))
```

No other layer needs to change.

### Adding a New Right Panel Section

**Step 1: Create the section widget** — create a new file in `src/views/annomate/sections/`:
```python
# src/views/annomate/sections/my_section.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

class MySection(QWidget):
    def __init__(self, dataset_model, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Hello from my new section!"))
```

**Step 2: Export it** — add it to `src/views/annomate/sections/__init__.py`.

**Step 3: Add it to RightPanel** — in `right_panel.py`, create a `_CollapsibleSection` for it and add it to the scrollable bottom area:
```python
my_sec = _CollapsibleSection("My Section")
self.my_section = MySection(dataset_model)
my_sec.body_layout().addWidget(self.my_section)
cl.addWidget(my_sec)
```

---

## 16. Common Pitfalls and Rules to Remember

### Rule 1: Never import Qt types in `core/`

The entire `core/` folder must remain importable without a `QApplication`. If you add a `from PySide6...` import to any file in `core/`, you break the ability to unit-test it.

**Wrong:**
```python
# In core/states/dataset_state.py — DO NOT DO THIS
from PySide6.QtGui import QColor
self.class_colors = {"crack": QColor(255, 0, 0)}
```

**Right:**
```python
# Colors as plain tuples in core/
self.class_colors = {"crack": (255, 0, 0)}
# Convert to QColor only at the widget paint boundary, in views/
```

### Rule 2: Controllers never open dialogs

Controllers raise exceptions. Views catch them and show `QMessageBox`. If you find yourself writing `QMessageBox.critical(...)` inside a controller method, move it to the view.

### Rule 3: Views never store canonical data

A view widget may have local state for UI purposes (e.g., which row is currently highlighted), but it must never be the single source of truth for application data. All real data lives in the models/states. If a view holds a copy of data, it must be a derived cache that is always rebuilt from the model on `dataChanged`.

### Rule 4: Sibling views never call each other directly

`AnnotationsSection` and `DataNavigatorSection` are siblings inside `RightPanel`. They should never import or reference each other. If one needs to react to something the other does, that communication must go through the shared model (via signals) or be routed through the parent `RightPanel` or `AnnoMateWindow`.

### Rule 5: Always emit `dataChanged` after a mutation

When you add a mutation method to a model, the last thing it must do is emit `dataChanged` (or `modelReset` for full reloads). If you forget this, views will appear out of sync — the data changes in memory but the screen does not update.

```python
def my_new_mutation(self, row, value):
    if not (0 <= row < self.rowCount()):
        return
    self.state.some_field[self.state.image_files[row]] = value
    self._emit_row(row)  # NEVER forget this
```

### Rule 6: Block signals when programmatically updating a widget

When your code sets a widget's value (not the user), you often need to block signals to prevent an accidental feedback loop:

```python
# WRONG — this will emit valueChanged → trigger _on_value_changed → loop
self.my_slider.setValue(42)

# RIGHT — block signals, set value, unblock
self.my_slider.blockSignals(True)
self.my_slider.setValue(42)
self.my_slider.blockSignals(False)
```

This pattern appears throughout the codebase whenever a widget is updated programmatically, such as when loading a new image updates the thickness slider to match the selected annotation.

### Rule 7: Canvas coordinates vs. image coordinates

The `ImageLabel` canvas displays images at a zoom factor. When the user clicks on the canvas, the raw click position is in **display space** (pixels on screen). But annotations are stored in **image space** (pixels in the original image).

`ImageLabel` handles this conversion internally. All signals it emits (`polygonFinished`, `polygonEdited`, `samBboxDrawn`) are already converted to image coordinates. You never need to do this conversion outside of `ImageLabel`.

### Rule 8: Background threads may not touch the GUI

Qt is single-threaded for all UI operations. If you try to update a label from a `QThread.run()` method, you will get crashes or undefined behavior.

The correct pattern is: the background thread emits a signal, and the signal handler (which runs on the main thread) updates the GUI. This is exactly how `InferenceWorker` and `SAMWorker` are designed — they emit signals, and `AnnoMateWindow` or the controllers have the connected slots that do the actual widget updates.
