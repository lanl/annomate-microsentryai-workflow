# AnnoMate User Guide

AnnoMate is the core manual annotation and review component of the suite. It allows users to create ground-truth segmentations, review part quality, manage project files, and export data.

## 1. Interface Overview
* **Left Panel (Dataset Navigator):** Lists every loaded image. Collapsible to a narrow icon rail — it auto-collapses whenever no dataset is loaded, and auto-expands once one loads.
* **Tool Palette:** The fixed column between the left panel and canvas. Contains the Polygon, SAM Segment, and Measure Distance tools.
* **Canvas:** Your primary workspace. Here you can zoom (Scroll Wheel), pan (Right-Click + Drag), and draw or edit annotations. A floating, draggable Accept/Reject bar sits top-right; zoom controls sit bottom-center.
* **Right Panel:** A five-tab activity bar — Active Tool, Dataset Setup, Microsentry AI, View Overlays, and Image Adjustments. Click a tab's icon to expand its panel; click the active tab again to collapse it back down to just the icon rail. The panel remembers which tab you last had open.
* **Status Bar:** Along the bottom — zoom level, image dimensions, the active tool/class, session time, and MicroSentryAI batch progress.

## 2. Project Management (`.annoproj`)
AnnoMate uses a robust project system. A `.annoproj` file saves your images, class definitions, annotations, inspector notes, and the path to your loaded AI model all in one place.

### Creating a New Workspace
1. Go to **File > Open Image Folder...** and select a local directory containing your dataset (`.jpg`, `.png`, `.bmp`, `.tif`) — or start from **File > New Project** (`Ctrl+N`) first if you'd rather set up classes before loading images.
2. The images will load into the **Dataset Navigator** on the left panel.

### Saving Your Progress
1. Go to **File > Save Project As...**
2. Choose a location and name for your project. This will generate your `.annoproj` file and an associated `annotations.coco.json` file.
3. **Autosave:** Once a project is saved, the application will automatically create an `autosave` backup inside your project folder every 5 minutes while you work.

### Relocating Images
Annotations are saved using relative paths to your image folder. If you ever move your image folder on your hard drive, your `.annoproj` file will warn you that the images are missing.
* Fix this by going to **File > Relocate Images...** and selecting the new folder location. Your annotations will immediately map to the new file paths. If the new folder's contents don't fully match the old ones, you'll see an orphaned-annotation warning before anything is overwritten.

## 3. The Dataset Navigator
The **Dataset Navigator** lives in the left panel and lists every image in your loaded dataset.

* **Navigation:** Click a card, or use the **Prev/Next** buttons (`A`/`D` keys) to step through images. The header shows your position, e.g. "3 / 12".
* **Collapsing:** Click the arrow at the top of the panel to collapse it to a narrow icon rail — you still get Prev/Next, the position counter, and live status counts there.
* **Status Markers:**
  * `○` **Undecided** (grey ring) — no Accept/Reject decision yet, no annotations.
  * `●` **Reviewed** (green dot) — accepted, or rejected with a polygon or class tag attached.
  * `!` **Incomplete** (orange) — needs a second look: rejected with no supporting annotation, accepted but still has leftover annotations, or has annotation work with no decision yet.
* **Filtering & Sorting:** Click **Filter** to open a panel with **Decision**, **Status**, and **Class** checkboxes (each showing live counts), plus **Sort by** (Filename / Annotations / Score — click again to reverse direction). **Clear Filters** resets everything. Three quick-filter chips matching the status markers sit next to the Filter button for one-click filtering.
* **Expanding a card:** Click it to expand in place and reveal:
  * **Annotations** (Pixel Level mode) or **Image Classes** (Image Level mode) — see Section 4.
  * **Metadata:** an Inspector name field (**Set** applies it to this image; **Set All** bulk-applies it across All / Reviewed / In Review images) and a collapsible Note field.

## 4. Creating Annotations
AnnoMate has three tools, selected from the tool palette or by hotkey.

### ⬠ Polygon Tool (Shortcut: `P`)
Used for manual, point-by-point drawing.
1. Click the **Polygon Tool** (⬠) or press `P`.
2. **Left Click:** Place vertices on the canvas.
3. **Backspace:** Undo the last vertex placed.
4. **Double-Click (or click near the start point, once ≥3 points are placed):** Finish the polygon.
5. **Escape:** Cancel the current drawing.
6. Once finished, a popup appears beside the shape — pick a class from the dropdown and click **Accept**, or discard it. Classification happens *after* you draw, not before.

### ✦ SAM Segment Tool (Shortcut: `S`)
Uses Meta's *Segment Anything 2* AI to automatically generate a precise polygon mask from a bounding box. *(Note: Requires internet access the first time a given model variant is used, to download its weights.)*
1. Click the **SAM Segment Tool** (✦) or press `S`.
2. **Left Click & Drag:** Draw a bounding box tightly around the defect.
3. The AI generates a mask and converts it to a polygon.
4. The same class-picker popup used by the Polygon tool appears — choose a class and **Accept**, or discard it.
5. **Escape:** Cancel an in-progress box or a pending result.
* *Tip: Pick the SAM model variant — Tiny, Small, Base+, or Large, trading speed for accuracy — from the **Active Tool** tab in the right panel while the SAM Segment tool is selected.*

### 📏 Measure Distance Tool (Shortcut: `M`)
Only enabled once a pixel-to-real-world calibration exists (set one up under **View Overlays > Grid** — see Section 6). Click a first point, then a second; the live distance between them is drawn directly on the canvas.

## 5. Editing & Reviewing
* **Modify Shapes:** With no tool selected, click inside a polygon to drag the entire shape, or click and drag a specific vertex (dot) to adjust its outline.
* **Adjust Line Width:** Open the **Active Tool** tab in the right panel — its **Line Width** slider sets the thickness of the currently selected polygon (or of newly drawn ones if nothing is selected).
* **Delete:** Select a polygon and press the **Delete** key, or click the Trash icon next to it in the expanded card's Annotations list.
* **Annotation Mode:** The **Dataset Setup** tab includes a **Pixel Level / Image Level** toggle. Pixel Level is the polygon-based workflow above (the default). Image Level instead tags whole images with one or more classes and disables the Polygon/SAM drawing tools — tag classes from the Image Classes list on an expanded card, which is only editable while that image's decision is set to Reject.
* **Accept/Reject Part:** Use the floating, draggable **✓ Accept** / **✗ Reject** buttons at the top right of the canvas to mark the part's quality — this also drives the navigator's status marker and the exported decision.
* **Inspector Notes:** Use an expanded card's **Metadata** section (left panel) to log an inspector name and any notes about the part. **Set as session inspector** auto-fills your name onto every image you view afterward; **Set All** bulk-assigns it across All / Reviewed / In Review images at once.

## 6. The Right Panel
Five tabs, each collapsed to an icon in the rail until clicked.

### Active Tool
Always shows a **Line Width** slider (1–40px). While a tool is selected, tool-specific options appear below it: the SAM Segment tool shows a model **Variant** picker (Tiny / Small / Base+ / Large) and a load-status label; the Measure tool shows a **Clear Measurement** button. Polygon and Calibrate have no extra settings.

### Dataset Setup
* The **Pixel Level / Image Level** toggle (see Section 5).
* The Annotation Classes list: a color swatch (click to change it), the class name, its total annotation count, a visibility (eye) toggle, and a delete button (warns if the class is still in use anywhere).
* An **Add Class** field to define new classes — each gets the next unused default color automatically.

### Microsentry AI
Batch AI-assisted anomaly detection and defect heatmaps. See the [MicroSentryAI Guide](MicroSentryAI.md).

### View Overlays
Three collapsible display overlays — none of these alter your source images or annotations:
* **Center Crop:** Overlays a rectangle or circle centered on the image, with adjustable size, opacity, and border color. **Calibrate Center** lets you drag it into position and lock it to a saved template so it automatically re-aligns on every image.
* **Grid:** Set a pixel-to-real-world calibration — type a ratio directly, or click two points on the image and enter the real distance between them — then enable a scaled reference grid with adjustable spacing, opacity, and color. This calibration is also what powers the Measure Distance tool.
* **Anomaly Constraints:** Flags annotations that break configurable limits: an **Area Threshold** highlights any annotation larger than a set maximum area, and a **Proximity Threshold** highlights annotations sitting closer together than a minimum distance (measured center-to-center or edge-to-edge). Violation counts update live on the canvas.

### Image Adjustments
Two collapsible, purely visual preview controls — they never touch your source images:
* **HSV:** Hue, Saturation, and Value sliders.
* **Brightness/Contrast:** Min/Max sliders for a linear contrast stretch.

## 7. Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` / `D` | Previous / Next image |
| `P` | Toggle Polygon tool |
| `S` | Toggle SAM Segment tool |
| `M` | Toggle Measure Distance tool (requires a calibration) |
| `C` | Toggle Calibrate tool |
| `Delete` | Delete the selected annotation |
| `Escape` | Cancel the active tool / discard a pending shape |
| `Ctrl+N` | New Project |
| `Ctrl+O` | Open Project |
| `Ctrl+S` | Save Project |
| `Ctrl+Shift+S` | Save Project As |
| `Ctrl+Q` | Exit |
| `F1` | Search Help |

## 8. Getting Help
* The **Help** menu has an inline search box that live-filters File/Data menu commands as you type.
* **Search Help…** (`F1`) opens a full documentation search, including a "Show Me in the App" button that highlights the matching control live on screen.
* **Show Welcome Tour** replays the guided onboarding tour that runs automatically the first time you launch the app.

## 9. Exporting & Importing Data
Found under the **Data** menu:
* **Import Annotation Classes…:** Loads class definitions from a `.txt` file.
* **Export Annotation Classes:** Writes your current class list to the project directory.
* **Export Binary Masks…:** Renders pure black-and-white `.png` images (defects are white, background is black). This is the standard format required for training AI models.
* **Export CSV…:** Generates a spreadsheet containing the Image Name, Inspector, Notes, Accept/Reject decision, and classes present.
* **Export COCO JSON…:** Exports annotations in standard COCO JSON format.
* **Export Pixel-Level Train Structure… / Export Image-Level Train Structure…:** Builds a ready-to-train folder structure matching the corresponding annotation mode.
* **Export Project Template…:** Exports your class/setting scaffolding — without images or annotations — as a starting point for a new project. Requires the project to be saved first.
