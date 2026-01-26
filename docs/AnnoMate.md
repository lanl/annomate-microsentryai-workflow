# AnnoMate User Guide

AnnoMate is the manual annotation and exportation component of the suite. It allows users to create ground-truth segmentations for datasets, keep note of key information, and export ground truth masks as images and key information as CSV/XLSX.

## Interface Overview

* **Left Panel (Canvas):** Displays the image. This is where you draw and edit polygons.
* **Right Panel (Controls):** Contains navigation, class management, and file metadata.

## Workflow

### 1. Loading Data
* Click **Open Folder** in the top right "Tray" section.
* Select a directory containing images (`.jpg`, `.png`, `.bmp`, `.tiff`).
* The table will populate with the file list. Rows colored **Green** have already been reviewed; **Yellow** rows are pending.

### 2. Navigation
* Use the **Prev** and **Next** buttons to cycle through images.
* **Zoom/Pan:**
    * **Scroll Wheel:** Zoom in/out.
    * **Right-Click + Drag:** Pan the image.
    * **Reset View:** Restores the image to fit the window.

### 3. Creating Annotations (Polygons)
1.  **Select a Class:** Type a name in the text box (e.g., "Scratch") and click **Add Class**, or select an existing one from the dropdown.
2.  **Activate Tool:** Click the **Polygon** button.
3.  **Draw:**
    * **Left Click:** Add points to the polygon.
    * **Double Click** (or click the start point): Close the polygon and save it.
    * **Backspace:** Undo the last point while drawing.
    * **Escape:** Cancel the current drawing.

### 4. Metadata & Review
* **Inspector:** Enter your name/ID to mark who reviewed the image.
* **Notes:** Add text notes regarding ambiguity or specific defect details.
* **Status:** Once an annotation is added or metadata is modified, the image status automatically updates to "Reviewed."

### 5. Exporting
AnnoMate supports exporting your work for external use:
* **Export Polygons + Data:** Saves a JSON file containing all coordinates and metadata, plus visual "burned-in" overlays of your annotations for quick review.
* **Export CSV:** Saves a spreadsheet summary of the dataset status (Inspector, Notes, Classes present).
* **Import Data JSON:** Loads previous sessions or annotations sent from other tools.