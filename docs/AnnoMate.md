# AnnoMate User Guide

AnnoMate is the core manual annotation and review component of the suite. It allows users to create ground-truth segmentations, review part quality, manage project files, and export data.

## 1. Interface Overview
* **Left Canvas:** Your primary workspace. Here you can zoom (Scroll Wheel), pan (Right-Click + Drag), and draw polygons.
* **Left Tool Palette:** Contains the Polygon tool, SAM 2 tool, and Brush Thickness controls.
* **Right Panel:** Contains the Dataset Navigator, Annotation Classes, Current Image Annotations, and Metadata sections.

## 2. Project Management (`.annoproj`)
AnnoMate uses a robust project system. A `.annoproj` file saves your images, class definitions, annotations, inspector notes, and the path to your loaded AI model all in one place.

### Creating a New Workspace
1. Go to **File > Open Image Folder...** and select a local directory containing your dataset (`.jpg`, `.png`, `.bmp`, `.tif`).
2. The images will load into the **Dataset Navigator** on the right panel.

### Saving Your Progress
1. Go to **File > Save Project As...**
2. Choose a location and name for your project. This will generate your `.annoproj` file and an associated `annotations.coco.json` file.
3. **Autosave:** Once a project is saved, the application will automatically create an `autosave` backup inside your project folder every 5 minutes while you work.

### Relocating Images
Annotations are saved using relative paths to your image folder. If you ever move your image folder on your hard drive, your `.annoproj` file will warn you that the images are missing. 
* Fix this by going to **File > Relocate Images...** and selecting the new folder location. Your annotations will immediately map to the new file paths.

## 3. The Dataset Navigator
The top of the right panel displays your loaded images.
* **Navigation:** Click a row or use the **Prev/Next** buttons to switch images.
* **Status Dots:** 
  * 🟢 **Green (Reviewed):** The image has annotations, an inspector name, a note, or a final Accept/Reject decision.
  * 🟠 **Orange (Pending):** The image has not been interacted with.

## 4. Creating Annotations

### ⬠ Polygon Tool (Shortcut: `P`)
Used for manual, point-by-point drawing.
1. Select a class from the **Annotation Classes** list.
2. Click the **Polygon Tool** (⬠).
3. **Left Click:** Place vertices on the canvas.
4. **Backspace:** Undo the last vertex placed.
5. **Double-Click (or click the start point):** Close and save the polygon.
6. **Escape:** Cancel the current drawing.

### ✦ SAM 2 Bounding Box Tool (Shortcut: `S`)
Uses Meta's *Segment Anything 2* AI to automatically generate precise polygon masks. *(Note: Requires internet access on first use to download model weights).*
1. Click the **SAM Tool** (✦). 
2. **Left Click & Drag:** Draw a bounding box tightly around the defect.
3. The AI generates a dashed "Ghost Polygon" preview.
4. **Enter:** Accept the ghost polygon as an annotation.
5. **Escape:** Reject the ghost polygon.
* *Tip: Click the gear icon (⚙) below the SAM tool to switch variants. "Tiny" is fastest; "Large" is most accurate.*

## 5. Editing & Reviewing
* **Modify Shapes:** With no tool selected, click inside a polygon to drag the entire shape, or click and drag a specific vertex (dot) to adjust its outline.
* **Adjust Thickness:** Click the **Brush Thickness (◢)** icon. If a polygon is selected, the slider adjusts that specific polygon's line thickness.
* **Delete:** Select a polygon and press the **Delete** key, or click the Trash icon in the "Current Image Annotations" list.
* **Accept/Reject Part:** Use the floating **✓ Accept** or **✗ Reject** buttons at the top right of the canvas to mark the part's quality.
* **Inspector Notes:** Use the bottom-right **Metadata** section to log your name and any notes about the part.

## 6. Exporting & Importing Data
Found under the **Data** menu:
* **Export Polygons + Data:** Saves a `_data.json` file containing all polygon coordinates and metadata. It also generates JPEG images with the annotations permanently "burned in" for sharing.
* **Export Binary Masks:** Renders pure black-and-white `.png` images (defects are white, background is black). This is the standard format required for training AI models.
* **Export CSV:** Generates a spreadsheet containing the Image Name, Inspector, Notes, Accept/Reject decision, and classes present.
* **Import JSON Data:** Loads previously exported AnnoMate JSON files or standard **COCO JSON** files directly into your workspace.