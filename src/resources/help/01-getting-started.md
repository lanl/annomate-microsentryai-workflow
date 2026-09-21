# Getting Started

AnnoMate with MicroSentryAI is a desktop tool for reviewing images of manufactured parts. You can mark defects by hand, let the SAM 2 tool outline them for you, and run your own trained AI model to highlight likely defects.

## Start a project

When the app opens you will see a **Start a project** panel with these choices:

- **Open Project** opens a saved `.annoproj` project.
- **Open Image Folder** loads a folder of images. Subfolders are included. Supported formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif` and `.tiff`.
- **Recent** lists your last project and image folder so you can jump back in.

You can also use the **File** menu: **New Project**, **Open Project…**, **Open Image Folder…**.

## The main window

- **Left: Dataset Navigator.** The list of your images with their review status. See *Reviewing Images*.
- **Center: the canvas.** The current image and your annotations.
- **Top right of the canvas: Accept / Reject.** Your decision for the current image.
- **Bottom center of the canvas: Zoom In, Zoom Out and Reset View** buttons.
- **Far right: the tab bar.** A column of icons, each opening a panel:

| Tab | What it holds |
|---|---|
| Active Tool | Settings for the tool you are using (stroke width, SAM size, point editing) |
| Dataset Setup | Annotation classes and the annotation mode |
| Microsentry AI | Load an AI model and view its results |
| View Overlays | Center Crop, Grid and Anomaly Constraints |
| Image Adjustments | Colour and contrast preview tools |

Click a tab to open its panel. Click the same tab again to collapse the panel and give the canvas more room. The app remembers which tab you had open.

## Move around the image

- Scroll to zoom in and out.
- Right-click and drag to pan.
- Use the buttons at the bottom of the canvas to zoom, or **Reset View** to fit the whole image on screen.
- Press **A** and **D** to go to the previous and next image.

## Save your work

Choose **File → Save Project** (`Ctrl+S`) or **Save Project As…** (`Ctrl+Shift+S`). A project is saved as a `.annoproj` folder containing your annotations, classes and any AI results. The app also autosaves every few minutes when there are unsaved changes, and shows "Autosaved" in the status bar.

If you move your images to a different folder later, use **File → Relocate Images…** to point the project at the new location. Your annotations are kept.
