# Annotating

Annotations mark where defects are. Each one belongs to a **class** such as "scratch" or "dent".

## Step 1: Set up classes

Open the **Dataset Setup** tab (in the tab bar on the right).

- Type a name under **Enter new class name** and click **Add Class**.
- Change a class's colour or delete it from the class list.
- Use the eye control next to a class to show or hide all of that class's annotations.
- To reuse a class list, use **Data → Export Annotation Classes** and **Data → Import Annotation Classes…**.

You must have at least one class before you can create an annotation.

## Step 2: Choose an annotation mode

At the top of the Dataset Setup tab, choose how you want to work:

- **Pixel Level**: you draw polygons around defects. A rejected image needs at least one polygon to count as reviewed.
- **Image Level**: you tag whole images with classes instead of drawing. Drawing tools are turned off. A rejected image needs at least one class tag to count as reviewed.

To tag an image in Image Level mode, expand its card in the Dataset Navigator and click a class to tag it. Click again to remove the tag.

## Step 3: Draw

The drawing tools are on the tool palette. Press a tool's key again to turn it off, or press **Esc** to cancel.

### Polygon (`P`)

1. Click to place each corner around the defect.
2. To finish, click near your first point.
3. Press **Backspace** to remove the last corner you placed.
4. A small class picker appears next to your shape. Choose a class, then click the tick to keep the annotation, or the **X** to discard it.

### SAM Segment (`S`)

SAM 2 draws the outline for you.

1. Open the **Active Tool** tab and choose a **Variant**. Tiny is the fastest and Large is the most accurate.
2. Drag a box around the object.
3. An outline is suggested. Press **Enter** to accept it, or **Esc** to throw it away.
4. Choose a class in the picker and click the tick, as with the polygon tool.

The first time you use a SAM size, its model file is downloaded to the `sam_weights` folder. On a computer without internet, copy that folder from a machine that already has it.

### Edit Points (`N`)

Fine-tune an existing polygon. Click a polygon, then:

- Drag a corner to move it, or drag inside the shape to move the whole polygon.
- In the **Active Tool** tab, choose **Add Point** (click an edge to add a corner) or **Delete Point** (click a corner to remove it).
- Click empty space to stop editing.

### Measure Distance (`M`)

Click two points on the image to see the distance between them on the canvas. Use **Clear Measurement** in the Active Tool tab to remove it. To get real units such as mm instead of pixels, calibrate first. See *View Tools*.

## Managing annotations

- Click a polygon to select it, then press **Delete** to remove it.
- Expand an image's card in the Dataset Navigator to see its list of annotations. From there you can show or hide each one, or delete it.
- The **Active Tool** tab has a **stroke width** slider that changes how thick outlines are drawn.
