# Importing a Public Dataset

Use **Data → Import Public Dataset…** to turn a public anomaly-detection benchmark into a working project, instead of setting up classes and decisions by hand. The dataset's own train/test split becomes Accept/Reject decisions, its defect folders or labels become your annotation classes, and its ground-truth masks become polygons, all in one step.

Each import covers **one category at a time** (for example, one MVTec object type, or one VisA object type). A category is its own project with its own class list; importing a second category starts a separate project rather than adding to the first, since the two don't share images or classes.

## Before you start

The dataset must be **on disk in its original, unmodified format**: exactly as the benchmark's own creators structured it, not a folder you've reorganized, renamed, or partially copied. The dialog reads the dataset's own folder/file conventions directly, so a rearranged copy will either show no categories to choose from, or show incomplete or incorrect statistics.

## Steps

1. Open **Data → Import Public Dataset…**. If your current project has unsaved changes, you'll be asked to save or discard them first, since importing replaces what's currently loaded.
2. Choose the dataset's **Format** from the dropdown.
3. Click **Browse…** and select the dataset's root folder.
4. Choose a **Category** from the dropdown. This list is only populated once a valid root folder for the chosen format is selected.
5. Check the statistics shown: total images, how many are normal vs. defective, and the defect classes found. This is your chance to confirm the folder you picked is the one you meant to import, before anything is loaded.
6. Click **Import**.

Nothing is written to disk during this step. The import only loads the data into the app, the same as opening an image folder. Use **File → Save Project As…** afterward to save it as a normal `.annoproj` project.

## What gets set automatically

- **Accept / Reject**: set from the dataset's own normal/defective labels.
- **Annotation classes**: one class per defect type the dataset defines, with polygons traced from its ground-truth masks.
- **Image notes and tags**: where a format records more detail than a single class can capture (see *VisA* below), that detail is kept as a note or an image-level tag rather than dropped.

## Supported formats

### MVTec AD

Expects each category folder to contain `train/`, `test/` and `ground_truth/` subfolders, with defect types as subfolders under `test/` and `ground_truth/`. This is the standard layout the benchmark ships in.

### MVTec AD 2

Expects `train/`, `test_public/` (with its own `good/`, `bad/` and `ground_truth/` subfolders) under each category, plus `validation/`, `test_private/` and `test_private_mixed/`. The two private test splits ship with no released ground truth. Those images import with no Accept/Reject decision, so you can review and decide on them yourself.

### KolektorSDD

Expects the **original** per-part folder layout (numbered part folders, each holding an image and its paired label file), not a copy that has already been reorganized into `train`/`test` folders. There's no category to choose; the whole dataset imports as a single project.

### VisA

Expects each category folder to contain its own manifest file listing every image, its label, and its mask path: the format this benchmark distributes its ground truth in, rather than folder-name conventions. Because a single image can carry more than one defect description at once, only the first-listed one becomes the pixel annotation's class; the full list is kept as that image's tags, and the original, unshortened description is kept as its note.

## Tips

- If the category dropdown stays empty after choosing a root folder, the folder you picked likely isn't the format's expected top-level folder. Try its parent or a subfolder.
- The statistics step is worth reading closely: a defect count of zero, or a much smaller total than you expect, usually means part of the dataset is missing from the folder you selected.
