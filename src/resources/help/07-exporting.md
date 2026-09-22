# Exporting Data

Use the **Data** menu to send your work elsewhere. All exports use the annotations in your current project.

| Menu item | What you get |
|---|---|
| Export COCO JSON… | Your annotations in COCO format, readable by most training tools |
| Export Binary Masks… | A black-and-white mask image for each annotated image |
| Export CSV… | A spreadsheet report for QA tracking |
| Export Pixel-Level Train Structure… | A training folder with a mask for every defect, made from your polygons |
| Export Image-Level Train Structure… | A training folder sorted into good and defect images, made from your image-level tags, with no masks |
| Export Annotation Classes | Your class list, so you can reuse it in another project |
| Export Project Template… | Your classes, calibration and constraint settings as a reusable template, without images or annotations |

## Tips

- Choose **Pixel-Level** when your images have polygons, and **Image-Level** when you only tagged images as a whole.
- Save your project first with `Ctrl+S`. Exports do not replace saving.
- Exports are for use elsewhere. To keep working in AnnoMate, save a project (`.annoproj`) instead.
