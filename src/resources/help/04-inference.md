# Inference (MicroSentryAI)

MicroSentryAI runs an AI model that you have trained on your own images and shows where it thinks defects are, as a heatmap and as suggested outlines.

## About Anomalib

The models used here are created with **Anomalib**, an open-source Python library for visual anomaly detection, maintained under Intel's Open Edge Platform. Anomalib provides a common way to train and export many anomaly-detection algorithms, such as PatchCore, EfficientAD and DRAEM.

Anomaly detection models are trained mostly on images of **good** parts. They learn what "normal" looks like, and then flag anything that looks different, without needing every kind of defect labelled in advance. That makes them well suited to inspection tasks where defects are rare or varied.

Anomalib is not part of this app. You train a model in Anomalib first, then load the result here. More information: https://github.com/open-edge-platform/anomalib

## Which file to load

Load the **exported model file**, named `model.pt`. Anomalib places it in the `weights/torch` folder of the training run. Do not load the training checkpoint (`.ckpt`). The app will tell you if you pick one by mistake.

## Run a model

1. Open a project with images.
2. Open the **Microsentry AI** tab and click **Load New Model**.
3. Choose your model file. A progress indicator shows while it loads.
4. The model then scores your images automatically. Each image's score appears in the Dataset Navigator when it is done.

If you load a different model later, the app asks you to confirm before switching away from the current one.

## View the results

- **Enable Heatmap** overlays a colour heatmap on the image. The **Transparency** slider controls how much of the image shows through.
- **Enable Segmentation** outlines the regions the model considers anomalous. The **Threshold** slider sets how strict this is: raise it to outline only the strongest anomalies, lower it to include more.
- Click an outline on the canvas to select it.

### Advanced Settings

Open **Advanced Settings** to fine-tune the display:

| Setting | What it does |
|---|---|
| Simplify Tolerance | Higher values give outlines with fewer, smoother corners |
| Heatmap Minimum | Scores below this are not coloured |
| Heatmap Ceiling | Scores above this are shown at full intensity |
| Heatmap Gamma | Adjusts how quickly colour intensifies |
| Heatmap Colormap | The colour scheme used for the heatmap |

## Turn AI outlines into annotations

Click **Accept AI Polygons** to add the suggested outlines as annotations. You can also select a single outline on the canvas and choose a class in the picker. Either way, you can then edit them with the Edit Points tool like any polygon you drew yourself.

## Saved results

Scores are saved with your project, so you can reopen it and see the same heatmaps without running the model again. If you have used more than one model, the **Cached Model** drop-down lets you switch which one's results you are viewing. An indicator appears when the current results have not been saved yet.
