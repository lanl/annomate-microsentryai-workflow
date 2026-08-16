# Try the Demo

A ready-to-open project so you can try AnnoMate & MicroSentryAI without gathering your own images or model. No setup beyond the normal environment install.

**What's here:**
* `AnnoMate_Demo.annoproj` — a pre-built project pointing at the data and model below. Classes (`gouge`, `inclusion`, `nick`) are already registered, matching the defect types present in the sample images.
* `data/` — 8 sample part images (2 each of `good`, `gouge`, `inclusion`, and `nick`) from a real manufacturing QA dataset.
* `3-efficientad.pt` — a small EfficientAD anomaly-detection model (trained via [Anomalib](https://github.com/openvinotoolkit/anomalib)) for the MicroSentryAI inference engine.

## Running it

1. Follow [Getting Started](../docs/GettingStarted.md) to install the conda environment for your platform.
2. From the repo root, launch the app:
   ```bash
   python src/main.py
   ```
3. On the start screen (or via **File > Open Project…**), open `demo/AnnoMate_Demo.annoproj`.
4. The 8 sample images load into the **Dataset Navigator** on the left.

## Loading the AI model

The project remembers the model's path, but doesn't auto-load it (loading triggers a batch inference pass, so it's an explicit step):

1. Open the **Microsentry AI** tab in the right panel's activity bar (the sparkle icon), and expand the **Microsentry** section if it isn't already.
2. Click **Load Previous**. MicroSentryAI will detect your hardware (CUDA/MPS/CPU) and run inference across all 8 images in the background.
3. Once it finishes, toggle **Enable Heatmap** and/or **Enable Segmentation** in that same section to see defect heatmaps and AI-generated outlines on the canvas.

See the [MicroSentryAI Guide](../docs/MicroSentryAI.md) for what the heatmap/threshold/smoothing controls do.

## Things to try

* **Polygon tool (`P`):** manually trace a defect on one of the `gouge`, `inclusion`, or `nick` images.
* **SAM 2 tool (`S`):** drag a box around a defect and let Segment Anything generate the polygon for you (first use downloads SAM 2 weights — needs internet).
* **Accept AI Polygons:** after enabling the MicroSentryAI overlay, adjust the threshold slider and convert the AI's dashed outlines into real annotations.
* **Accept/Reject:** mark a part's overall quality using the floating Accept/Reject bar in the canvas's top-right corner.

Full walkthroughs live in the [AnnoMate Guide](../docs/AnnoMate.md) and [MicroSentryAI Guide](../docs/MicroSentryAI.md).
