# MicroSentryAI Guide

MicroSentryAI is the integrated batch inference engine. It allows you to load a trained anomaly detection model, run it across your entire dataset in the background, and project defect heatmaps and suggested polygons directly onto your images.

## 1. Supported Models

The system currently supports PyTorch (`.pt`/`.pth`) models, specifically those trained via **Anomalib** (e.g., PatchCore, EfficientAD, Padim).

*Note: The file must be an exported inference checkpoint (`.pt`/`.pth`), not a raw training `.ckpt` file. Loading a `.ckpt` produces an explicit error asking you to export `model.pt` instead.*

## 2. Loading an AI Model

1. In the right panel's activity bar, open the **Microsentry AI** tab (sparkle icon).
2. Click **Load New** and select your trained model file.
3. The system will automatically detect your best hardware (NVIDIA CUDA, Apple MPS, or CPU).
4. **Batch Inference:** Upon loading, the system will process every loaded image in the background. A progress bar appears at the bottom of the window ("Microsentry: done / total"). Doing this upfront ensures instant, smooth navigation between images later. Any images added to the dataset afterward are inferred automatically too, without needing to reload the model.

*Note: If you save your `.annoproj` project, the application remembers your model path. Next time you open the project, simply click **Load Previous**.*

## 3. Microsentry AI Controls

All controls live directly in the **Microsentry AI** tab — there is no separate menu toggle. Enabling Heatmap or Segmentation immediately overlays the corresponding layer on the canvas for the currently displayed image.

### Visual Controls
* **Enable Heatmap:** Overlays a color-coded thermal map of anomalies.
  * **Transparency Slider:** Adjusts the heatmap's opacity (default **45%**) — how clearly you can see the original image underneath.
* **Enable Segmentation:** Generates dashed polygons outlining the anomalies, ready to accept as annotations.
  * **Threshold Slider:** Percentile of the anomaly score map used to generate polygons (0.0–100.0, default **95.0**). Lower values outline fainter anomalies; higher values outline only the most severe defects.

### Advanced Settings
Expand "Advanced Settings" within the Microsentry AI tab for finer control:
* **Simplify Tolerance (ε):** Controls how jagged the generated polygons are (default **12**). Higher values create smoother, simpler shapes.
* **Heatmap Minimum:** Percentile used to hide the coldest (lowest-scoring) areas of the heatmap entirely (default **0%**).

## 4. Converting AI Predictions to Annotations

MicroSentryAI is designed to accelerate manual annotation.
1. Adjust the **Threshold** and **Simplify Tolerance** until the dashed AI outlines accurately capture the defects.
2. Either:
   * Click **Accept AI Polygons** in the Microsentry AI tab to convert every currently displayed AI polygon at once, assigned to the dataset's first defined class, or
   * Click a single AI polygon on the canvas to accept just that one via a popup, letting you choose its class individually.
3. The accepted segments become standard AnnoMate polygons. You can now edit their vertices manually if needed.

## 5. MicroSentryAI vs. the SAM Segment Tool

MicroSentryAI (this tab) runs **batch** anomaly detection across your whole dataset up front. It's a separate AI-assisted feature from the interactive **SAM Segment** tool (shortcut `S`) in the tool palette, which segments one object at a time from a bounding box you draw and doesn't require a Microsentry model to be loaded at all. See the AnnoMate User Guide for SAM Segment usage.
