# MicroSentryAI Guide

MicroSentryAI is the inference engine that allows you to load trained anomaly detection models and visualize defect heatmaps overlaid on your images.

## Supported Models
The system currently supports models trained via **Anomalib**, including:
* **PatchCore**
* **Reverse Distillation**
* **EfficientAD**
* **DRAEM**

*Note: Models must be exported using the engine.export() method in your training script.*

## Getting Started

1.  **Load Model Folder:** Select a .pt file. You will be prompted to select a target device (CPU, CUDA, or Auto).
2.  **Load Image Folder:** Select the directory of images you wish to inspect.
3.  **Pre-computation (Caching):**
    * *Note:* Upon loading images, the system runs a **Batch Inference** process. You will see a progress bar. This performs the heavy AI calculations upfront so that navigating between images later is instant and smooth.

## Visualization Controls

Once images are loaded, you will see a side-by-side view: **Segmentation Mask** (Left) and **Heatmap Overlay** (Right).

### Thresholding & Sensitivity
* **Percentile Threshold Slider (Bottom):** This is the most critical control. It determines which pixels in the heatmap are hot enough to be considered a "defect."
    * *Higher (e.g., 98.0):* Only the most intense anomalies are shown.
    * *Lower (e.g., 90.0):* Fainter anomalies appear, but noise may increase.

### Display Parameters
* **Display Size:** Adjusts the resolution of the viewer (default: 600px).
* **Heat α (Alpha):** Controls the transparency of the heatmap overlay.
* **Smooth (σ):** Applies Gaussian smoothing to the heatmap to reduce pixel noise.
* **Simplify ε (Epsilon):** Controls how jagged the generated polygon outlines are. Higher values create smoother, simpler shapes.

## Integration with AnnoMate

MicroSentryAI allows you to convert AI predictions into editable annotations:

1.  Adjust the **Threshold** and **Smoothing** until the green outlines on the left accurately capture the defect.
2.  You can manually increase or decrease the size of the generated outline as well as move the vertices.
3.  Click **"Send to AnnoMate"**.
4.  Switch to the **AnnoMate** tab. The AI-generated polygons will appear there, ready for manual refinement or saving.