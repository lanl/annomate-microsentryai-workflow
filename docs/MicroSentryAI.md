# MicroSentryAI Guide

MicroSentryAI is the integrated inference engine. It allows you to load trained anomaly detection models, run them across your dataset, and project defect heatmaps directly onto your images.

## 1. Supported Models
The system currently supports PyTorch (`.pt`) models, specifically those trained via **Anomalib** (e.g., PatchCore, EfficientAD, Padim).

## 2. Loading an AI Model
1. In the right panel, expand the **Microsentry** section.
2. Click **Load New** and select your trained model file.
3. The system will automatically detect your best hardware (NVIDIA CUDA, Apple MPS, or CPU).
4. **Batch Inference:** Upon loading, the system will process all loaded images in the background. A progress bar will appear at the bottom. Doing this upfront ensures instant, smooth navigation between images later.

*Note: If you save your `.annoproj` project, the application remembers your model path. Next time you open the project, simply click **Load Previous**.*

## 3. The AI View Overlay
Once inference is complete, enable the overlay via the top menu: **View > Enable MicroSentryAI**. A floating control panel will appear on the top right of the canvas.

### Visual Controls
* **Heatmap Toggle:** Overlays a color-coded thermal map of anomalies.
  * **Transparency Slider:** Adjusts how clearly you can see the original image underneath the heatmap.
* **Segments Toggle:** Generates green dashed polygons outlining the anomalies.
  * **Threshold Slider:** Determines how "hot" an anomaly must be to generate a polygon (0-100 Percentile). Lower values outline fainter anomalies; higher values outline only the most severe defects.

### Advanced Settings (Right Panel)
Expand "Advanced Settings" under the Microsentry section in the right panel for finer control:
* **Smoothing (σ):** Blurs the heatmap slightly to reduce pixel noise before polygons are generated.
* **Simplify Tolerance (ε):** Controls how jagged the generated polygons are. Higher values create smoother, simpler shapes.
* **Heatmap Minimum:** Hides cold (blue) areas of the heatmap entirely.

## 4. Converting AI Predictions to Annotations
MicroSentryAI is designed to accelerate manual annotation.
1. Adjust the **Threshold**, **Smoothing**, and **Tolerance** until the dashed green AI outlines accurately capture the defects.
2. Click **Accept AI Polygons** in the floating AI View panel.
3. The AI segments will instantly convert into standard AnnoMate polygons assigned to your currently active class. You can now edit their vertices manually if needed.