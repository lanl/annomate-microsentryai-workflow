# View Tools

These tools change how you see or check an image. They never change the image file itself or your annotations.

## Grid and calibration (View Overlays tab)

Calibration tells the app how big one pixel is in real life, so measurements and checks use real units.

**Set the scale, either way:**
- Click **Click two points…**, then click two points on the image that are a known distance apart. Enter the real distance and its unit (for example `5 mm`) when asked.
- Or type a **pixels : real-world value** ratio and click **Apply**. Use **Export** and **Import** to save a ratio to a file or reuse one.

**Grid display:** turn on **Enable Grid** to overlay a reference grid. You can change its opacity, colour, and spacing (**Auto** follows your zoom level, **Fixed** uses a set real-world distance). **Reset to Defaults** removes the calibration and restores the grid settings.

Once calibrated, the Measure tool and Anomaly Constraints show real units.

## Center Crop (View Overlays tab)

Shows a shape on the image marking the region that matters, dimming everything outside it.

- **Enable Center Crop** turns it on. Choose a **Rectangle** or **Circle** and set its size.
- **Outside opacity** sets how dark the area outside the shape is. You can also change the border colour and show a **center dot**.
- To line the shape up on your part, click **Calibrate Center** and use the arrow keys (hold **Shift** for bigger steps) or drag. Click **Accept** to save that position as the matching template for the other images, or **Clear** to remove it.
- **Reset Defaults** restores the crop settings.

## Anomaly Constraints (View Overlays tab)

Automatic checks that flag annotations for you to look at again. Turn on **Enable Anomaly Constraints**, then choose the checks:

- **Area Threshold**: flags annotations larger than **Max Area**.
- **Proximity Threshold**: flags pairs of annotations closer together than **Min Dist**. Measure between **Centroid** (centres) or **Edge** (nearest edges).

Flagged annotations are outlined in the colour you choose, and a count is shown in the panel.

## Image Adjustments tab

Temporary changes that help you see faint defects. They only affect the display, and they stay on as you move between images while enabled.

- **HSV**: adjust **Hue**, **Saturation** and **Value**. **Reset** returns them to normal.
- **Brightness/Contrast**: set a black level and a white level to stretch contrast. **Reset** returns it to normal.
