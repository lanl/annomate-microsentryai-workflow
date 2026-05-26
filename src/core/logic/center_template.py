import cv2
import numpy as np


def locate_center(
    image: np.ndarray,
    template: np.ndarray,
    anchor_x: int,
    anchor_y: int,
) -> tuple[int, int, float]:
    """Locate *template* in *image* and return center-like anchor coordinates.

    ``anchor_x`` and ``anchor_y`` are offsets inside the template, measured
    from the template top-left to the calibrated crop center.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    tpl_gray = (
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
    )
    result = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    cx = max_loc[0] + int(anchor_x)
    cy = max_loc[1] + int(anchor_y)
    return int(cx), int(cy), float(max_val)


def extract_template(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, int]:
    """Extract a rectangular template around a center point.

    Returns ``(template, anchor_x, anchor_y)``. If the requested crop extends
    beyond the source image, the template is clipped and the anchor is adjusted
    to the center point inside the clipped template.
    """
    h, w = image.shape[:2]
    tpl_w = max(1, int(width))
    tpl_h = max(1, int(height))
    cx = max(0, min(int(round(center_x)), w - 1))
    cy = max(0, min(int(round(center_y)), h - 1))
    x0 = cx - tpl_w // 2
    y0 = cy - tpl_h // 2
    x1 = x0 + tpl_w
    y1 = y0 + tpl_h

    clip_x0 = max(0, x0)
    clip_y0 = max(0, y0)
    clip_x1 = min(w, x1)
    clip_y1 = min(h, y1)
    anchor_x = cx - clip_x0
    anchor_y = cy - clip_y0

    return image[clip_y0:clip_y1, clip_x0:clip_x1].copy(), anchor_x, anchor_y
