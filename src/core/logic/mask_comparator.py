"""
MaskComparator — pure Python domain logic.

Computes IoU / precision / recall and generates visual comparison maps
between a ground-truth binary mask and a prediction binary mask.
No Qt dependencies.
"""

import cv2
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple


class MaskComparator:
    """Compute validation metrics and visual overlap maps for two binary masks.

    Accepts a ground-truth binary mask and a prediction binary mask, then
    produces pixel-level IoU, precision, and recall metrics along with
    colour-coded comparison visualizations. Contains zero Qt dependencies.

    Attributes:
        gt_outline_color (Tuple[int, int, int]): BGR color used when drawing
            the GT contour on overlay images.
        gt_outline_thickness (int): Pixel thickness of the GT contour.
    """

    def __init__(
        self,
        gt_outline_color: Tuple[int, int, int] = (0, 0, 255),
        gt_outline_thickness: int = 2,
    ) -> None:
        """Initialize MaskComparator with visualization parameters.

        Args:
            gt_outline_color (Tuple[int, int, int]): BGR color tuple for the
                GT contour drawn on overlay images. Defaults to
                ``(0, 0, 255)`` (red in BGR).
            gt_outline_thickness (int): Pixel thickness of the GT contour.
                Defaults to ``2``.
        """
        self.gt_outline_color = gt_outline_color
        self.gt_outline_thickness = gt_outline_thickness

    def _get_centroid(
        self, mask: np.ndarray, area: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Compute the centroid of a binary mask using image moments.

        Args:
            mask (np.ndarray): Single-channel binary mask (uint8).
            area (int): Pre-computed non-zero pixel count of *mask*. Used as
                a fast early-exit check to avoid moment computation on empty
                masks.

        Returns:
            Tuple[Optional[int], Optional[int]]: ``(cx, cy)`` centroid
                coordinates as integers. Both values are ``None`` when *area*
                is zero or when the zeroth moment is zero.
        """
        if area > 0:
            M = cv2.moments(mask)
            if M["m00"] != 0:
                return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None, None

    def calculate_metrics(
        self, gt_binary: np.ndarray, pred_binary: np.ndarray
    ) -> Dict[str, Any]:
        """Compute pixel-level IoU, precision, recall, areas, and centroids.

        Args:
            gt_binary (np.ndarray): Ground-truth binary mask (uint8, single
                channel). Non-zero pixels are treated as positive.
            pred_binary (np.ndarray): Prediction binary mask with the same
                shape and dtype as *gt_binary*.

        Returns:
            Dict[str, Any]: Metrics dictionary with the following keys:

            - ``gt_area`` (int): Non-zero pixel count of the GT mask.
            - ``pred_area`` (int): Non-zero pixel count of the prediction mask.
            - ``overlap_area`` (int): Pixel count of the intersection.
            - ``union_area`` (int): Pixel count of the union.
            - ``iou`` (float): Intersection-over-union as a percentage (0–100).
            - ``precision`` (float): Prediction precision as a percentage.
            - ``recall`` (float): GT recall as a percentage.
            - ``gt_centroid`` (Tuple[Optional[int], Optional[int]]): ``(cx, cy)``
              centroid of the GT mask, or ``(None, None)`` if empty.
            - ``pred_centroid`` (Tuple[Optional[int], Optional[int]]): ``(cx, cy)``
              centroid of the prediction mask, or ``(None, None)`` if empty.
            - ``euclidean_distance`` (Optional[float]): Pixel distance between
              centroids, or ``None`` if either mask is empty.
        """
        gt_area = cv2.countNonZero(gt_binary)
        pred_area = cv2.countNonZero(pred_binary)

        overlap_mask = cv2.bitwise_and(gt_binary, pred_binary)
        union_mask = cv2.bitwise_or(gt_binary, pred_binary)
        overlap_area = cv2.countNonZero(overlap_mask)
        union_area = cv2.countNonZero(union_mask)

        iou = (overlap_area / union_area) * 100 if union_area > 0 else 0.0
        precision = (overlap_area / pred_area) * 100 if pred_area > 0 else 0.0
        recall = (overlap_area / gt_area) * 100 if gt_area > 0 else 0.0

        cx_gt, cy_gt = self._get_centroid(gt_binary, gt_area)
        cx_pred, cy_pred = self._get_centroid(pred_binary, pred_area)

        euclidean_distance = None
        if cx_gt is not None and cx_pred is not None:
            euclidean_distance = math.sqrt(
                (cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2
            )

        return {
            "gt_area": gt_area,
            "pred_area": pred_area,
            "overlap_area": overlap_area,
            "union_area": union_area,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "gt_centroid": (cx_gt, cy_gt),
            "pred_centroid": (cx_pred, cy_pred),
            "euclidean_distance": euclidean_distance,
        }

    def generate_comparison_viz(
        self,
        gt_binary: np.ndarray,
        pred_binary: np.ndarray,
        metrics: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate colour-coded comparison and overlay visualizations.

        Args:
            gt_binary (np.ndarray): Ground-truth binary mask (uint8, single
                channel).
            pred_binary (np.ndarray): Prediction binary mask with the same
                shape and dtype as *gt_binary*.
            metrics (Dict[str, Any]): Metrics dict as returned by
                :meth:`calculate_metrics`. The ``gt_centroid`` and
                ``pred_centroid`` keys are used to draw centroid markers.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A two-element tuple:

            - ``comparison_map``: BGR image where TP pixels are white,
              FP pixels are green, and FN pixels are red.
            - ``overlay_viz``: BGR image where TP pixels are green, FP pixels
              are red, the GT contour is drawn in :attr:`gt_outline_color`,
              the GT centroid is marked blue, and the prediction centroid
              is marked cyan.
        """
        fn_mask = cv2.subtract(gt_binary, pred_binary)
        fp_mask = cv2.subtract(pred_binary, gt_binary)
        overlap_mask = cv2.bitwise_and(gt_binary, pred_binary)
        h, w = gt_binary.shape

        comparison_map = np.zeros((h, w, 3), dtype=np.uint8)
        comparison_map[fn_mask > 0] = [0, 0, 255]  # FN = red
        comparison_map[fp_mask > 0] = [0, 255, 0]  # FP = green
        comparison_map[overlap_mask > 0] = [255, 255, 255]  # TP = white

        overlay_viz = np.zeros_like(comparison_map)
        overlay_viz[fp_mask > 0] = [0, 0, 255]  # FP = red
        overlay_viz[overlap_mask > 0] = [0, 255, 0]  # TP = green

        contours, _ = cv2.findContours(
            gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay_viz,
            contours,
            -1,
            self.gt_outline_color,
            self.gt_outline_thickness,
        )

        cx_gt, cy_gt = metrics["gt_centroid"]
        cx_pred, cy_pred = metrics["pred_centroid"]
        if cx_gt is not None:
            cv2.circle(overlay_viz, (cx_gt, cy_gt), 5, (255, 0, 0), -1)
        if cx_pred is not None:
            cv2.circle(overlay_viz, (cx_pred, cy_pred), 5, (0, 255, 255), -1)

        return comparison_map, overlay_viz

    def compare_masks(
        self, gt_mask: np.ndarray, pred_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Run a full mask comparison and return all results.

        Convenience method that calls :meth:`calculate_metrics` followed by
        :meth:`generate_comparison_viz` and bundles the outputs.

        Args:
            gt_mask (np.ndarray): Ground-truth binary mask (uint8, single
                channel).
            pred_mask (np.ndarray): Prediction binary mask with the same
                shape and dtype as *gt_mask*.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: A three-element
            tuple of ``(comparison_map, overlay_viz, metrics)`` as described
            in :meth:`generate_comparison_viz` and :meth:`calculate_metrics`.
        """
        metrics = self.calculate_metrics(gt_mask, pred_mask)
        comparison_map, overlay_viz = self.generate_comparison_viz(
            gt_mask, pred_mask, metrics
        )
        return comparison_map, overlay_viz, metrics
