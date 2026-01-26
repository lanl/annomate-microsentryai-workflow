import cv2
import numpy as np
import math
from typing import Tuple, Dict, Any, Optional

class MaskComparator:
    """
    Handles the calculation of metrics and visualization generation between a 
    ground truth and a prediction mask, which must be provided as NumPy arrays.
    """
    def __init__(self, gt_outline_color: Tuple[int, int, int], gt_outline_thickness: int):
        self.gt_outline_color = gt_outline_color
        self.gt_outline_thickness = gt_outline_thickness

    def _get_centroid(self, mask: np.ndarray, area: int) -> Tuple[Optional[int], Optional[int]]:
        """Helper to compute the centroid (cx, cy) of a binary mask."""
        if area > 0:
            M = cv2.moments(mask)
            # Ensure division by zero is avoided
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
        return None, None

    def calculate_metrics(self, gt_binary: np.ndarray, pred_binary: np.ndarray) -> Dict[str, Any]:
        """Calculates IoU, precision, recall, and centroid distance."""
        
        # Calculate Areas
        gt_area = cv2.countNonZero(gt_binary)
        pred_area = cv2.countNonZero(pred_binary)

        # Calculate Intersection (Overlap) and Union
        overlap_mask = cv2.bitwise_and(gt_binary, pred_binary)
        overlap_area = cv2.countNonZero(overlap_mask)
        union_mask = cv2.bitwise_or(gt_binary, pred_binary)
        union_area = cv2.countNonZero(union_mask)

        # Calculate Metrics
        iou = (overlap_area / union_area) * 100 if union_area > 0 else 0.0
        precision = (overlap_area / pred_area) * 100 if pred_area > 0 else 0.0
        recall = (overlap_area / gt_area) * 100 if gt_area > 0 else 0.0
        
        # Calculate Centroids
        cx_gt, cy_gt = self._get_centroid(gt_binary, gt_area)
        cx_pred, cy_pred = self._get_centroid(pred_binary, pred_area)

        # Calculate Euclidean Distance
        euclidean_distance = None
        if cx_gt is not None and cx_pred is not None:
            # Note: Fixed the potential bug in the original script where cy_gt was used twice in distance calculation.
            euclidean_distance = math.sqrt((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2)

        return {
            'gt_area': gt_area,
            'pred_area': pred_area,
            'overlap_area': overlap_area,
            'union_area': union_area,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'gt_centroid': (cx_gt, cy_gt),
            'pred_centroid': (cx_pred, cy_pred),
            'euclidean_distance': euclidean_distance
        }

    def generate_comparison_viz(self, gt_binary: np.ndarray, pred_binary: np.ndarray, metrics: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates two visual comparison maps:
        1. TP/FP/FN Map: True Positive (White), False Positive (Green), False Negative (Red).
        2. Overlay Map: True Positive (Green), False Positive (Red), with GT outline and centroids.
        """
        fn_mask = cv2.subtract(gt_binary, pred_binary)
        fp_mask = cv2.subtract(pred_binary, gt_binary)
        overlap_mask = cv2.bitwise_and(gt_binary, pred_binary)
        height, width = gt_binary.shape
        
        # Map 1: TP/FP/FN (White/Green/Red)
        comparison_map = np.zeros((height, width, 3), dtype=np.uint8)
        comparison_map[fn_mask > 0] = [0, 0, 255]      # FN = Red (BGR)
        comparison_map[fp_mask > 0] = [0, 255, 0]      # FP = Green (BGR)
        comparison_map[overlap_mask > 0] = [255, 255, 255] # TP = White (BGR)

        # Map 2: Overlay (FP/TP/GT Outline/Centroids)
        overlay_viz = np.zeros_like(comparison_map)
        overlay_viz[fp_mask > 0] = [0, 0, 255]      # FP = Red (BGR)
        overlay_viz[overlap_mask > 0] = [0, 255, 0] # TP = Green (BGR)
        
        # Draw GT outline
        contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_viz, contours, -1, self.gt_outline_color, self.gt_outline_thickness)
        
        # Draw Centroids
        cx_gt, cy_gt = metrics['gt_centroid']
        cx_pred, cy_pred = metrics['pred_centroid']

        if cx_gt is not None:
            cv2.circle(overlay_viz, (cx_gt, cy_gt), 5, (255, 0, 0), -1)      # Blue GT centroid
        if cx_pred is not None:
            cv2.circle(overlay_viz, (cx_pred, cy_pred), 5, (0, 255, 255), -1) # Yellow Pred centroid

        return comparison_map, overlay_viz

    def compare_masks(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Performs the full comparison workflow for two mask NumPy arrays.
        Assumes masks are already validated for shape and loaded correctly.
        """
        metrics = self.calculate_metrics(gt_mask, pred_mask)
        comparison_map, overlay_viz = self.generate_comparison_viz(gt_mask, pred_mask, metrics)

        return comparison_map, overlay_viz, metrics