"""
Comparison Logger — pure Python logging utilities.

Text-based logging for MaskComparator results. No Qt dependencies.
"""

import time
from typing import Dict, Any, TextIO, Tuple


def write_log_header(
    f: TextIO,
    gt_dir: str,
    pred_dir: str,
    out_dir: str,
    gt_outline_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Write a formatted header block to the comparison log file.

    Args:
        f (TextIO): Open, writable file-like object to write to.
        gt_dir (str): Path to the ground truth mask directory.
        pred_dir (str): Path to the prediction mask directory.
        out_dir (str): Path to the output directory for results.
        gt_outline_bgr (Tuple[int, int, int]): BGR color tuple used for the
            GT contour drawn on overlay images.
        thickness (int): Pixel thickness of the GT contour.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f.write("=================================================\n")
    f.write(f"MASK COMPARISON LOG: {timestamp}\n")
    f.write(f"Ground Truth Dir:  {gt_dir}\n")
    f.write(f"Prediction Dir:    {pred_dir}\n")
    f.write(f"Output Dir:        {out_dir}\n")
    f.write(f"Outline Color:     BGR {gt_outline_bgr}\n")
    f.write(f"Outline Thickness: {thickness}\n")
    f.write("=================================================\n\n")


def log_results(f: TextIO, relative_path: str, metrics: Dict[str, Any]) -> None:
    """Write comparison metrics for a single mask pair to the log file.

    Args:
        f (TextIO): Open, writable file-like object to write to.
        relative_path (str): Relative file path used as the log entry header.
        metrics (Dict[str, Any]): Metrics dict as returned by
            ``MaskComparator.calculate_metrics``. Expected keys:
            ``gt_area``, ``pred_area``, ``overlap_area``, ``iou``,
            ``precision``, ``recall``, and ``euclidean_distance``.
    """
    f.write(f"Comparison for: {relative_path}\n")
    f.write(f"  Ground Truth Area:      {metrics['gt_area']} px\n")
    f.write(f"  Prediction Area:        {metrics['pred_area']} px\n")
    f.write(f"  Overlap (Intersection): {metrics['overlap_area']} px\n")
    f.write("  --- METRICS ---\n")
    f.write(f"  IoU:       {metrics['iou']:.2f}%\n")
    f.write(f"  Precision: {metrics['precision']:.2f}%\n")
    f.write(f"  Recall:    {metrics['recall']:.2f}%\n")
    dist = metrics["euclidean_distance"]
    if dist is not None:
        f.write(f"  Centroid Distance: {dist:.2f} px\n\n")
    else:
        f.write("  Centroid Distance: N/A (one or both masks empty)\n\n")


def log_skip(f: TextIO, relative_path: str, reason: str) -> None:
    """Write a skip notice for a file that was excluded from comparison.

    Args:
        f (TextIO): Open, writable file-like object to write to.
        relative_path (str): Relative path of the skipped file.
        reason (str): Human-readable explanation of why the file was skipped.
    """
    f.write(f"SKIPPED: {reason} for file: {relative_path}\n\n")
