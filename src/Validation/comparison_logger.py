import time
import argparse
from typing import Dict, Any, TextIO, Tuple

def write_log_header(
    f: TextIO, 
    args: argparse.Namespace, 
    gt_outline_bgr: Tuple[int, int, int]
):
    """Writes the standardized header information to the log file."""
    log_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    f.write(f"=================================================\n")
    f.write(f"MASK COMPARISON LOG: {log_timestamp}\n")
    f.write(f"Ground Truth Dir: {args.ground_truth_dir}\n")
    f.write(f"Prediction Dir:   {args.prediction_dir}\n")
    f.write(f"Output Dir:       {args.output_dir}\n")
    f.write(f"Outline Color:    {args.gt_outline_color} (BGR: {gt_outline_bgr})\n")
    f.write(f"Outline Thickness: {args.gt_outline_thickness}\n")
    f.write(f"=================================================\n\n")

def log_results(f: TextIO, relative_path: str, metrics: Dict[str, Any]):
    """Writes the comparison metrics for a single file to the log."""
    f.write(f"Comparison for: {relative_path}\n")
    f.write(f"  Ground Truth Area: {metrics['gt_area']} px\n")
    f.write(f"  Prediction Area:   {metrics['pred_area']} px\n")
    f.write(f"  Overlap (Intersection): {metrics['overlap_area']} px\n")
    f.write(f"  --- METRICS ---\n")
    f.write(f"  Intersection over Union (IoU): {metrics['iou']:.2f}%\n")
    f.write(f"  Precision (Overlap/Prediction):  {metrics['precision']:.2f}%\n")
    f.write(f"  Recall (Overlap/GroundTruth):    {metrics['recall']:.2f}%\n")
    
    distance = metrics['euclidean_distance']
    if distance is not None:
        f.write(f"  Centroid Distance (Euclidean): {distance:.2f} px\n\n")
    else:
        f.write(f"  Centroid Distance (Euclidean): N/A (One or both masks empty)\n\n")

def log_skip(f: TextIO, relative_path: str, reason: str):
    """Logs a skipped file with a specific reason."""
    f.write(f"SKIPPED: {reason} for file: {relative_path}\n\n")