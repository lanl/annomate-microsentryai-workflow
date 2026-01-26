"""
Export Utilities for Annotation Data.

This module provides functions to export annotation data into standard formats,
specifically COCO JSON and a custom "Polygon + Image" format for visualization.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Third-party imports (Pillow)
from PIL import Image, ImageDraw


def export_coco_json(
    out_path: str,
    images_meta: List[Dict[str, Any]],
    annotations: Dict[str, List[Dict[str, Any]]],
    categories: Dict[str, int],
) -> str:
    """
    Exports annotations to the standard COCO JSON format.

    The exported JSON includes:
    -   Images metadata (filenames, IDs).
    -   Annotations (bounding boxes, segmentation polygons, area).
    -   Categories (class names mapped to IDs).

    Args:
        out_path (str): The destination file path for the JSON output.
        images_meta (List[Dict[str, Any]]): List of image metadata dictionaries
                                            (must contain 'id' and 'file_name').
        annotations (Dict[str, List[Dict[str, Any]]]): Dictionary mapping filenames
                                                        to lists of annotation dicts.
        categories (Dict[str, int]): Dictionary mapping class names to integer IDs.

    Returns:
        str: The absolute path to the saved JSON file.
    """
    coco_data = {
        "images": images_meta,
        "annotations": [],
        "categories": [
            {"id": cid, "name": name} for name, cid in categories.items()
        ],
    }

    # Helper map to find image ID by filename
    name_to_id = {im['file_name']: im['id'] for im in images_meta}

    ann_id = 1
    for img_name, anns_list in annotations.items():
        image_id = name_to_id.get(img_name)
        if image_id is None:
            continue

        for ann in anns_list:
            # Flatten polygon points: [[x,y], [x,y]] -> [x, y, x, y]
            seg = ann.get("polygon", [])
            seg_flat = [coord for point in seg for coord in point]

            if not seg:
                continue

            # Calculate Bounding Box
            xs = [p[0] for p in seg]
            ys = [p[1] for p in seg]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            width = x_max - x_min
            height = y_max - y_min
            area = width * height

            # COCO bbox format: [x_min, y_min, width, height]
            bbox = [float(x_min), float(y_min), float(width), float(height)]

            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": categories.get(ann["category_name"], 0),
                "segmentation": [seg_flat],
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
            })
            ann_id += 1

    # Ensure parent directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    return out_path


def export_polygons_images(
    out_dir: str,
    image_dir: str,
    annotations: Dict[str, List[Dict[str, Any]]],
    class_colors: Dict[str, Tuple[int, int, int]],
    line_width: int = 3,
) -> str:
    """
    Exports images with polygon overlays drawn directly onto them.

    This function creates a copy of each annotated image, draws the polygons
    (outline only) using the specified class colors, and saves the result
    to the output directory. It also generates a `polygons.json` summary.

    Args:
        out_dir (str): The directory to save the visualized images.
        image_dir (str): The source directory containing original images.
        annotations (Dict[str, List[Dict[str, Any]]]): Map of filename -> annotations.
        class_colors (Dict[str, Tuple[int, int, int]]): Map of class name -> (R, G, B).
        line_width (int, optional): Thickness of the polygon lines. Defaults to 3.

    Returns:
        str: The path to the output directory.
    """
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_polys_summary = {}

    for img_name, anns_list in annotations.items():
        src_file = Path(image_dir) / img_name
        if not src_file.exists():
            continue

        try:
            # Open and convert to RGB to ensure drawing compatibility
            with Image.open(src_file) as im:
                im = im.convert("RGB")
                draw = ImageDraw.Draw(im)
                items_summary = []

                for ann in anns_list:
                    poly = ann.get("polygon", [])
                    if not poly:
                        continue

                    # Get color or default to white
                    rgb = class_colors.get(ann["category_name"], (255, 255, 255))
                    
                    # Convert points to flat list of tuples
                    pts = [(float(x), float(y)) for (x, y) in poly]

                    if len(pts) >= 2:
                        # Draw closed loop: pts + [pts[0]]
                        draw.line(pts + [pts[0]], fill=tuple(rgb), width=line_width)

                    items_summary.append({
                        "class": ann["category_name"],
                        "polygon": pts
                    })

                # Save the visual result
                out_name = f"{Path(img_name).stem}_poly.jpg"
                im.save(output_path / out_name, "JPEG", quality=95)
                
                all_polys_summary[img_name] = items_summary

        except Exception as e:
            print(f"Warning: Failed to export image {img_name}: {e}")
            continue

    # Save summary JSON
    with open(output_path / "polygons.json", "w", encoding="utf-8") as f:
        json.dump(all_polys_summary, f, indent=2)

    return str(output_path)