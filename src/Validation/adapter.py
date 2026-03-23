"""
Adapter Module for the Validation UI.

This module provides the GUI for generating ground-truth masks from JSON 
annotations and running automated IoU evaluations against model predictions.
"""

import os
import glob
import cv2
import numpy as np
import json
import re

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QProgressBar, QGroupBox, QScrollArea, QFrame
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap

# Relative imports
from .mask_comparator import MaskComparator
from .comparison_logger import write_log_header, log_results, log_skip


def get_robust_id(filename: str) -> str:
    """
    Extracts an identifier to match JSON keys and tie GT masks to predictions.
    Supports formats like:
    - '118_images_003_01-25-26-20-43-41_poly.jpg' -> '118_003'
    - 'hole_003_02-16-26-01-41-33_poly.jpg' -> '003'
    - '003_binary_mask.png' -> '003'
    """
    # 1. Match 'Tray_images_Index' format
    match = re.search(r'(\d+)_images_(\d+)', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    
    # 2. Match class_index_timestamp format (e.g., hole_003_...)
    match = re.search(r'_(\d{3,})_', filename)
    if match:
        return match.group(1)
        
    # 3. Fallback: extract 3+ digit numbers
    fallback = re.findall(r'(\d{3,})', filename)
    if len(fallback) >= 2:
        return f"{fallback[0]}_{fallback[1]}"
    elif len(fallback) == 1:
        return fallback[0]
        
    # 4. Catch-all: grab the very first number found
    first_num = re.search(r'(\d+)', filename)
    if first_num:
        return first_num.group(1)
        
    return os.path.splitext(filename)[0]

# =========================================================================
# WORKERS
# =========================================================================

class MaskGenWorker(QThread):
    """Background worker to parse JSON annotations and generate binary masks."""
    progress = Signal(int)
    log_message = Signal(str)
    finished = Signal()

    def __init__(self, input_dir: str, json_path: str, output_dir: str):
        super().__init__()
        self.input_dir = input_dir
        self.json_path = json_path
        self.output_dir = output_dir

    def run(self):
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            try:
                with open(self.json_path, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    image_data_map = data.get('images', {}) 
                    if not image_data_map and isinstance(data, dict):
                         image_data_map = data.get('_via_img_metadata', data)
            except Exception as e:
                self.log_message.emit(f"Critical Error loading JSON: {e}")
                return

            files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                files.extend(glob.glob(os.path.join(self.input_dir, ext)))
            
            total = len(files)
            if total == 0:
                self.log_message.emit("Error: No images found in input folder.")
                return

            processed_count = 0
            for i, filepath in enumerate(files):
                try:
                    filename = os.path.basename(filepath)
                    image_id = get_robust_id(filename)

                    # Try an exact file match first (e.g., "003" -> "003.png")
                    json_key = next((k for k in image_data_map.keys() if k == f"{image_id}.png" or k == f"{image_id}.jpg"), None)
                    
                    # Fallback to substring search
                    if not json_key:
                        json_key = next((k for k in image_data_map.keys() if image_id in k), None)

                    if not json_key and '_' in image_id:
                        simple_index = image_id.split('_')[-1] 
                        json_key = next((k for k in image_data_map.keys() if k.startswith(f"{simple_index}.")), None)

                    if json_key:
                        img = cv2.imread(filepath)
                        if img is None: 
                            continue
                        
                        h, w = img.shape[:2]
                        final_mask = np.zeros((h, w), dtype=np.uint8)

                        entry = image_data_map[json_key]
                        annotations = entry.get('annotations', [])
                        if isinstance(annotations, dict): 
                            annotations = annotations.values()

                        drawn_polys = 0
                        for ann in annotations:
                            poly_points = ann.get('polygon')
                            if not poly_points and 'shape_attributes' in ann:
                                sa = ann['shape_attributes']
                                if sa.get('name') == 'polygon':
                                    poly_points = list(zip(sa.get('all_points_x', []), sa.get('all_points_y', [])))

                            if poly_points:
                                pts = np.array(poly_points, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(final_mask, [pts], 255)
                                drawn_polys += 1
                        
                        if drawn_polys > 0:
                            clean_name = f"{image_id}_binary_mask.png"
                            out_path = os.path.join(self.output_dir, clean_name)
                            cv2.imwrite(out_path, final_mask)
                            self.log_message.emit(f"✓ Matched {filename} -> Mask Saved")
                            processed_count += 1
                    else:
                        self.log_message.emit(f"Warning: ID {image_id} not found in JSON.")
                except Exception as e:
                    self.log_message.emit(f"Error processing {filename}: {e}")

                self.progress.emit(int((i + 1) / total * 100))

            self.log_message.emit(f"Generation Complete. Processed {processed_count}/{total}.")
            
        finally:
            self.finished.emit()


class EvaluationWorker(QThread):
    """Background worker to calculate IoU between ground truth and predictions."""
    progress = Signal(int)
    log_message = Signal(str)       
    match_found = Signal(str, str, float)
    finished = Signal()

    def __init__(self, gt_dir: str, pred_dir: str, out_dir: str):
        super().__init__()
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.out_dir = out_dir

    def run(self):
        try:
            outline_color = (0, 0, 255)
            thickness = 2
            comparator = MaskComparator(gt_outline_color=outline_color, gt_outline_thickness=thickness)
            os.makedirs(self.out_dir, exist_ok=True)
            
            # Initialize Logger
            log_path = os.path.join(self.out_dir, "evaluation_log.txt")
            log_file = open(log_path, "w", encoding="utf-8")
            write_log_header(log_file, self.gt_dir, self.pred_dir, self.out_dir, outline_color, thickness)
            
            valid_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            gt_files = []
            pred_files_list = []
            for ext in valid_exts:
                gt_files.extend(glob.glob(os.path.join(self.gt_dir, ext)))
                pred_files_list.extend(glob.glob(os.path.join(self.pred_dir, ext)))

            total = len(gt_files)
            if total == 0:
                self.log_message.emit("Error: No images found in Ground Truth folder.")
                log_file.write("ERROR: No Ground Truth images found.\n")
                log_file.close()
                return

            pred_map = {get_robust_id(os.path.basename(p)): p for p in pred_files_list}

            for i, gt_path in enumerate(sorted(gt_files)):
                gt_filename = os.path.basename(gt_path)
                gt_id = get_robust_id(gt_filename)
                
                if gt_id in pred_map:
                    pred_path = pred_map[gt_id]
                    try:
                        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                        if gt is None or pred is None: 
                            continue

                        _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
                        _, pred = cv2.threshold(pred, 1, 255, cv2.THRESH_BINARY)

                        if gt.shape != pred.shape:
                            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

                        _, overlay, metrics = comparator.compare_masks(gt, pred)
                        
                        out_name = f"eval_{gt_id}.png"
                        out_full_path = os.path.join(self.out_dir, out_name)
                        cv2.imwrite(out_full_path, overlay)
                        
                        iou = metrics['iou']
                        msg = f"Tray_Image: {gt_id} | IoU: {iou:.1f}%"
                        
                        self.log_message.emit(f"✓ Match Found: {msg}")
                        self.match_found.emit(out_full_path, msg, iou)
                        
                        # Log to file
                        log_results(log_file, gt_id, metrics)
                        
                    except Exception as e:
                        self.log_message.emit(f"Error evaluating {gt_id}: {e}")
                        log_skip(log_file, gt_id, f"Processing Error: {e}")
                else:
                    self.log_message.emit(f"⚠ Skip {gt_id}: No matching prediction found.")
                    log_skip(log_file, gt_id, "No prediction match found")
                
                self.progress.emit(int((i + 1) / total * 100))

            log_file.close()
            self.log_message.emit(f"Evaluation complete. Log saved to: {log_path}")
            
        finally:
            self.finished.emit()


# =========================================================================
# UI COMPONENT
# =========================================================================

class ValidationTab(QWidget):
    """
    Main interface for the validation workflow.
    Provides controls for mask generation and prediction evaluation.
    """
    
    def __init__(self):
        super().__init__()
        self.poly_path = ""
        self.json_path = ""
        self.mask_out_path = ""
        self.gt_path = ""
        self.pred_path = ""
        self.eval_out_path = os.path.join(os.getcwd(), "evaluation_results")
        
        # Thread handles
        self.gen_worker = None
        self.eval_worker = None
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Step 1: Generation
        grp_gen = QGroupBox("Step 1: Generate Ground Truth Masks from JSON")
        gen_layout = QVBoxLayout()
        gen_layout.addLayout(self.create_row("Select Images:", self.select_poly, "lbl_poly"))
        gen_layout.addLayout(self.create_row("Select JSON:", self.select_json, "lbl_json"))
        
        # Note: Selecting mask output automatically sets the Ground Truth path for Step 2
        gen_layout.addLayout(self.create_row("Mask Output:", self.select_mask_out, "lbl_mask_out"))
        
        self.btn_gen = QPushButton("Generate Binary Masks")
        self.btn_gen.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; height: 35px;")
        self.btn_gen.clicked.connect(self.run_generation)
        gen_layout.addWidget(self.btn_gen)
        grp_gen.setLayout(gen_layout)
        layout.addWidget(grp_gen)

        # Step 2: Evaluation
        grp_eval = QGroupBox("Step 2: Run Evaluation")
        eval_layout = QVBoxLayout()
        eval_layout.addLayout(self.create_row("Select GT Masks:", self.select_gt, "lbl_gt"))
        eval_layout.addLayout(self.create_row("Select Predictions:", self.select_pred, "lbl_pred"))
        
        self.btn_run = QPushButton("Run Comparison")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 35px;")
        self.btn_run.clicked.connect(self.run_evaluation)
        eval_layout.addWidget(self.btn_run)
        grp_eval.setLayout(eval_layout)
        layout.addWidget(grp_eval)

        # Progress & Logging
        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)

        lbl_results = QLabel("Evaluation Feed:")
        layout.addWidget(lbl_results)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.results_container)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def create_row(self, label_text: str, callback, attr_name: str) -> QHBoxLayout:
        """Helper to create a standardized file-selection row."""
        row = QHBoxLayout()
        btn = QPushButton(label_text)
        btn.setFixedWidth(140)
        btn.clicked.connect(callback)
        
        lbl = QLabel("Not selected")
        lbl.setStyleSheet("color: gray;")
        lbl.setWordWrap(True)  # Prevent long paths from pushing UI off-screen
        
        setattr(self, attr_name, lbl)
        
        row.addWidget(btn)
        row.addWidget(lbl, 1) # Give label stretch priority
        return row

    def add_log_text(self, text: str):
        """Appends a text message to the evaluation feed."""
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #333; font-family: monospace;")
        self.results_layout.addWidget(lbl)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def add_result_card(self, image_path: str, text: str, iou: float):
        """Appends a visual scorecard to the evaluation feed."""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        color = "#4CAF50" if iou > 50 else "#F44336"
        card.setStyleSheet(f"background-color: white; border: 2px solid {color}; border-radius: 5px; margin-bottom: 10px;")
        
        l = QVBoxLayout(card)
        t = QLabel(text)
        t.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")
        l.addWidget(t)
        
        img_lbl = QLabel()
        pix = QPixmap(image_path)
        if not pix.isNull():
            img_lbl.setPixmap(pix.scaledToWidth(400, Qt.SmoothTransformation))
        l.addWidget(img_lbl)
        
        self.results_layout.addWidget(card)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def clear_results(self):
        """Empties the evaluation feed."""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget(): 
                item.widget().deleteLater()

    # --- Button Callbacks ---
    
    def select_poly(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p: 
            self.poly_path = p
            self.lbl_poly.setText(p)

    def select_json(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select JSON", "", "JSON (*.json)")
        if p: 
            self.json_path = p
            self.lbl_json.setText(p)

    def select_mask_out(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p: 
            self.mask_out_path = p
            self.lbl_mask_out.setText(p)
            # Convenience: Auto-populate Step 2 GT input
            self.gt_path = p
            self.lbl_gt.setText(p)

    def select_gt(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p: 
            self.gt_path = p
            self.lbl_gt.setText(p)

    def select_pred(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p: 
            self.pred_path = p
            self.lbl_pred.setText(p)

    def run_generation(self):
        if not all([self.poly_path, self.mask_out_path, self.json_path]): 
            return
            
        self.clear_results()
        self.set_ui_state(False)
        
        self.gen_worker = MaskGenWorker(self.poly_path, self.json_path, self.mask_out_path)
        self.gen_worker.progress.connect(self.pbar.setValue)
        self.gen_worker.log_message.connect(self.add_log_text)
        self.gen_worker.finished.connect(lambda: self.set_ui_state(True))
        self.gen_worker.start()

    def run_evaluation(self):
        if not all([self.gt_path, self.pred_path]): 
            return
            
        self.clear_results()
        self.set_ui_state(False)
        
        self.eval_worker = EvaluationWorker(self.gt_path, self.pred_path, self.eval_out_path)
        self.eval_worker.progress.connect(self.pbar.setValue)
        self.eval_worker.log_message.connect(self.add_log_text)
        self.eval_worker.match_found.connect(self.add_result_card)
        self.eval_worker.finished.connect(lambda: self.set_ui_state(True))
        self.eval_worker.start()

    def set_ui_state(self, enabled: bool):
        """Toggles the state of primary action buttons to prevent double-clicks."""
        self.btn_gen.setEnabled(enabled)
        self.btn_run.setEnabled(enabled)