#!/usr/bin/env python3
"""Simple GUI to browse anomalib inference results.

Left side shows the selected image with an anomaly heatmap overlay. Right side
lists every image in the loaded folder with its score/label - click one to
view it. Inference runs on a background thread so the UI stays responsive.

Usage:
    python inference_gui.py [--model path/to/model.pt] [--input path/to/images_dir]

Both --model and --input are optional; use File > Load Model / Load Images
in the GUI to pick them instead. Loading a .pt model requires unpickling, so
the GUI will ask you to confirm you trust the file the first time it's needed.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1] for heatmap coloring."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def overlay_heatmap(image: Image.Image, anomaly_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a jet-colormap heatmap of the anomaly map over the original image."""
    heatmap = (cm.jet(normalize_for_display(anomaly_map))[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heatmap_image, alpha)


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    """Convert a PIL image to a QPixmap."""
    image = image.convert("RGB")
    data = image.tobytes("raw", "RGB")
    qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class InferenceResult:
    """Holds inference output for a single image."""

    def __init__(self, path: Path, score: float, label: int, anomaly_map: np.ndarray | None) -> None:
        self.path = path
        self.score = score
        self.label = label
        self.anomaly_map = anomaly_map


class InferenceWorker(QObject):
    """Runs model loading + per-image inference on a background thread."""

    progress = Signal(int, int, str)
    result_ready = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, model_path: Path, input_dir: Path) -> None:
        super().__init__()
        self.model_path = model_path
        self.input_dir = input_dir

    @Slot()
    def run(self) -> None:
        try:
            if not self.input_dir.is_dir():
                self.failed.emit(f"{self.input_dir} is not a directory")
                return

            filenames = sorted(p for p in self.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not filenames:
                self.failed.emit(f"No images found in {self.input_dir}")
                return

            inferencer = TorchInferencer(path=self.model_path, device="auto")

            for i, path in enumerate(filenames, start=1):
                self.progress.emit(i, len(filenames), path.name)
                try:
                    prediction = inferencer.predict(image=path)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed on {path}: {exc}", file=sys.stderr)
                    continue

                score = float(prediction.pred_score.item()) if prediction.pred_score is not None else float("nan")
                label = int(prediction.pred_label.item()) if prediction.pred_label is not None else -1
                anomaly_map = None
                if prediction.anomaly_map is not None:
                    anomaly_map = prediction.anomaly_map.squeeze().cpu().numpy()

                self.result_ready.emit(InferenceResult(path=path, score=score, label=label, anomaly_map=anomaly_map))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ImageLabel(QLabel):
    """QLabel that keeps the source pixmap and rescales it on resize, preserving aspect ratio."""

    def __init__(self) -> None:
        super().__init__()
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #202020;")
        self.setMinimumSize(200, 200)

    def set_source(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            self.clear()
            return
        scaled = self._source.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class InferenceGUI(QMainWindow):
    """Main window: heatmap viewer on the left, clickable image/score table on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Anomalib Inference Viewer")
        self.resize(1150, 680)

        self.model_path: Path | None = None
        self.input_dir: Path | None = None
        self.results: list[InferenceResult] = []
        self.current_index: int | None = None

        self.thread: QThread | None = None
        self.worker: InferenceWorker | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        load_model_action = QAction("Load Model...", self)
        load_model_action.triggered.connect(self._choose_model)
        load_images_action = QAction("Load Images Folder...", self)
        load_images_action.triggered.connect(self._choose_images)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(load_model_action)
        file_menu.addAction(load_images_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        toolbar_layout = QHBoxLayout()
        model_btn = QPushButton("Load Model...")
        model_btn.clicked.connect(self._choose_model)
        images_btn = QPushButton("Load Images Folder...")
        images_btn.clicked.connect(self._choose_images)
        self.paths_label = QLabel("No model or images loaded yet.")
        toolbar_layout.addWidget(model_btn)
        toolbar_layout.addWidget(images_btn)
        toolbar_layout.addWidget(self.paths_label, 1)
        root_layout.addLayout(toolbar_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.image_label = ImageLabel()
        left_layout.addWidget(self.image_label, 1)
        self.info_label = QLabel("")
        left_layout.addWidget(self.info_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Images"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Image", "Score", "Label"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_select)
        right_layout.addWidget(self.table)
        splitter.addWidget(right)
        splitter.setSizes([800, 350])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    # -- loading -------------------------------------------------------------

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select model file", "", "PyTorch model (*.pt *.pth)")
        if path:
            self.load_model(Path(path))

    def _choose_images(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder of images")
        if path:
            self.load_images(Path(path))

    def load_model(self, model_path: Path) -> None:
        self.model_path = model_path
        self._update_paths_label()
        self._maybe_run_inference()

    def load_images(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        self._update_paths_label()
        self._maybe_run_inference()

    def _ensure_trust_remote_code(self) -> bool:
        """Ask the user to confirm they trust the model file, then set the env var.

        Loading a .pt checkpoint requires unpickling, which anomalib gates behind
        TRUST_REMOTE_CODE=1. Rather than relying on the user setting that in whatever
        shell happens to launch this script, ask once per session and set it here.
        """
        if os.environ.get("TRUST_REMOTE_CODE", "0").lower() in {"1", "true"}:
            return True

        answer = QMessageBox.warning(
            self,
            "Trust this model file?",
            "Loading a .pt model requires executing arbitrary code via Python's pickle "
            "module. Only continue if you trust the source of this file.\n\n"
            "Continue loading?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.status_bar.showMessage("Cancelled: model file not trusted.")
            return False

        os.environ["TRUST_REMOTE_CODE"] = "1"
        return True

    def _update_paths_label(self) -> None:
        model_str = self.model_path.name if self.model_path else "<no model>"
        images_str = self.input_dir.name if self.input_dir else "<no images>"
        self.paths_label.setText(f"Model: {model_str}    Images: {images_str}")

    def _maybe_run_inference(self) -> None:
        if self.model_path is None or self.input_dir is None:
            return
        if not self._ensure_trust_remote_code():
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Busy", "Inference is already running, please wait.")
            return

        self.table.setRowCount(0)
        self.results = []
        self.current_index = None
        self.image_label.set_source(QPixmap())
        self.info_label.setText("")

        self.thread = QThread(self)
        self.worker = InferenceWorker(self.model_path, self.input_dir)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot(int, int, str)
    def _on_progress(self, i: int, total: int, name: str) -> None:
        self.status_bar.showMessage(f"Running inference {i}/{total}: {name}")

    @Slot(object)
    def _on_result(self, result: InferenceResult) -> None:
        self.results.append(result)
        row = self.table.rowCount()
        self.table.insertRow(row)

        label_str = "ANOMALY" if result.label == 1 else "normal"
        color = Qt.GlobalColor.red if result.label == 1 else Qt.GlobalColor.darkGreen

        name_item = QTableWidgetItem(result.path.name)
        score_item = QTableWidgetItem(f"{result.score:.3f}")
        label_item = QTableWidgetItem(label_str)
        for item in (name_item, score_item, label_item):
            item.setForeground(color)
        score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        label_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, score_item)
        self.table.setItem(row, 2, label_item)

        if row == 0:
            self.table.selectRow(0)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Inference failed", message)

    @Slot()
    def _on_finished(self) -> None:
        self.status_bar.showMessage(f"Loaded {len(self.results)} image(s).")

    # -- display ---------------------------------------------------------------

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._show_result(rows[0].row())

    def _show_result(self, index: int) -> None:
        if index < 0 or index >= len(self.results):
            return
        self.current_index = index
        result = self.results[index]

        image = Image.open(result.path).convert("RGB")
        display = overlay_heatmap(image, result.anomaly_map) if result.anomaly_map is not None else image
        self.image_label.set_source(pil_to_pixmap(display))

        label_str = "ANOMALY" if result.label == 1 else "normal"
        self.info_label.setText(f"{result.path.name}   |   score = {result.score:.4f}   |   label = {label_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple GUI viewer for anomalib inference results.")
    parser.add_argument("--model", type=Path, default=None, help="Path to trained .pt model")
    parser.add_argument("--input", type=Path, default=None, help="Directory of images to run inference on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = InferenceGUI()
    window.show()

    if args.model is not None:
        window.load_model(args.model)
    if args.input is not None:
        window.load_images(args.input)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
