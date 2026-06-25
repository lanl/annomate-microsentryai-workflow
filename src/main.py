"""
Entry point — composition root only.

Creates all layers in dependency order (states → models → controllers → view)
and starts the Qt event loop. Contains no UI code.
"""

import os
import sys

# PyInstaller --windowed sets sys.stdout/stderr to None (no console).
# Any library that calls .write() on them (e.g. tqdm in huggingface_hub) will
# crash with 'NoneType has no attribute write'. Redirect to devnull early.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from core.states.calibration_state import CalibrationState
from core.states.center_template_state import CenterTemplateState
from core.states.anomaly_constraint_state import AnomalyConstraintState

from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.calibration_model import CalibrationModel
from models.center_template_model import CenterTemplateModel
from models.anomaly_constraint_model import AnomalyConstraintModel

from controllers.io_controller import IOController
from controllers.inference_controller import InferenceController
from controllers.project_controller import ProjectController
from controllers.center_template_controller import CenterTemplateController

from views.app_window import AppWindow

from core.utils.logger import setup_logging

setup_logging()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Force Light Theme globally across all OS platforms
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    # States
    dataset_state = DatasetState()
    inference_state = InferenceState()
    calibration_state = CalibrationState()
    center_template_state = CenterTemplateState()
    anomaly_constraint_state = AnomalyConstraintState()

    # Models
    dataset_model = DatasetTableModel(dataset_state)
    inference_model = InferenceModel(inference_state)
    calibration_model = CalibrationModel(calibration_state)
    center_template_model = CenterTemplateModel(center_template_state)
    anomaly_constraint_model = AnomalyConstraintModel(anomaly_constraint_state)

    # Controllers
    io_controller = IOController(dataset_model)
    inference_controller = InferenceController(dataset_model, inference_model)
    center_template_controller = CenterTemplateController(center_template_model)
    project_controller = ProjectController(
        dataset_model,
        inference_model,
        io_controller,
        inference_controller,
        calibration_model=calibration_model,
        center_template_model=center_template_model,
        anomaly_constraint_model=anomaly_constraint_model,
    )

    # View
    window = AppWindow(
        dataset_model=dataset_model,
        inference_model=inference_model,
        io_controller=io_controller,
        inference_controller=inference_controller,
        project_controller=project_controller,
        calibration_model=calibration_model,
        center_template_model=center_template_model,
        center_template_controller=center_template_controller,
        anomaly_constraint_model=anomaly_constraint_model,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
