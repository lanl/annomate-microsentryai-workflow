"""
Entry point — composition root only.

Creates all layers in dependency order (states → models → controllers → view)
and starts the Qt event loop. Contains no UI code.
"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from core.states.validation_state import ValidationState

from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.validation_model import ValidationModel

from controllers.io_controller import IOController
from controllers.inference_controller import InferenceController
from controllers.validation_controller import ValidationController
from controllers.project_controller import ProjectController

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
    validation_state = ValidationState()

    # Models
    dataset_model = DatasetTableModel(dataset_state)
    inference_model = InferenceModel(inference_state)
    validation_model = ValidationModel(validation_state)

    # Controllers
    io_controller = IOController(dataset_model)
    inference_controller = InferenceController(dataset_model, inference_model)
    validation_controller = ValidationController(validation_model)
    project_controller = ProjectController(
        dataset_model,
        inference_model,
        validation_model,
        io_controller,
        inference_controller,
    )

    # View
    window = AppWindow(
        dataset_model=dataset_model,
        inference_model=inference_model,
        validation_model=validation_model,
        io_controller=io_controller,
        inference_controller=inference_controller,
        validation_controller=validation_controller,
        project_controller=project_controller,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
