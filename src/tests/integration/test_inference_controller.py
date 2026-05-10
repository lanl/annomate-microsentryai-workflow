"""
Tests for the Inference Controller, Worker, and Strategy fallbacks.
Ensures background ML processing queues correctly and device resolution works.
"""

import pytest
import numpy as np
from unittest.mock import patch

from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from controllers.inference_controller import InferenceController, InferenceWorker
from ai_strategies.anomalib_strategy import AnomalibStrategy


class MockStrategy:
    """A lightweight mock strategy to simulate inference without PyTorch."""

    def __init__(self):
        self.model_name = "Mock Model"

    def predict(self, image_path: str):
        # Return a dummy score and a 2x2 heatmap
        return 0.85, np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)


class TestInferenceWorker:
    """Test the QThread worker responsible for running inference without blocking the UI."""

    def test_worker_emits_signals_correctly(self, qtbot):
        """Verify that the worker processes files and emits progress and result signals."""
        # Arrange
        strategy = MockStrategy()
        file_list = ["dummy1.jpg", "dummy2.jpg"]
        worker = InferenceWorker(strategy, file_list)

        # Manually collect signals to avoid pytest-qt MultiSignalBlocker KeyErrors
        emitted_results = []
        emitted_progress = []
        worker.resultReady.connect(lambda p, s: emitted_results.append((p, s)))
        worker.progress.connect(emitted_progress.append)

        # Act & Assert
        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.start()

        worker.wait()  # CRITICAL: Block until the OS thread completely exits

        assert len(emitted_results) == 2, "Should emit one result per file."
        assert len(emitted_progress) == 2, "Should emit one progress update per file."


class TestInferenceController:
    """Test the orchestration of inference tasks through the Controller."""

    @pytest.fixture
    def setup_controller(self):
        dataset_model = DatasetTableModel(DatasetState())
        inference_model = InferenceModel(InferenceState())
        controller = InferenceController(dataset_model, inference_model)
        # Inject mock strategy to prevent real model loading
        controller._strategy = MockStrategy()
        return dataset_model, inference_model, controller

    def test_start_batch_inference_queues_and_completes(self, qtbot, setup_controller):
        """Verify batch inference starts the worker and emits the batch_done signal."""
        # Arrange
        _, _, controller = setup_controller
        file_paths = ["fileA.png", "fileB.png"]

        # Act & Assert
        with qtbot.waitSignal(controller.batch_done, timeout=2000):
            controller.start_batch_inference(file_paths)

        # CRITICAL: Clean up the worker thread to prevent Qt teardown exceptions
        if controller._worker:
            controller._worker.wait()


class TestDeviceResolution:
    """Test the AnomalibStrategy device fallback logic for cross-platform ML."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_resolve_device_cuda(self, mock_cuda):
        """Verify that CUDA is selected when available."""
        # Arrange
        strategy = AnomalibStrategy()
        strategy.device = "auto"

        # Act
        resolved = strategy._resolve_device()

        # Assert
        assert resolved == "cuda", "Should resolve to CUDA when available."

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True, create=True)
    def test_resolve_device_mps(self, mock_mps, mock_cuda):
        """Verify that Apple Silicon MPS is selected when available and CUDA is not."""
        # Arrange
        strategy = AnomalibStrategy()
        strategy.device = "auto"

        # Act
        resolved = strategy._resolve_device()

        # Assert
        assert resolved == "mps", "Should resolve to MPS on Apple Silicon."

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False, create=True)
    def test_resolve_device_cpu_fallback(self, mock_mps, mock_cuda):
        """Verify CPU is the ultimate fallback."""
        # Arrange
        strategy = AnomalibStrategy()
        strategy.device = "auto"

        # Act
        resolved = strategy._resolve_device()

        # Assert
        assert resolved == "cpu", "Should fallback to CPU if no accelerators exist."
