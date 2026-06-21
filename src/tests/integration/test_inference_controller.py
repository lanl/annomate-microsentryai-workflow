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
from ai_strategies.onnx_strategy import OnnxStrategy


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
    """Test OnnxStrategy provider selection across device configurations."""

    @patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"])
    def test_resolve_providers_cuda(self, mock_providers):
        """Verify CUDAExecutionProvider is chosen first when available."""
        strategy = OnnxStrategy()
        strategy.set_device("auto")
        providers = strategy._resolve_providers()
        assert providers[0] == "CUDAExecutionProvider", "Should prefer CUDA when available."
        assert "CPUExecutionProvider" in providers, "CPU must always be present as fallback."

    @patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])
    def test_resolve_providers_cpu_when_no_cuda(self, mock_providers):
        """Verify CPU-only providers when CUDA is unavailable."""
        strategy = OnnxStrategy()
        strategy.set_device("auto")
        providers = strategy._resolve_providers()
        assert providers == ["CPUExecutionProvider"], "Should use CPU only when CUDA unavailable."

    def test_resolve_providers_mps_falls_back_to_cpu(self):
        """Verify MPS device request is silently downgraded to CPU (no MPS provider in onnxruntime)."""
        strategy = OnnxStrategy()
        strategy.set_device("mps")
        # set_device should have already normalised mps → cpu
        assert strategy.device == "cpu", "MPS should be remapped to CPU."
        providers = strategy._resolve_providers()
        assert "CPUExecutionProvider" in providers
        assert "CUDAExecutionProvider" not in providers
