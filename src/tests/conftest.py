"""Shared pytest fixtures for the test suite.

This module is automatically loaded by pytest before any tests run. Fixtures
defined here are available to all tests across the unit, model, and integration
sub-packages without needing an explicit import.
"""

import pytest
from core.states.dataset_state import DatasetState


@pytest.fixture
def state():
    """Provide a fresh, empty DatasetState for each test.

    Returns a new DatasetState instance with no images, annotations, or classes
    loaded. Tests that receive this fixture get an isolated state object so that
    mutations in one test cannot affect another.
    """
    return DatasetState()
