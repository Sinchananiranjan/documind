"""Pytest configuration and global fixtures for DocuMind test suite."""

import os
import pytest

@pytest.fixture(autouse=True)
def set_mock_groq_api_key(monkeypatch):
    """Ensure GROQ_API_KEY environment variable is set for tests if not already set."""
    if "GROQ_API_KEY" not in os.environ:
        monkeypatch.setenv("GROQ_API_KEY", "mock_groq_api_key_for_testing")
