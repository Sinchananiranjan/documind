import pytest
from app.graph.nodes import conv_manager

@pytest.fixture(autouse=True)
def reset_conversation_manager():
    """Reset conv_manager in-memory state before every test to prevent cross-test leakage."""
    conv_manager.conversations = {}
    conv_manager.drafts = {}
    yield
    conv_manager.conversations = {}
    conv_manager.drafts = {}
