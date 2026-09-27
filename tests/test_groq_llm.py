"""Tests for Groq ChatGroq LLM Manager integration and environment variable handling."""

import os
import pytest
from unittest.mock import patch, MagicMock
from app.models.llm import GroqLLMManager, LocalLLMManager

def test_missing_groq_api_key_raises_configuration_error(monkeypatch):
    """Verify that missing GROQ_API_KEY raises a clear ValueError configuration error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    manager = GroqLLMManager()
    
    with pytest.raises(ValueError) as exc_info:
        manager.get_llm()
    
    assert "GROQ_API_KEY environment variable is not set" in str(exc_info.value)


def test_groq_llm_instantiation(monkeypatch):
    """Verify ChatGroq is instantiated with exact model openai/gpt-oss-20b and temperature=0."""
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key_12345")
    
    with patch("app.models.llm.ChatGroq") as mock_chat_groq:
        manager = GroqLLMManager()
        llm = manager.get_llm()
        
        mock_chat_groq.assert_called_once_with(
            model="openai/gpt-oss-20b",
            temperature=0
        )
        assert llm == mock_chat_groq.return_value


def test_generate_text_with_metrics_mocked(monkeypatch):
    """Verify text generation and metrics structure."""
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key_12345")
    
    manager = GroqLLMManager()
    mock_response = MagicMock()
    mock_response.content = "This is a document-grounded response."
    mock_response.response_metadata = {"token_usage": {"completion_tokens": 6}}
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    
    with patch.object(manager, "get_llm", return_value=mock_llm):
        answer, metrics = manager.generate_text_with_metrics("Test prompt", system_prompt="System prompt")
        
        assert answer == "This is a document-grounded response."
        assert "total_llm_time" in metrics
        assert metrics["tokens_generated"] == 5
        assert metrics["tokens_per_sec"] >= 0


def test_backward_compatibility_alias():
    """Verify LocalLLMManager is an alias of GroqLLMManager."""
    assert LocalLLMManager is GroqLLMManager
