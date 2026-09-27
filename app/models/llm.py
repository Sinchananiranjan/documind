"""Groq LLM Manager interface replacing local Ollama generation layer with ChatGroq."""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env file for local development
load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-20b"


class GroqLLMManager:
    """Domain-agnostic LLM Manager interfacing with Groq API via ChatGroq."""

    def __init__(self, model_name: str = GROQ_MODEL):
        self.model_name = model_name
        self._llm: Optional[ChatGroq] = None

    def _validate_api_key(self) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please configure GROQ_API_KEY in your .env file or environment variables."
            )
        return api_key.strip()

    def get_llm(self) -> ChatGroq:
        """Instantiate or return cached ChatGroq instance."""
        if self._llm is None:
            self._validate_api_key()
            self._llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0
            )
        return self._llm

    def check_connection(self) -> bool:
        """Check if GROQ_API_KEY is configured."""
        try:
            key = self._validate_api_key()
            return bool(key and key != "your_key_here")
        except ValueError:
            return False

    def get_available_models(self) -> List[str]:
        """Return configured Groq model."""
        return [self.model_name]

    def select_active_model(self) -> str:
        """Return active model name."""
        return self.model_name

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "Groq Cloud API"

    def has_vision_model(self) -> bool:
        """Vision analysis falls back to OCR pipeline."""
        return False

    def analyze_image(self, image_b64: str, prompt: str) -> str:
        """Vision analysis fallback for image processing."""
        return ""

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        num_predict: int = 256,
        timeout: int = 45
    ) -> str:
        """Generate text using ChatGroq with system and human messages."""
        try:
            llm = self.get_llm()
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = llm.invoke(messages)
            if hasattr(response, "content"):
                return str(response.content).strip()
            return str(response).strip()
        except Exception as e:
            logger.error(f"Error calling ChatGroq ({self.model_name}): {e}")
            return ""

    def generate_text_with_metrics(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        num_predict: int = 256,
        timeout: int = 45
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text using ChatGroq with temperature=0 and performance metrics.
        Delegates to generate_text so both generate_text and generate_text_with_metrics
        can be easily mocked or invoked.
        """
        t0 = time.perf_counter()
        response_text = self.generate_text(
            prompt, system_prompt=system_prompt, num_predict=num_predict, timeout=timeout
        )
        total_llm_time = round(time.perf_counter() - t0, 3)

        completion_tokens = len(response_text.split()) if response_text else 0
        tps = round(completion_tokens / total_llm_time, 2) if (total_llm_time > 0 and completion_tokens > 0) else 0.0

        metrics = {
            "total_llm_time": total_llm_time,
            "prompt_eval_time": 0.0,
            "gen_time": total_llm_time,
            "tokens_generated": completion_tokens,
            "tokens_per_sec": tps
        }
        return response_text, metrics


# Class alias for backward compatibility across imports and mocks
LocalLLMManager = GroqLLMManager
