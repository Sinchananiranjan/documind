"""LLM Manager supporting multiple configurable providers (Groq, Google Gemini API, Ollama) with temperature=0 and performance metrics."""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
import requests

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env file for local development
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")


class GroqLLMManager:
    """Domain-agnostic LLM Manager supporting Groq, Google Gemini, and Ollama with fallback safety."""

    def __init__(self, model_name: str = DEFAULT_GROQ_MODEL):
        self.model_name = model_name
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
        self._llm: Optional[Any] = None

    def _validate_api_key(self) -> str:
        """Validate key depending on active provider."""
        if self.provider in ["google", "gemini"]:
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key or not key.strip():
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            return key.strip()
        elif self.provider == "ollama":
            return "local_ollama"
        else:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or not api_key.strip():
                raise ValueError(
                    "GROQ_API_KEY environment variable is not set. "
                    "Please configure GROQ_API_KEY in your .env file or environment variables."
                )
            return api_key.strip()

    def get_llm(self) -> Any:
        """Instantiate or return cached ChatGroq instance for Groq provider."""
        if self._llm is None:
            self._validate_api_key()
            if self.provider == "groq":
                model_to_use = os.getenv("GROQ_MODEL", self.model_name)
                self._llm = ChatGroq(
                    model=model_to_use,
                    temperature=0
                )
        return self._llm

    def check_connection(self) -> bool:
        """Check if configured provider credentials are present."""
        try:
            key = self._validate_api_key()
            return bool(key and key != "your_key_here")
        except ValueError:
            return False

    def get_available_models(self) -> List[str]:
        """Return list of supported models for active provider."""
        if self.provider in ["google", "gemini"]:
            return [DEFAULT_GEMINI_MODEL]
        elif self.provider == "ollama":
            return [DEFAULT_OLLAMA_MODEL]
        return [self.model_name]

    def select_active_model(self) -> str:
        """Return active model name."""
        if self.provider in ["google", "gemini"]:
            return DEFAULT_GEMINI_MODEL
        elif self.provider == "ollama":
            return DEFAULT_OLLAMA_MODEL
        return self.model_name

    def get_provider_name(self) -> str:
        """Return friendly provider name."""
        if self.provider in ["google", "gemini"]:
            return "Google AI Studio Gemini API"
        elif self.provider == "ollama":
            return "Ollama Local Model"
        return "Groq Cloud API"

    def has_vision_model(self) -> bool:
        """Vision analysis falls back to OCR pipeline."""
        return False

    def analyze_image(self, image_b64: str, prompt: str) -> str:
        """Vision analysis fallback for image processing."""
        return ""

    def _generate_text_google(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 45
    ) -> str:
        """Call Google Gemini Flash API via REST interface."""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is not set.")
            return ""
        
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instruction: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will strictly follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.0
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            logger.error(f"Google Gemini API error ({resp.status_code}): {resp.text}")
            return ""
        except Exception as e:
            logger.error(f"Failed to call Google Gemini API: {e}")
            return ""

    def _generate_text_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 45
    ) -> str:
        """Call local Ollama server via REST API."""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": False, "options": {"temperature": 0.0}},
                timeout=timeout
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            logger.error(f"Ollama API error ({resp.status_code}): {resp.text}")
            return ""
        except Exception as e:
            logger.error(f"Failed to call Ollama server at {base_url}: {e}")
            return ""

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        num_predict: int = 256,
        timeout: int = 45
    ) -> str:
        """Generate text using active provider with system and human messages."""
        if self.provider in ["google", "gemini"]:
            res = self._generate_text_google(prompt, system_prompt=system_prompt, timeout=timeout)
            if res:
                return res
            # Fallback to Groq if configured
            if os.getenv("GROQ_API_KEY"):
                logger.info("Falling back from Gemini to Groq API...")
                self.provider = "groq"
                return self.generate_text(prompt, system_prompt=system_prompt, num_predict=num_predict, timeout=timeout)
            return res

        elif self.provider == "ollama":
            res = self._generate_text_ollama(prompt, system_prompt=system_prompt, timeout=timeout)
            if res:
                return res
            if os.getenv("GROQ_API_KEY"):
                logger.info("Falling back from Ollama to Groq API...")
                self.provider = "groq"
                return self.generate_text(prompt, system_prompt=system_prompt, num_predict=num_predict, timeout=timeout)
            return res

        # Default Groq provider
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
        Generate text using temperature=0 and performance metrics.
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

