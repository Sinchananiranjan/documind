"""Local Ollama LLM and optional Vision Model interface with fallback logic."""

import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "qwen2.5:3b")
FALLBACK_MODEL = os.getenv("FALLBACK_LLM_MODEL", "qwen2.5:1.5b")
VISION_MODEL = os.getenv("VISION_LLM_MODEL", "moondream")

class LocalLLMManager:
    """Manages invocations to local Ollama LLM and optional Vision models with graceful fallbacks."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def get_available_models(self) -> list:
        """Fetch list of locally installed Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama at {self.base_url}: {e}")
        return []

    def select_active_model(self) -> str:
        """Select the best available text LLM model (qwen2.5:3b > qwen2.5:1.5b > first available)."""
        models = self.get_available_models()
        if not models:
            logger.warning("No Ollama models detected. Using DEFAULT_MODEL setting.")
            return DEFAULT_MODEL
        
        # Exact match or prefix match check
        for m in [DEFAULT_MODEL, FALLBACK_MODEL]:
            for installed in models:
                if installed == m or installed.startswith(m):
                    return installed
        
        # Default to first available installed model
        logger.info(f"Preferred models not found. Using installed model: {models[0]}")
        return models[0]

    def has_vision_model(self) -> bool:
        """Check if moondream or another vision model is installed."""
        models = self.get_available_models()
        return any(VISION_MODEL in m for m in models)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using selected local Ollama LLM model."""
        model_name = self.select_active_model()
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for factual RAG answers
                "num_predict": 512
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                logger.error(f"Ollama generate returned status {resp.status_code}: {resp.text}")
                return ""
        except Exception as e:
            logger.error(f"Error calling Ollama LLM ({model_name}): {e}")
            return ""

    def analyze_image(self, image_b64: str, prompt: str) -> str:
        """
        Analyze image using vision model if available;
        otherwise return metadata notice for text LLM fallback.
        """
        if self.has_vision_model():
            logger.info(f"Using vision model '{VISION_MODEL}' for image query...")
            payload = {
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
            try:
                resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
            except Exception as e:
                logger.warning(f"Vision model call failed: {e}")

        # Fallback if vision model is not installed or call fails
        logger.info("Vision model unavailable. Returning OCR/metadata fallback description.")
        return ""

    def verify_groundedness(self, question: str, context: str, answer: str) -> bool:
        """
        Check whether the answer is strictly derived from context.
        Returns True if answer is grounded, False if hallucinated or missing info.
        """
        if "I couldn't find this in the document" in answer:
            return True
            
        verify_prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Proposed Answer:\n{answer}\n\n"
            "Task: Is the Proposed Answer directly supported and factual based ONLY on the Context provided?\n"
            "Respond ONLY with 'YES' or 'NO'."
        )
        
        verdict = self.generate_text(verify_prompt)
        return "YES" in verdict.upper()
