"""Embeddings generator using Sentence Transformers (local, lightweight)."""

import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Default model is lightweight, CPU-friendly all-MiniLM-L6-v2
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

_embedding_instance = None

def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Get or initialize a cached local HuggingFace embeddings instance.
    Runs locally on CPU with small footprint (~90MB).
    """
    global _embedding_instance
    if _embedding_instance is None:
        logger.info(f"Loading embedding model '{model_name}' on CPU...")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_instance
