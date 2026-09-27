"""Tests for domain-independent evidence sufficiency/answerability evaluation and web routing."""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow
from app.graph.nodes import _evaluate_evidence_sufficiency, _needs_gk_web_search


def test_evidence_sufficiency_evaluation_unit():
    """Unit test for domain-independent evidence sufficiency evaluator."""
    chunks_high_match = [
        {
            "content": "Hamming distance is the number of positions at which corresponding symbols differ. Minimum Hamming distance determines error detection capability.",
            "combined_score": 0.65,
            "page_num": 1,
            "doc_id": "doc1",
            "chunk_type": "text"
        }
    ]
    # Question with key terms present in chunk
    eval_sufficient = _evaluate_evidence_sufficiency("What is Hamming distance for error detection?", chunks_high_match)
    assert eval_sufficient["is_sufficient"] is True
    assert eval_sufficient["key_term_coverage"] >= 0.35

    chunks_low_term_coverage = [
        {
            "content": "Quantum computing uses qubits in superposition to execute quantum algorithms.",
            "combined_score": 0.50,
            "page_num": 1,
            "doc_id": "doc1",
            "chunk_type": "text"
        }
    ]
    # Question topic is quantum computing, but specific question details (2026, QuEra, record) are absent
    eval_absent = _evaluate_evidence_sufficiency("What is the 2026 qubit record achieved by QuEra?", chunks_low_term_coverage)
    assert eval_absent["is_sufficient"] is False


def test_semantically_relevant_doc_answer_absent_routes_to_web():
    """Req 1: Semantically relevant document (e.g. quantum computing) but answer absent → web search."""
    mock_chunks = [
        {
            "doc_id": "quantum_doc",
            "filename": "quantum.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "q1",
            "content": "Quantum computing is a field of computer science focusing on quantum mechanics.",
            "combined_score": 0.48
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text", return_value="Web answer"):
        res = run_documind_workflow(
            question="What is the 2026 qubit record achieved by QuEra?",
            doc_id="quantum_doc",
            mode="document_mode"
        )
        # Because key terms (2026, QuEra, record) are missing from doc, sufficiency is False → routes to web_search
        assert res["route"] == "web_search"


def test_document_clearly_answers_no_web_search():
    """Req 2: Document clearly answers → no web search (routes to text_rag or tool)."""
    mock_chunks = [
        {
            "doc_id": "tech_doc",
            "filename": "tech.pdf",
            "page_num": 3,
            "chunk_type": "text",
            "chunk_id": "t3",
            "content": "The minimum Hamming distance is 3 for single-bit error correction in block codes.",
            "combined_score": 0.72
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text", return_value="The minimum Hamming distance is 3."):
        res = run_documind_workflow(
            question="What is the minimum Hamming distance for single-bit error correction?",
            doc_id="tech_doc",
            mode="document_mode"
        )
        assert res["route"] == "text_rag"
        assert res["route"] != "web_search"


def test_current_factual_question_arbitrary_wording_routes_to_web():
    """Req 3: Current factual question with arbitrary wording in GK mode → web search."""
    # Intent evaluator unit check
    assert _needs_gk_web_search("Can you look up what the recent population of Tokyo is right now?") is True
    assert _needs_gk_web_search("Please search for the latest release date of Python 3.14") is True

    with patch("app.tools.web_search.search_web", return_value=[{"title": "Tokyo Pop", "snippet": "Tokyo population is 14 million.", "url": "https://tokyo.gov"}]), \
         patch("app.models.llm.LocalLLMManager.generate_text", return_value="The recent population of Tokyo is 14 million."):
        res = run_documind_workflow(
            question="Can you look up what the recent population of Tokyo is right now?",
            mode="general_knowledge_mode"
        )
        assert res["route"] == "web_search"


def test_document_plus_web_combined_answer():
    """Req 4: Document + web combined answer synthesizes both sources and attributes citations."""
    mock_chunks = [
        {
            "doc_id": "acme_doc",
            "filename": "acme.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "a1",
            "content": "Acme Corp was founded in 2010 as a software development enterprise.",
            "combined_score": 0.30
        }
    ]
    mock_web_results = [
        {
            "title": "Acme Corp 2026 Financials",
            "snippet": "In 2026, Acme Corp reported $50M revenue.",
            "url": "https://acme.com/financials",
            "source": "acme.com"
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.graph.nodes.search_web", return_value=mock_web_results), \
         patch("app.models.llm.LocalLLMManager.generate_text", return_value="Acme Corp was founded in 2010 [Page 1]. In 2026, Acme Corp reported $50M revenue (🌐 Web: acme.com)."):
        res = run_documind_workflow(
            question="Compare Acme Corp's founding year with its latest 2026 revenue.",
            doc_id="acme_doc",
            mode="document_mode"
        )
        assert res["route"] == "web_search"
        assert len(res["sources"]) >= 2
        
        doc_sources = [s for s in res["sources"] if s.get("doc_id") == "acme_doc"]
        web_sources = [s for s in res["sources"] if s.get("doc_id") == "web_search"]
        assert len(doc_sources) > 0
        assert len(web_sources) > 0


def test_web_failure_graceful_fallback():
    """Req 5: Web search failure (e.g., offline or network error returning []) → graceful fallback."""
    with patch("app.tools.web_search.search_web", return_value=[]), \
         patch("app.models.llm.LocalLLMManager.generate_text", return_value="I couldn't find relevant information."):
        res = run_documind_workflow(
            question="Look up latest quantum records in 2026",
            mode="general_knowledge_mode"
        )
        # Should complete execution without raising an error
        assert "answer" in res
        assert isinstance(res["answer"], str)
        assert len(res["answer"]) > 0
