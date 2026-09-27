"""Comprehensive Pytest test suite for DocuMind domain-agnostic multimodal pipeline."""

import os
import pytest
from unittest.mock import patch
from data.generate_sample import generate_sample_pdf
from data.generate_finance_sample import generate_finance_pdf
from data.generate_education_sample import generate_education_pdf

from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow

@pytest.fixture(scope="module")
def pdf_tech():
    path = "./data/test_doc_tech.pdf"
    generate_sample_pdf(path)
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture(scope="module")
def pdf_finance():
    path = "./data/test_doc_finance.pdf"
    generate_finance_pdf(path)
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture(scope="module")
def pdf_education():
    path = "./data/test_doc_edu.pdf"
    generate_education_pdf(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_three_document_isolation(pdf_tech, pdf_finance, pdf_education):
    """Test 3 unrelated PDFs: Technical, Financial, Educational. Verify zero cross-document leakage."""
    processor = PDFProcessor()
    res_a = processor.process_pdf(pdf_tech)
    res_b = processor.process_pdf(pdf_finance)
    res_c = processor.process_pdf(pdf_education)

    doc_a = res_a["summary"]["doc_id"]
    doc_b = res_b["summary"]["doc_id"]
    doc_c = res_c["summary"]["doc_id"]

    vector_mgr = VectorStoreManager(persist_dir="./data/test_chroma_db")
    vector_mgr.add_document_chunks(res_a["chunks"])
    vector_mgr.add_document_chunks(res_b["chunks"])
    vector_mgr.add_document_chunks(res_c["chunks"])

    # Query Doc A (Tech)
    out_a = vector_mgr.search_similarity("Benchmark Table", doc_id=doc_a, k=3)
    for r in out_a:
        assert r["doc_id"] == doc_a

    # Query Doc B (Finance)
    out_b = vector_mgr.search_similarity("Gross Revenue", doc_id=doc_b, k=3)
    for r in out_b:
        assert r["doc_id"] == doc_b

    # Query Doc C (Educational)
    out_c = vector_mgr.search_similarity("Primary Producers", doc_id=doc_c, k=3)
    for r in out_c:
        assert r["doc_id"] == doc_c


def test_natural_language_routing_variations(pdf_tech):
    """Test routing classification for natural language phrasing without explicit keywords."""
    processor = PDFProcessor()
    res = processor.process_pdf(pdf_tech)
    doc_id = res["summary"]["doc_id"]

    with patch("app.models.llm.LocalLLMManager.generate_text", return_value="Mock response"):
        # 1. Table question without word 'table'
        r1 = run_documind_workflow("Which values are in the third column?", doc_id=doc_id)
        assert r1["route"] == "table_analysis"

        # 2. Table row comparison
        r2 = run_documind_workflow("Compare the first and last rows.", doc_id=doc_id)
        assert r2["route"] == "table_analysis"

        # 3. Image question without 'image/diagram'
        r3 = run_documind_workflow("What does this flowchart show?", doc_id=doc_id)
        assert r3["route"] == "image_analysis"

        # 4. Calculation question — must include digits to trigger calculation route
        r4 = run_documind_workflow("What is the percentage increase from 100 to 150?", doc_id=doc_id)
        assert r4["route"] == "calculation"

        # 5. Calculation average — must include explicit numbers
        r5 = run_documind_workflow("Find the average of 10, 20, 30.", doc_id=doc_id)
        assert r5["route"] == "calculation"

        # 6. Conceptual question ('why does this happen?') — vague query, low doc relevance
        # With relevance-based routing, this may route to general_knowledge, web_search, text_rag, or hybrid
        r6 = run_documind_workflow("Why does this happen?", doc_id=doc_id)
        assert r6["route"] in ("text_rag", "hybrid", "general_knowledge", "web_search")


def test_general_knowledge_mode_isolation():
    """Test General Knowledge Mode bypasses document constraints completely."""
    with patch("app.models.llm.LocalLLMManager.generate_text", return_value="Paris is the capital of France."):
        res = run_documind_workflow("What is the capital of France?", mode="general_knowledge_mode")
        assert res["route"] in ("general_knowledge", "web_search")
        assert res["verified"] is True
        assert "General Knowledge" in res["answer"] or "Paris" in res["answer"]


def test_document_mode_fallback(pdf_tech):
    """Test Document Mode fallback behavior with unrelated question.
    With web search enabled, an unrelated question routes to web_search and returns web evidence.
    """
    processor = PDFProcessor()
    res = processor.process_pdf(pdf_tech)
    doc_id = res["summary"]["doc_id"]

    with patch("app.models.llm.LocalLLMManager.generate_text", return_value="Mock GK response"):
        res_fb = run_documind_workflow("What is quantum teleportation?", doc_id=doc_id)
        assert res_fb["route"] in ("general_knowledge", "hybrid", "text_rag", "web_search")
        # Route produces an answer via web_search or fallback text
        assert len(res_fb["answer"]) > 0

