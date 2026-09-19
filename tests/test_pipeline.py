"""Automated test suite for DocuMind pipeline components."""

import os
import pytest
from data.generate_sample import generate_sample_pdf
from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow

@pytest.fixture(scope="module")
def sample_pdf():
    """Fixture to generate test PDF."""
    pdf_path = "./data/test_sample_document.pdf"
    generate_sample_pdf(pdf_path)
    yield pdf_path
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


def test_pdf_processing(sample_pdf):
    """Test PDF text, table, and image extraction."""
    processor = PDFProcessor()
    res = processor.process_pdf(sample_pdf)
    
    assert "summary" in res
    assert "chunks" in res
    summary = res["summary"]
    chunks = res["chunks"]
    
    assert summary["total_pages"] == 2
    assert summary["tables_found"] >= 1
    assert len(chunks) > 0


def test_vector_store_indexing(sample_pdf):
    """Test ChromaDB chunk indexing and similarity retrieval."""
    processor = PDFProcessor()
    res = processor.process_pdf(sample_pdf)
    summary = res["summary"]
    chunks = res["chunks"]

    vector_mgr = VectorStoreManager(persist_dir="./data/test_chroma_db")
    count = vector_mgr.add_document_chunks(chunks)
    assert count > 0

    results = vector_mgr.search_similarity("Benchmark Table", doc_id=summary["doc_id"], k=3)
    assert len(results) > 0
    assert results[0]["doc_id"] == summary["doc_id"]


def test_langgraph_workflow_execution(sample_pdf):
    """Test full LangGraph state machine execution on text, table, and out-of-context questions."""
    processor = PDFProcessor()
    res = processor.process_pdf(sample_pdf)
    summary = res["summary"]
    doc_id = summary["doc_id"]

    vector_mgr = VectorStoreManager()
    vector_mgr.add_document_chunks(res["chunks"])

    # 1. Text question test
    res_text = run_documind_workflow("What OS and CPU specifications are required?", doc_id=doc_id)
    assert res_text["doc_id"] == doc_id
    assert "answer" in res_text
    assert len(res_text["sources"]) > 0

    # 2. Table question test
    res_table = run_documind_workflow("What is the average latency for Qwen 2.5 3B?", doc_id=doc_id)
    assert res_table["route"] == "table_analysis"
    assert "answer" in res_table

    # 3. Out-of-context question test (Hallucination prevention test)
    res_ooc = run_documind_workflow("Who won the 2024 FIFA World Cup?", doc_id=doc_id)
    assert "I couldn't find this in the document" in res_ooc["answer"]
