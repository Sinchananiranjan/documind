"""Comprehensive Universality Test Suite for DocuMind.

Verifies 14 core universal capabilities across completely unrelated domains:
Biology, Finance, Physics, Electronics, History, Computer Science, Astronomy.

No hardcoded questions, keywords, subjects, or formulas are used in application code.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow
from app.tools.web_search import search_web, format_web_results_as_context


# ── Sample Multi-Domain Document Fixtures ──────────────────────────────────
@pytest.fixture(scope="module")
def sample_bio_chunks():
    return [
        {
            "doc_id": "doc_bio_101",
            "filename": "biology_notes.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "bio_c1",
            "content": "Photosynthesis is the process by which green plants and some organisms use sunlight to synthesize nutrients from carbon dioxide and water. Primary producers convert solar energy into chemical energy.",
            "score": 0.1,
            "combined_score": 0.75
        }
    ]


@pytest.fixture(scope="module")
def sample_table_chunks():
    return [
        {
            "doc_id": "doc_fin_202",
            "filename": "financial_report.pdf",
            "page_num": 4,
            "chunk_type": "table",
            "chunk_id": "fin_t1",
            "content": "Quarterly Financial Table:\n| Quarter | Revenue ($M) | Operating Expense ($M) |\n| Q1 | 45.2 | 30.1 |\n| Q2 | 52.8 | 32.5 |\n| Q3 | 61.0 | 35.0 |\n| Q4 | 70.4 | 38.2 |",
            "score": 0.1,
            "combined_score": 0.80
        }
    ]


@pytest.fixture(scope="module")
def sample_circuit_chunks():
    return [
        {
            "doc_id": "doc_elec_303",
            "filename": "circuit_diagrams.pdf",
            "page_num": 2,
            "chunk_type": "image",
            "chunk_id": "elec_img1",
            "content": "[Circuit Diagram Figure 2.1] Amplifier Circuit showing operational amplifier with feedback resistor R_f = 10k and input resistor R_in = 1k.",
            "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "score": 0.1,
            "combined_score": 0.78
        }
    ]


@pytest.fixture(scope="module")
def sample_scanned_ocr_chunks():
    return [
        {
            "doc_id": "doc_hist_404",
            "filename": "ancient_history_scanned.pdf",
            "page_num": 5,
            "chunk_type": "text",
            "chunk_id": "hist_ocr1",
            "content": "[Scanned Document OCR Text]\nThe Rosetta Stone, discovered in 1799 in Rashid (Rosetta), contains a decree issued at Memphis in 196 BC on behalf of King Ptolemy V.",
            "score": 0.1,
            "combined_score": 0.70
        }
    ]


@pytest.fixture(scope="module")
def sample_physics_chunks():
    return [
        {
            "doc_id": "doc_phys_505",
            "filename": "physics_mechanics.pdf",
            "page_num": 3,
            "chunk_type": "text",
            "chunk_id": "phys_c1",
            "content": "Newton's Second Law states that force equals mass times acceleration: F = m * a. Given mass m = 12.5 kg and acceleration a = 4.0 m/s^2.",
            "score": 0.1,
            "combined_score": 0.82
        }
    ]


# ── 1. Text Question ────────────────────────────────────────────────────────
def test_1_text_question(sample_bio_chunks):
    """Test 1: Universal text question grounded in document evidence."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_bio_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Photosynthesis is the process by which green plants convert sunlight into nutrients [Page 1].", {})):
        res = run_documind_workflow(
            question="What is photosynthesis according to the document?",
            doc_id="doc_bio_101",
            mode="document_mode"
        )
        assert res["route"] == "text_rag"
        assert "Photosynthesis" in res["answer"]
        assert len(res["sources"]) > 0
        assert res["sources"][0]["doc_id"] == "doc_bio_101"


# ── 2. Table Question ───────────────────────────────────────────────────────
def test_2_table_question(sample_table_chunks):
    """Test 2: Universal table question correctly routed to table analysis."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_table_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("According to the table on Page 4, Q3 Revenue was $61.0M.", {})):
        res = run_documind_workflow(
            question="Which values are listed in the Q3 column of the financial table?",
            doc_id="doc_fin_202",
            mode="document_mode"
        )
        assert res["route"] == "table_analysis"
        assert "Q3" in res["answer"]


# ── 3. Image / Diagram Question ─────────────────────────────────────────────
def test_3_image_diagram_question(sample_circuit_chunks):
    """Test 3: Diagram query correctly routed to image analysis."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_circuit_chunks), \
         patch("app.models.llm.LocalLLMManager.analyze_image", return_value="The diagram shows an amplifier circuit with feedback resistor Rf."), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Circuit diagram analysis showing feedback resistor.", {})):
        res = run_documind_workflow(
            question="What components are shown in the amplifier circuit diagram?",
            doc_id="doc_elec_303",
            mode="document_mode"
        )
        assert res["route"] == "image_analysis"
        assert res["verified"] is True


# ── 4. OCR / Scanned Document ───────────────────────────────────────────────
def test_4_ocr_scanned_document(sample_scanned_ocr_chunks):
    """Test 4: OCR scanned text document correctly answered and cited."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_scanned_ocr_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("The Rosetta Stone was discovered in 1799 in Rashid [Page 5].", {})):
        res = run_documind_workflow(
            question="When and where was the Rosetta Stone discovered according to the scanned text?",
            doc_id="doc_hist_404",
            mode="document_mode"
        )
        assert res["route"] in ("text_rag", "hybrid")
        assert "Rosetta" in res["answer"]


# ── 5. Calculation Question ────────────────────────────────────────────────
def test_5_calculation_question():
    """Test 5: Numerical equation calculation evaluated deterministically via safe AST engine."""
    res = run_documind_workflow(
        question="If m = 12.5 and a = 4.0, calculate F = m * a",
        mode="document_mode"
    )
    assert res["route"] == "calculation"
    assert "50" in res["answer"]
    assert res["verified"] is True


# ── 6. Multi-Part Question ──────────────────────────────────────────────────
def test_6_multi_part_question(sample_bio_chunks):
    """Test 6: Multi-part query requiring sub-query expansion and synthesis."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_bio_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Photosynthesis converts solar energy to chemical energy [Page 1]. Primary producers use this process to support ecosystems.", {})):
        res = run_documind_workflow(
            question="Explain photosynthesis and describe the role of primary producers",
            doc_id="doc_bio_101",
            mode="document_mode"
        )
        assert res["route"] in ("text_rag", "hybrid")
        assert len(res["answer"]) > 0


# ── 7. Answer Present in Document ──────────────────────────────────────────
def test_7_answer_present_in_document(sample_bio_chunks):
    """Test 7: Answer present in document → verified document grounding."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_bio_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Primary producers convert solar energy into chemical energy [Page 1].", {})):
        res = run_documind_workflow(
            question="What do primary producers do?",
            doc_id="doc_bio_101",
            mode="document_mode"
        )
        assert res["route"] == "text_rag"
        assert res["verified"] is True


# ── 8. Answer Absent from Document ──────────────────────────────────────────
def test_8_answer_absent_from_document(sample_bio_chunks):
    """Test 8: Document semantically distant / answer absent → routes to web_search."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=[]), \
         patch("app.tools.web_search.search_web", return_value=[{"title": "JWST 2026", "snippet": "JWST observed exoplanet atmospheres in 2026.", "url": "https://nasa.gov"}]), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("James Webb Space Telescope observed exoplanet atmospheres in 2026 (🌐 Web: nasa.gov).", {})):
        res = run_documind_workflow(
            question="What is the latest 2026 exoplanet discovery by James Webb Space Telescope?",
            doc_id="doc_bio_101",
            mode="auto"
        )
        assert res["route"] in ("web_search", "hybrid")


# ── 9. Document + Web Combined Question ─────────────────────────────────────
def test_9_document_plus_web_combined_question(sample_bio_chunks):
    """Test 9: Partial document evidence combined with web search evidence."""
    mock_web = [
        {
            "title": "Global Ecosystems 2026",
            "snippet": "In 2026, satellite mapping showed 30% increase in primary production measurement accuracy.",
            "url": "https://nature.example.com",
            "source": "nature.example.com"
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_bio_chunks), \
         patch("app.graph.nodes.search_web", return_value=mock_web), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Primary producers convert solar energy into chemical energy [Page 1]. In 2026, satellite mapping increased measurement accuracy (🌐 Web: nature.example.com).", {})):
        res = run_documind_workflow(
            question="Compare primary producers in biology with latest 2026 satellite ecosystem mapping.",
            doc_id="doc_bio_101",
            mode="auto"
        )
        assert res["route"] in ("web_search", "hybrid")
        assert len(res["sources"]) >= 2


# ── 10. General Knowledge Question ─────────────────────────────────────────
def test_10_general_knowledge_question():
    """Test 10: Static astronomy question in General Knowledge mode."""
    with patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Jupiter is the largest planet in our solar system.", {})):
        res = run_documind_workflow(
            question="What is the largest planet in our solar system?",
            mode="general_knowledge_mode"
        )
        assert res["route"] == "general_knowledge"
        assert res["verified"] is True


# ── 11. Multiple-Document Question ──────────────────────────────────────────
def test_11_multiple_document_question(sample_bio_chunks, sample_scanned_ocr_chunks):
    """Test 11: Multi-document query across multiple active docs in conversation."""
    combined_chunks = sample_bio_chunks + sample_scanned_ocr_chunks
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=combined_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Biology notes discuss primary producers [biology_notes.pdf, Page 1], while history notes cover the Rosetta Stone [ancient_history_scanned.pdf, Page 5].", {})):
        res = run_documind_workflow(
            question="Summarize the topics covered in both biology notes and history notes.",
            active_docs=["doc_bio_101", "doc_hist_404"],
            mode="document_mode"
        )
        assert res["route"] in ("text_rag", "hybrid")
        assert len(res["sources"]) >= 2


# ── 12. Missing-Information Question ────────────────────────────────────────
def test_12_missing_information_question():
    """Test 12: Question with no available evidence in document or web -> graceful fallback."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=[]), \
         patch("app.tools.web_search.search_web", return_value=[]), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("", {})):
        res = run_documind_workflow(
            question="What is the undisclosed secret password of hypothetical project X?",
            doc_id="nonexistent_doc",
            mode="document_mode"
        )
        assert "answer" in res
        assert isinstance(res["answer"], str)
        assert len(res["answer"]) > 0


# ── 13. Vision Failure with OCR Fallback ────────────────────────────────────
def test_13_vision_failure_ocr_fallback(sample_circuit_chunks):
    """Test 13: Vision model times out / fails -> falls back gracefully to OCR text."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_circuit_chunks), \
         patch("app.models.llm.LocalLLMManager.analyze_image", return_value=""), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("From OCR text: Amplifier Circuit showing feedback resistor Rf = 10k.", {})):
        res = run_documind_workflow(
            question="What is the feedback resistor value in the diagram?",
            doc_id="doc_elec_303",
            mode="document_mode"
        )
        assert res["route"] == "image_analysis"
        assert "Amplifier" in res["answer"] or "resistor" in res["answer"] or "10k" in res["answer"]


# ── 14. Web Failure with Graceful Fallback ──────────────────────────────────
def test_14_web_failure_graceful_fallback():
    """Test 14: Network failure or empty web search results -> graceful degradation."""
    with patch("app.tools.web_search.search_web", side_effect=Exception("Network timeout")), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Unable to retrieve external web data due to network connection.", {})):
        res = run_documind_workflow(
            question="Look up 2026 stock prices for company ABC",
            mode="general_knowledge_mode"
        )
        assert "answer" in res
        assert isinstance(res["answer"], str)
        assert len(res["answer"]) > 0
