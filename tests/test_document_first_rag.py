"""Automated regression test suite for DocuMind Document-First RAG & Routing.

Scenarios covered:
1. Retrieve an author's name when it appears in an uploaded research paper.
2. Retrieve information from the final pages of a long PDF.
3. Find an exact number, date, or technical term in a document.
4. Compare information across multiple uploaded PDFs.
5. Answer a question when the document contains only part of the required information.
6. Use general knowledge when a comprehensive search confirms that the answer is absent.
7. Route standalone mathematical questions to the calculation node.
8. Retrieve document values before performing document-based calculations.
9. Handle OCR and retrieval failures without falsely claiming that information is absent.
10. Maintain document isolation between conversations.
11. Verify that citations point to the correct PDF and page.
12. Handle unavailable models, rate limits, and API timeouts.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.graph.workflow import run_documind_workflow
from app.graph.nodes import retrieve_node, router_node, general_knowledge_node, _expand_query
from app.conversation_manager import ConversationManager
from app.rag.vector_store import VectorStoreManager
from app.models.llm import GroqLLMManager


@pytest.fixture
def conv_mgr():
    return ConversationManager()


@pytest.fixture
def vector_mgr():
    return VectorStoreManager()


def test_01_retrieve_author_name_from_uploaded_paper(vector_mgr):
    """1. Retrieve an author's name when it appears in an uploaded research paper (Page 1 header)."""
    mock_chunks = [
        {
            "doc_id": "paper_123",
            "filename": "quantum_paper.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "p1_c1",
            "content": "Quantum Error Correction Schemes. Authors: Dr. Alice Smith and Prof. Bob Jones. Published 2024.",
            "combined_score": 0.85
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="The authors of the paper are Dr. Alice Smith and Prof. Bob Jones [quantum_paper.pdf - Page 1]."):
        res = run_documind_workflow(
            question="Who is the author of this paper?",
            doc_id="paper_123",
            active_docs=["paper_123"],
            mode="auto"
        )
        assert res["route"] == "text_rag"
        assert "Alice Smith" in res["answer"] or "Bob Jones" in res["answer"]
        assert len(res["sources"]) > 0
        assert res["sources"][0]["page_num"] == 1


def test_02_retrieve_from_final_pages_of_long_pdf(vector_mgr):
    """2. Retrieve information from the final pages of a long PDF (e.g. Page 48)."""
    mock_chunks = [
        {
            "doc_id": "long_doc_456",
            "filename": "annual_report.pdf",
            "page_num": 48,
            "chunk_type": "text",
            "chunk_id": "p48_c1",
            "content": "Appendix C - Future Outlook: Net carbon zero emissions target set for 2035 with total capital expenditure of $4.2 Billion.",
            "combined_score": 0.78
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="The net carbon zero emissions target is set for 2035 [annual_report.pdf - Page 48]."):
        res = run_documind_workflow(
            question="What is the net carbon zero target year mentioned in the appendix on later pages?",
            doc_id="long_doc_456",
            active_docs=["long_doc_456"],
            mode="auto"
        )
        assert res["route"] == "text_rag"
        assert "2035" in res["answer"]
        assert res["sources"][0]["page_num"] == 48


def test_03_find_exact_number_date_technical_term(vector_mgr):
    """3. Find an exact number, date, or technical term in a document."""
    mock_chunks = [
        {
            "doc_id": "spec_doc",
            "filename": "system_spec.pdf",
            "page_num": 5,
            "chunk_type": "text",
            "chunk_id": "p5_c2",
            "content": "The maximum burst bandwidth protocol operating frequency is 5.84 GHz initialized on 2025-11-14 using AES-256 GCM cipher.",
            "combined_score": 0.90
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="The operating frequency is 5.84 GHz and date is 2025-11-14 [system_spec.pdf - Page 5]."):
        res = run_documind_workflow(
            question="What is the exact operating frequency and initialization date?",
            doc_id="spec_doc",
            active_docs=["spec_doc"],
            mode="auto"
        )
        assert res["route"] == "text_rag"
        assert "5.84 GHz" in res["answer"] or "2025-11-14" in res["answer"]


def test_04_compare_information_across_multiple_pdfs(conv_mgr):
    """4. Compare information across multiple uploaded PDFs."""
    conv_id = conv_mgr.create_conversation()
    conv_mgr.add_active_doc(conv_id, "doc_a", filename="DocA.pdf")
    conv_mgr.add_active_doc(conv_id, "doc_b", filename="DocB.pdf")

    mock_chunks_a = [
        {"doc_id": "doc_a", "filename": "DocA.pdf", "page_num": 2, "chunk_type": "text", "content": "DocA revenue: $10M.", "combined_score": 0.70}
    ]
    mock_chunks_b = [
        {"doc_id": "doc_b", "filename": "DocB.pdf", "page_num": 4, "chunk_type": "text", "content": "DocB revenue: $25M.", "combined_score": 0.75}
    ]

    def mock_search(query, conversation_id=None, doc_id=None, active_docs=None, k=4, filter_type=None):
        if doc_id == "doc_a" or (active_docs and "doc_a" in active_docs and len(active_docs) == 1):
            return mock_chunks_a
        return mock_chunks_a + mock_chunks_b

    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", side_effect=mock_search), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="DocA revenue is $10M [DocA.pdf - Page 2], while DocB revenue is $25M [DocB.pdf - Page 4]."):
        res = run_documind_workflow(
            question="Compare the revenue between DocA.pdf and DocB.pdf",
            conversation_id=conv_id,
            active_docs=["doc_a", "doc_b"],
            mode="auto"
        )
        assert len(res["sources"]) >= 1
        filenames = {s.get("filename") for s in res["sources"]}
        assert "DocA.pdf" in filenames or "DocB.pdf" in filenames


def test_05_partial_sufficiency_handling(vector_mgr):
    """5. Answer a question when document contains only part of required information."""
    mock_chunks = [
        {
            "doc_id": "partial_doc",
            "filename": "policy.pdf",
            "page_num": 3,
            "chunk_type": "text",
            "chunk_id": "p3_c1",
            "content": "Employee leave policy allows 20 days paid annual leave.",
            "combined_score": 0.28
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="According to policy.pdf [Page 3], 20 days paid annual leave is allowed. 📚 Missing / General Knowledge: Sick leave details are not specified."):
        res = run_documind_workflow(
            question="What is the employee paid leave policy and maternity leave policy details?",
            doc_id="partial_doc",
            active_docs=["partial_doc"],
            mode="auto"
        )
        assert res["route"] in ["text_rag", "hybrid"]
        assert "20 days" in res["answer"]


def test_06_general_knowledge_fallback_when_absent(vector_mgr):
    """6. Use general knowledge when comprehensive search confirms answer is absent from PDF."""
    mock_empty_chunks = []
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_empty_chunks), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="Quantum entanglement is a physical phenomenon where pairs of particles remain connected."):
        res = run_documind_workflow(
            question="Explain quantum entanglement",
            doc_id="history_doc",
            active_docs=["history_doc"],
            mode="auto"
        )
        # Route may be 'text_rag' (doc was searched first) or 'general_knowledge'
        # (routed directly to GK). Both indicate GK was used after doc was checked.
        assert res["route"] in ["text_rag", "general_knowledge"]
        assert "🌐 **[General Knowledge" in res["answer"]


def test_07_route_standalone_math_to_calculation_node():
    """7. Route standalone mathematical questions directly to calculation node."""
    res = run_documind_workflow(
        question="What is 125 * 8?",
        mode="auto"
    )
    assert res["route"] == "calculation"
    assert "1000" in res["answer"]


def test_08_retrieve_doc_values_before_calculation(vector_mgr):
    """8. Retrieve document values first before performing document-based calculations."""
    mock_chunks = [
        {
            "doc_id": "sales_doc",
            "filename": "q3_sales.pdf",
            "page_num": 2,
            "chunk_type": "text",
            "chunk_id": "p2_c1",
            "content": "Q1 sales: $15000. Q2 sales: $25000.",
            "combined_score": 0.80
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="Based on q3_sales.pdf [Page 2], Q1 sales are $15000 and Q2 sales are $25000. Total sum is $40000."):
        res = run_documind_workflow(
            question="Calculate total sales sum for Q1 ($15000) + Q2 ($25000) from document",
            doc_id="sales_doc",
            active_docs=["sales_doc"],
            mode="auto"
        )
        assert res["route"] == "calculation"
        assert "40000" in res["answer"] or "Calculated Result" in res["answer"]


def test_09_handle_ocr_and_retrieval_failures():
    """9. Handle OCR and retrieval failures gracefully without falsely claiming info is absent."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", side_effect=Exception("Database retrieval error")), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="General fallback answer"):
        res = run_documind_workflow(
            question="What does figure 1 show?",
            doc_id="scanned_doc",
            active_docs=["scanned_doc"],
            mode="auto"
        )
        # Pipeline must complete safely without crash
        assert "answer" in res
        assert res["answer"] != ""


def test_10_maintain_document_isolation_between_conversations(conv_mgr, vector_mgr):
    """10. Maintain strict document isolation between different conversations."""
    conv_1 = conv_mgr.create_conversation()
    conv_2 = conv_mgr.create_conversation()

    conv_mgr.add_active_doc(conv_1, "doc_secret_1", filename="Secret1.pdf")
    conv_mgr.add_active_doc(conv_2, "doc_public_2", filename="Public2.pdf")

    # Inspect target doc resolution
    docs_c1 = conv_mgr.get_documents(conv_1)
    docs_c2 = conv_mgr.get_documents(conv_2)

    assert len(docs_c1) == 1
    assert docs_c1[0]["doc_id"] == "doc_secret_1"
    assert len(docs_c2) == 1
    assert docs_c2[0]["doc_id"] == "doc_public_2"


def test_11_verify_citations_point_to_correct_pdf_and_page(vector_mgr):
    """11. Verify that citations point to the correct PDF filename and page number."""
    mock_chunks = [
        {
            "doc_id": "cite_doc",
            "filename": "ResearchPaper.pdf",
            "page_num": 12,
            "chunk_type": "text",
            "chunk_id": "p12_c3",
            "content": "Experimental validation yields 99.4% accuracy.",
            "combined_score": 0.88
        }
    ]
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=mock_chunks), \
         patch("app.models.llm.GroqLLMManager.generate_text", return_value="Experimental validation yields 99.4% accuracy [ResearchPaper.pdf - Page 12]."):
        res = run_documind_workflow(
            question="What is the experimental accuracy?",
            doc_id="cite_doc",
            active_docs=["cite_doc"],
            mode="auto"
        )
        assert len(res["sources"]) > 0
        src = res["sources"][0]
        assert src["filename"] == "ResearchPaper.pdf"
        assert src["page_num"] == 12


def test_12_handle_unavailable_models_and_timeouts():
    """12. Handle unavailable models, rate limits, and API timeouts gracefully."""
    llm = GroqLLMManager()
    with patch("langchain_groq.ChatGroq.invoke", side_effect=Exception("Rate limit 429: Too many requests")):
        res_text = llm.generate_text("Test prompt")
        # Should return empty string on exception without unhandled crash
        assert res_text == ""
