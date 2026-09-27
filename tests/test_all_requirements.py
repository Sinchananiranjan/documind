"""Dedicated test suite for the 8 user-requested requirement tests.

Validates:
1. Apply a document-described algorithm to new numerical inputs
2. Compare two concepts using only document evidence
3. Question about a figure/diagram
4. Question about a table
5. Absent concept
6. Document definition
7. Document explanation
8. Calculation requiring document context

No subject-specific hardcoding is used.
"""

import pytest
from unittest.mock import patch

from app.graph.workflow import run_documind_workflow


@pytest.fixture(scope="module")
def algorithm_doc_chunks():
    return [
        {
            "doc_id": "doc_algo_1",
            "filename": "algorithm_guide.pdf",
            "page_num": 2,
            "chunk_type": "text",
            "chunk_id": "algo_c1",
            "content": "Step 1: Compute product P = a * b. Step 2: Add constant C = 10. Formula: Output = (a * b) + 10.",
            "score": 0.1,
            "combined_score": 0.85
        }
    ]


@pytest.fixture(scope="module")
def comparison_doc_chunks():
    return [
        {
            "doc_id": "doc_comp_1",
            "filename": "concept_comparison.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "comp_c1",
            "content": "Concept Alpha uses synchronous processing with fixed block sizes. Concept Beta uses asynchronous processing with variable packet sizes.",
            "score": 0.1,
            "combined_score": 0.80
        }
    ]


@pytest.fixture(scope="module")
def figure_doc_chunks():
    return [
        {
            "doc_id": "doc_fig_1",
            "filename": "architecture_diagram.pdf",
            "page_num": 3,
            "chunk_type": "image",
            "chunk_id": "fig_img1",
            "content": "[Figure 3.2 System Architecture Diagram] Shows Client connected to Load Balancer, which routes traffic to three App Worker Nodes.",
            "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "score": 0.1,
            "combined_score": 0.78
        }
    ]


@pytest.fixture(scope="module")
def table_doc_chunks():
    return [
        {
            "doc_id": "doc_tbl_1",
            "filename": "metrics_table.pdf",
            "page_num": 5,
            "chunk_type": "table",
            "chunk_id": "tbl_t1",
            "content": "Performance Metrics Table:\n| System | Throughput (req/s) | Latency (ms) |\n| System A | ~5000 | 12.4 |\n| System B | ~8500 | 8.1 |",
            "score": 0.1,
            "combined_score": 0.82
        }
    ]


# ── Test 1: Apply document-described algorithm to new inputs ─────────────
def test_req_1_apply_algorithm_to_new_inputs(algorithm_doc_chunks):
    """Test 1: Apply a document-described algorithm to new numerical inputs."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=algorithm_doc_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Following the document method (Output = a * b + 10) on Page 2, for a=5 and b=4: (5 * 4) + 10 = 30 [Page 2].", {})):
        res = run_documind_workflow(
            question="Apply the document algorithm for a = 5 and b = 4 to compute Output",
            doc_id="doc_algo_1",
            mode="document_mode"
        )
        assert res["route"] in ("calculation", "text_rag")
        assert "30" in res["answer"]
        assert len(res["sources"]) > 0


# ── Test 2: Compare two concepts using only document evidence ─────────────
def test_req_2_compare_two_concepts(comparison_doc_chunks):
    """Test 2: Compare two concepts using only document evidence."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=comparison_doc_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("According to Page 1, Concept Alpha uses synchronous processing with fixed block sizes, whereas Concept Beta uses asynchronous processing with variable packet sizes [Page 1].", {})):
        res = run_documind_workflow(
            question="Compare Concept Alpha and Concept Beta based on the document",
            doc_id="doc_comp_1",
            mode="document_mode"
        )
        assert res["route"] in ("text_rag", "hybrid")
        assert "Alpha" in res["answer"]
        assert "Beta" in res["answer"]
        assert res["verified"] is True


# ── Test 3: Question about a figure/diagram ────────────────────────────────
def test_req_3_question_about_figure(figure_doc_chunks):
    """Test 3: Question about a figure/diagram combining visual & surrounding page text."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=figure_doc_chunks), \
         patch("app.models.llm.LocalLLMManager.analyze_image", return_value="Diagram depicting load balancer routing to three app worker nodes."), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Figure 3.2 shows the Client connecting to a Load Balancer, which routes to three App Worker Nodes [Page 3].", {})):
        res = run_documind_workflow(
            question="What architecture components are shown in Figure 3.2 diagram?",
            doc_id="doc_fig_1",
            mode="document_mode"
        )
        assert res["route"] == "image_analysis"
        assert "Load Balancer" in res["answer"] or "Worker Nodes" in res["answer"] or "Figure" in res["answer"]


# ── Test 4: Question about a table ─────────────────────────────────────────
def test_req_4_question_about_table(table_doc_chunks):
    """Test 4: Question about a table preserving exact table notation and values."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=table_doc_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("According to the table on Page 5, System A has a throughput of ~5000 req/s and latency of 12.4 ms [Page 5].", {})):
        res = run_documind_workflow(
            question="According to the table, what is the throughput and latency of System A?",
            doc_id="doc_tbl_1",
            mode="document_mode"
        )
        assert res["route"] == "table_analysis"
        assert "5000" in res["answer"] or "12.4" in res["answer"]


# ── Test 5: Absent concept in Document Mode ─────────────────────────────────
def test_req_5_absent_concept():
    """Test 5: Absent concept in Document Mode returns strict document fallback."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=[]):
        res = run_documind_workflow(
            question="What is the quantum entanglement protocol described in the PDF?",
            doc_id="doc_algo_1",
            mode="document_mode"
        )
        assert "not provide" in res["answer"].lower() or "couldn't find" in res["answer"].lower()
        assert res["verified"] is False


# ── Test 6: Document definition ─────────────────────────────────────────────
def test_req_6_document_definition(algorithm_doc_chunks):
    """Test 6: Document definition query grounded in context."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=algorithm_doc_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Output is defined as the result of multiplying a and b and adding constant C = 10 [Page 2].", {})):
        res = run_documind_workflow(
            question="Define Output as stated in the algorithm guide",
            doc_id="doc_algo_1",
            mode="document_mode"
        )
        assert res["route"] == "text_rag"
        assert res["verified"] is True


# ── Test 7: Document explanation ───────────────────────────────────────────
def test_req_7_document_explanation(comparison_doc_chunks):
    """Test 7: Document explanation query preserving exact steps/rationale."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=comparison_doc_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("The document explains that Concept Alpha processes data synchronously with fixed block sizes to ensure deterministic throughput [Page 1].", {})):
        res = run_documind_workflow(
            question="Explain how Concept Alpha processes data according to the document",
            doc_id="doc_comp_1",
            mode="document_mode"
        )
        assert res["route"] in ("text_rag", "hybrid")
        assert res["verified"] is True


# ── Test 8: Calculation requiring document context ────────────────────────
def test_req_8_calculation_requiring_document_context(algorithm_doc_chunks):
    """Test 8: Calculation requiring document context combines method + deterministic result."""
    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=algorithm_doc_chunks), \
         patch("app.models.llm.LocalLLMManager.generate_text_with_metrics", return_value=("Using the formula Output = (a * b) + 10 on Page 2, given a=10 and b=2: (10 * 2) + 10 = 30 [Page 2].", {})):
        res = run_documind_workflow(
            question="For a = 10 and b = 2, calculate Output = a * b + 10 using the document formula",
            doc_id="doc_algo_1",
            mode="document_mode"
        )
        assert res["route"] == "calculation"
        assert "30" in res["answer"]
        assert res["verified"] is True
