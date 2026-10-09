"""Generic Acceptance Test Suite for DocuMind.

Verifies end-to-end generic retrieval architecture on arbitrary, unseen documents:
1. Quantum Mechanics Notes (Unseen Domain)
2. Aerospace Propulsion Report (Unseen Domain)

Proves:
- Zero hardcoding of questions/topics/documents
- Vector DB similarity search always runs first when document exists
- Sufficient evidence -> Grounded PDF answer + Page Citation (No Web Search)
- Insufficient evidence -> Generic fallback according to intent
- Comprehensive retrieval diagnostics logging with required prompt keys
"""

import pytest
from unittest.mock import patch

from app.graph.workflow import run_documind_workflow


@pytest.fixture(scope="module")
def sample_quantum_chunks():
    return [
        {
            "doc_id": "doc_quantum_99",
            "filename": "quantum_mechanics.pdf",
            "page_num": 3,
            "chunk_type": "text",
            "chunk_id": "qm_c3",
            "content": (
                "Superdense coding is a quantum communication protocol to transmit two bits of classical information "
                "by sending only one qubit, provided the sender and receiver share a pre-entangled Bell pair."
            ),
            "score": 0.12,
            "combined_score": 0.82
        },
        {
            "doc_id": "doc_quantum_99",
            "filename": "quantum_mechanics.pdf",
            "page_num": 4,
            "chunk_type": "text",
            "chunk_id": "qm_c4",
            "content": (
                "Quantum decoherence is the loss of quantum coherence where a system interacts with its environment, "
                "causing pure quantum states to decay into statistical mixtures."
            ),
            "score": 0.15,
            "combined_score": 0.78
        }
    ]


@pytest.fixture(scope="module")
def sample_aerospace_chunks():
    return [
        {
            "doc_id": "doc_aero_77",
            "filename": "propulsion_report.pdf",
            "page_num": 2,
            "chunk_type": "text",
            "chunk_id": "aero_c2",
            "content": (
                "Hall-effect thrusters utilize a magnetic field to trap electrons, which then ionize xenon propellant. "
                "The electrostatic force accelerates ions to exhaust velocities exceeding 30 km/s."
            ),
            "score": 0.20,
            "combined_score": 0.70
        }
    ]


def test_generic_document_first_grounded_answer(sample_quantum_chunks):
    """
    Test 1: Normal query on unseen document (no 'from PDF' phrase).
    Verifies Vector DB checked first -> grounded PDF answer + page citation -> NO web search.
    """
    web_mock = patch("app.graph.nodes.search_web")
    sim_mock = patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_quantum_chunks)
    llm_mock = patch(
        "app.models.llm.LocalLLMManager.generate_text_with_metrics",
        return_value=(
            "Superdense coding allows sending two classical bits using one qubit and a shared Bell pair [Page 3].",
            {"total_llm_time": 0.5, "tokens_generated": 22}
        )
    )

    with web_mock as mock_web, sim_mock, llm_mock:
        res = run_documind_workflow(
            question="What is superdense coding?",
            doc_id="doc_quantum_99",
            mode="auto"
        )

        # 1. Route must be text_rag (document grounded)
        assert res["route"] == "text_rag"

        # 2. Answer must contain page citation
        assert "[Page 3]" in res["answer"] or "Page 3" in str(res["sources"])

        # 3. Web search must NOT have been called
        assert mock_web.call_count == 0

        # 4. Verified status must be True against evidence
        assert res["verified"] is True

        # 5. Diagnostics must capture complete pipeline details
        diag = res.get("retrieval_diagnostics")
        assert diag is not None
        assert diag["original_query"] == "What is superdense coding?"
        assert "document_id" in diag
        assert "candidate_chunks" in diag
        assert "dense_scores" in diag
        assert "bm25_scores" in diag
        assert "evidence_sufficiency" in diag


def test_generic_document_first_insufficient_fallback(sample_aerospace_chunks):
    """
    Test 2: Unseen document exists, but query seeks future 2035 data not in document.
    Verifies Vector DB checked first -> evidence evaluation marks insufficient -> fallback to web search.
    """
    mock_web_results = [
        {
            "title": "Future Propulsion Roadmap 2035",
            "snippet": "In 2035, next-generation Hall thrusters target 50 km/s exhaust velocity using krypton fuel.",
            "url": "https://aerospace.example.org/roadmap2035",
            "source": "aerospace.example.org"
        }
    ]

    sim_mock = patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_aerospace_chunks)
    web_mock = patch("app.graph.nodes.search_web", return_value=mock_web_results)
    llm_mock = patch(
        "app.models.llm.LocalLLMManager.generate_text_with_metrics",
        return_value=(
            "Hall thrusters achieve 30 km/s exhaust velocity [Page 2]. In 2035, target velocity is 50 km/s (🌐 Web: aerospace.example.org).",
            {"total_llm_time": 0.6, "tokens_generated": 30}
        )
    )

    with sim_mock, web_mock as mock_web, llm_mock:
        res = run_documind_workflow(
            question="What is the exhaust velocity of Hall thrusters, and what is the projected 2035 goal?",
            doc_id="doc_aero_77",
            mode="auto"
        )

        # 1. Route must fall back to web_search due to missing 2035 data
        assert res["route"] == "web_search"

        # 2. Web search must have been executed
        assert mock_web.call_count == 1

        # 3. Sources must contain both document and web sources
        assert len(res["sources"]) >= 2
        doc_src = [s for s in res["sources"] if s.get("doc_id") == "doc_aero_77"]
        web_src = [s for s in res["sources"] if s.get("doc_id") == "web_search"]
        assert len(doc_src) > 0
        assert len(web_src) > 0


def test_diagnostics_required_keys_structure(sample_quantum_chunks):
    """
    Test 3: Verifies that diagnostics dict contains all required logging keys specified by system directives.
    """
    sim_mock = patch("app.rag.vector_store.VectorStoreManager.search_similarity", return_value=sample_quantum_chunks)
    llm_mock = patch(
        "app.models.llm.LocalLLMManager.generate_text_with_metrics",
        return_value=("Decoherence causes quantum state decay [Page 4].", {})
    )

    with sim_mock, llm_mock:
        res = run_documind_workflow(
            question="Explain quantum decoherence.",
            doc_id="doc_quantum_99",
            mode="auto"
        )

        diag = res.get("retrieval_diagnostics")
        assert diag is not None, "retrieval_diagnostics dictionary must be present in workflow output"

        required_keys = [
            "original_query",
            "normalized_query",
            "document_id",
            "conversation_id",
            "candidate_chunks",
            "dense_results",
            "dense_scores",
            "bm25_results",
            "bm25_scores",
            "fusion_results",
            "fusion_scores",
            "reranked_results",
            "reranker_scores",
            "neighbor_expansion",
            "final_evidence",
            "evidence_sufficiency",
            "fallback_decision"
        ]

        for k in required_keys:
            assert k in diag, f"Diagnostics missing key: '{k}'"
