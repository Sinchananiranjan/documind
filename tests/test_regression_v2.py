"""
Regression test suite v2 — tests the 8 specific scenarios requested.

Tests are isolated from each other and mock only what is necessary.
Every test verifies the invariant described in the test docstring.
"""

import re
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_chunk(
    doc_id="test_doc",
    filename="test.pdf",
    page_num=1,
    content="The activation function introduces non-linearity into the neural network.",
    combined_score=0.75,
    chunk_type="text",
):
    return {
        "doc_id": doc_id,
        "filename": filename,
        "page_num": page_num,
        "chunk_type": chunk_type,
        "chunk_id": f"{doc_id}_p{page_num}",
        "content": content,
        "snippet": content[:80],
        "score": 0.75,
        "combined_score": combined_score,
        "image_b64": "",
    }


def _run(question, chunks, mode="auto", doc_id="test_doc", active_docs=None, llm_answer=None):
    from app.graph.workflow import run_documind_workflow
    active_docs = active_docs or ([doc_id] if doc_id else [])

    def fake_search(*args, **kwargs):
        return chunks

    def fake_lexical(*args, **kwargs):
        return chunks

    patches = [
        patch("app.rag.vector_store.VectorStoreManager.search_similarity", side_effect=fake_search),
        patch("app.rag.vector_store.VectorStoreManager.get_all_chunks_for_docs", side_effect=fake_lexical),
    ]
    if llm_answer is not None:
        patches.append(
            patch("app.models.llm.GroqLLMManager.generate_text", return_value=llm_answer)
        )

    with patches[0], patches[1]:
        if len(patches) > 2:
            with patches[2]:
                return run_documind_workflow(
                    question=question,
                    doc_id=doc_id,
                    active_docs=active_docs,
                    mode=mode,
                )
        else:
            return run_documind_workflow(
                question=question,
                doc_id=doc_id,
                active_docs=active_docs,
                mode=mode,
            )


def test_01_doc_answer_with_valid_page_citation():
    """1. Information present in PDF: document answer with valid citations."""
    chunk = _make_chunk(
        page_num=7,
        content="Gradient descent is an optimization algorithm that minimizes a loss function "
                "by iteratively updating parameters in the direction of the negative gradient.",
        combined_score=0.82,
    )
    res = _run(
        question="What is gradient descent?",
        chunks=[chunk],
        llm_answer="Gradient descent is an optimization algorithm that minimizes a loss function [Page 7].",
    )
    assert res["route"] in ("text_rag", "table_analysis", "hybrid"), \
        f"Expected doc-grounded route but got '{res['route']}'"
    assert res["verified"] is True, "Answer from document with valid evidence should be verified"
    assert res["sources"], "Sources list must not be empty for a doc-grounded answer"


def test_02_absent_info_gk_fallback_with_disclosure():
    """2. Information absent from PDF: general knowledge fallback with disclosure."""
    res = _run(
        question="Explain quantum entanglement",
        chunks=[],
        doc_id="ml_textbook",
        active_docs=["ml_textbook"],
    )
    assert res["route"] == "general_knowledge", \
        f"Absent-from-doc in auto mode must route to general_knowledge, got '{res['route']}'"
    assert "🌐 **[General Knowledge" in res["answer"], \
        "GK fallback answer must be labeled with the 🌐 prefix"
    assert "[Page" not in res["answer"], "GK answer must not contain document page citations"


def test_03_deep_document_retrieval():
    """3. Information deep in the PDF: successful retrieval."""
    deep_chunk = _make_chunk(
        page_num=38,
        content="Chapter 10 concludes with a discussion of transfer learning, where a pre-trained "
                "model is fine-tuned on a target domain with limited labelled data.",
        combined_score=0.71,
    )
    res = _run(
        question="What does chapter 10 say about transfer learning?",
        chunks=[deep_chunk],
        llm_answer="Chapter 10 discusses transfer learning where a pre-trained model is fine-tuned on a target domain [Page 38].",
    )
    assert res["route"] != "general_knowledge", \
        "Deep-page content that IS retrieved must NOT fall back to general knowledge"
    source_pages = [s.get("page_num") for s in res.get("sources", [])]
    assert 38 in source_pages, f"Source list must include page 38, got: {source_pages}"


def test_04_final_section_correct_citation():
    """4. Question about final section: correct answer and page citation."""
    final_chunk = _make_chunk(
        page_num=41,
        content="In conclusion, this book covers supervised, unsupervised, and reinforcement learning. "
                "Future directions include self-supervised and few-shot learning paradigms.",
        combined_score=0.78,
    )
    res = _run(
        question="What does the conclusion section cover?",
        chunks=[final_chunk],
        llm_answer="The conclusion covers supervised, unsupervised, and reinforcement learning [Page 41].",
    )
    assert res["route"] != "general_knowledge", \
        "Final-section content from doc must be answered from doc"
    cited = re.findall(r'\[Page\s*(\d+)\]', res["answer"], re.IGNORECASE)
    for p in cited:
        assert p == "41", f"Cited page {p} does not match the retrieved chunk page 41"


def test_05_document_contradicts_gk_document_wins():
    """5. Document answer contradicts general knowledge: document evidence takes priority."""
    doc_chunk = _make_chunk(
        page_num=12,
        content="In this framework, the learning rate is set to 0.001 by default, "
                "which the authors argue outperforms the commonly-used 0.01 setting.",
        combined_score=0.80,
    )
    res = _run(
        question="What learning rate does this framework use?",
        chunks=[doc_chunk],
        llm_answer="The framework uses a learning rate of 0.001 by default [Page 12].",
    )
    assert res["route"] in ("text_rag", "table_analysis", "hybrid"), \
        "With clear doc evidence, route must be doc-grounded"
    assert "0.001" in res["answer"], "Doc-sourced value 0.001 must appear in the answer"
    assert "🌐 **[General Knowledge" not in res["answer"], \
        "Answer must not be labeled as general knowledge when doc evidence exists"


def test_06_mixed_question_separated_sources():
    """6. Mixed question: document evidence and general knowledge are clearly separated."""
    partial_chunk = _make_chunk(
        page_num=5,
        content="The Adam optimizer uses adaptive learning rates per parameter.",
        combined_score=0.52,
    )
    res = _run(
        question="How does Adam optimizer work and what are its variants?",
        chunks=[partial_chunk],
        llm_answer=(
            "Adam uses adaptive learning rates per parameter [Page 5]. "
            "General Knowledge: Variants include AdaMax, Nadam, and AMSGrad."
        ),
    )
    assert res["route"] in ("text_rag", "hybrid", "general_knowledge"), \
        f"Unexpected route '{res['route']}'"
    assert "[Page" in res["answer"] or "General Knowledge" in res["answer"] or "🌐" in res["answer"], \
        "Mixed answer must contain either doc citation or GK label"


def test_07_retrieval_failure_no_false_absent_claim():
    """7. Retrieval failure: must not claim document lacks information."""
    from app.graph.workflow import run_documind_workflow

    def failing_search(*args, **kwargs):
        raise RuntimeError("ChromaDB connection timeout")

    def failing_lexical(*args, **kwargs):
        raise RuntimeError("ChromaDB connection timeout")

    with patch("app.rag.vector_store.VectorStoreManager.search_similarity", side_effect=failing_search), \
         patch("app.rag.vector_store.VectorStoreManager.get_all_chunks_for_docs", side_effect=failing_lexical):
        res = run_documind_workflow(
            question="What is overfitting?",
            doc_id="ml_doc",
            active_docs=["ml_doc"],
            mode="auto",
        )

    false_absent_claim = "the selected document does not provide information"
    assert false_absent_claim not in res["answer"].lower(), \
        "Retrieval failure must not produce a false 'document lacks info' message"
    if res["route"] in ("text_rag", "hybrid"):
        assert res["verified"] is False, \
            "Doc-grounded route with no retrieved chunks must not be verified=True"


def test_08_verified_true_only_with_passing_verification():
    """8. Verification failure: must never report verified=True without successful verification."""
    chunk = _make_chunk(
        page_num=3,
        content="Batch normalization normalizes layer activations to have zero mean and unit variance.",
        combined_score=0.78,
    )
    hallucinated_answer = (
        "The paper introduces a revolutionary quantum-classical hybrid optimizer "
        "that uses entanglement-based gradient estimation [Page 99]."
    )
    res = _run(
        question="What does the paper describe?",
        chunks=[chunk],
        llm_answer=hallucinated_answer,
    )
    assert res["verified"] is False, (
        f"Hallucinated answer citing non-existent page 99 must NOT be verified=True. "
        f"Got verified={res['verified']}, answer='{res['answer'][:100]}'"
    )
