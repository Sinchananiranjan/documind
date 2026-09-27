"""Domain-agnostic LangGraph Workflow construction for DocuMind."""

import time
import logging
from typing import Dict, Any, Optional, List

from langgraph.graph import StateGraph, START, END

from app.graph.state import DocuMindState
from app.graph.nodes import (
    retrieve_node,
    router_node,
    text_rag_node,
    table_analysis_node,
    image_analysis_node,
    calculation_node,
    hybrid_node,
    general_knowledge_node,
    web_search_node,
    web_enhanced_answer_node,
    verify_answer_node,
    fallback_node
)

logger = logging.getLogger(__name__)


def route_decision(state: DocuMindState) -> str:
    """Conditional Edge function: Routes state after router node."""
    return state.get("route", "text_rag")


def verify_decision(state: DocuMindState) -> str:
    """
    Conditional Edge function: Evaluates verification outcome.
    """
    answer = state.get("answer", "")
    if state.get("verified", False):
        return "end"
    else:
        return "fallback"


def build_documind_graph():
    """
    Builds, connects, and compiles the domain-agnostic DocuMind LangGraph workflow.
    Includes web search node for external knowledge fallback.
    """
    graph = StateGraph(DocuMindState)

    # 1. Add Nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("router", router_node)
    graph.add_node("text_rag", text_rag_node)
    graph.add_node("table_analysis", table_analysis_node)
    graph.add_node("image_analysis", image_analysis_node)
    graph.add_node("calculation", calculation_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("general_knowledge", general_knowledge_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("web_enhanced_answer", web_enhanced_answer_node)
    graph.add_node("verify_answer", verify_answer_node)
    graph.add_node("fallback", fallback_node)

    # 2. Add Fixed Edges: Start with Router Node FIRST
    graph.add_edge(START, "router")

    # 3. Add Router Conditional Edge: route to retrieve or direct capability
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "text_rag": "retrieve",
            "table_analysis": "retrieve",
            "image_analysis": "retrieve",
            "hybrid": "retrieve",
            "calculation": "retrieve",
            "general_knowledge": "general_knowledge",
            "web_search": "web_search"
        }
    )

    # 4. Add Retrieve Conditional Edge: route to document analysis or adjusted capability
    graph.add_conditional_edges(
        "retrieve",
        route_decision,
        {
            "text_rag": "text_rag",
            "table_analysis": "table_analysis",
            "image_analysis": "image_analysis",
            "hybrid": "hybrid",
            "calculation": "calculation",
            "general_knowledge": "general_knowledge",
            "web_search": "web_search"
        }
    )

    # 5. Connect analysis nodes to verification
    graph.add_edge("text_rag", "verify_answer")
    graph.add_edge("table_analysis", "verify_answer")
    graph.add_edge("image_analysis", "verify_answer")
    graph.add_edge("calculation", "verify_answer")
    graph.add_edge("hybrid", "verify_answer")
    graph.add_edge("general_knowledge", "verify_answer")

    # 5. Web search pipeline: web_search → web_enhanced_answer → verify_answer
    graph.add_edge("web_search", "web_enhanced_answer")
    graph.add_edge("web_enhanced_answer", "verify_answer")

    # 6. Add Verification Conditional Edge
    graph.add_conditional_edges(
        "verify_answer",
        verify_decision,
        {
            "end": END,
            "fallback": "fallback"
        }
    )

    graph.add_edge("fallback", END)

    compiled_graph = graph.compile()
    return compiled_graph


# Pre-compiled workflow instance
documind_graph = build_documind_graph()


def run_documind_workflow(
    question: str,
    doc_id: Optional[str] = None,
    mode: str = "auto",
    pdf_path: Optional[str] = None,
    filename: Optional[str] = None,
    uploaded_image_b64: Optional[str] = None,
    uploaded_image_ocr: Optional[str] = None,
    uploaded_image_path: Optional[str] = None,
    conversation_id: Optional[str] = None,
    active_docs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Invoke compiled LangGraph workflow with mode support and execution timing.
    """
    start_total = time.perf_counter()

    initial_state: DocuMindState = {
        "conversation_id": conversation_id,
        "active_docs": active_docs or [],
        "question": question,
        "doc_id": doc_id,
        "mode": mode,
        "pdf_path": pdf_path,
        "filename": filename,
        "context_chunks": [],
        "route": "text_rag",
        "answer": "",
        "sources": [],
        "verified": False,
        "retry_count": 0,
        "timings": {},
        "error": None,
        "doc_relevance": 0.0,
        "uploaded_image_b64": uploaded_image_b64,
        "uploaded_image_ocr": uploaded_image_ocr,
        "uploaded_image_path": uploaded_image_path,
        "web_search_results": [],
        "evidence_sufficiency": None,
    }


    final_state = documind_graph.invoke(initial_state)

    total_time = round(time.perf_counter() - start_total, 3)
    timings = final_state.get("timings", {})
    timings["total"] = total_time

    return {
        "question": final_state["question"],
        "doc_id": final_state["doc_id"],
        "mode": final_state.get("mode", "document_mode"),
        "route": final_state.get("route", "text_rag"),
        "answer": final_state.get("answer", "I couldn't find this in the selected document."),
        "sources": final_state.get("sources", []),
        "verified": final_state.get("verified", False),
        "doc_relevance": final_state.get("doc_relevance", 0.0),
        "evidence_sufficiency": final_state.get("evidence_sufficiency"),
        "retry_count": final_state.get("retry_count", 0),
        "timings": timings,
        "web_search_results": final_state.get("web_search_results", []),
    }
