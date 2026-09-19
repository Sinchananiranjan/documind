"""LangGraph Workflow construction for DocuMind."""

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END

from app.graph.state import DocuMindState
from app.graph.nodes import (
    retrieve_node,
    router_node,
    text_rag_node,
    table_analysis_node,
    image_analysis_node,
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
    - If verified: proceed to END.
    - If unverified and retries remaining: retry via router.
    - If retries exhausted: proceed to fallback node.
    """
    if state.get("verified", False):
        return "end"
    elif state.get("retry_count", 0) < 2:
        return "retry"
    else:
        return "fallback"


def build_documind_graph():
    """
    Builds, connects, and compiles the DocuMind LangGraph workflow.
    
    Graph Topology:
    
        [START]
           │
      (retrieve)
           │
       (router)
        ╱  │  ╲
       ╱   │   ╲
      v    v    v
    (text)(table)(image)
      ╲    │    ╱
       ╲   │   ╱
        v  v  v
     (verify_answer)
        ╱     ╲
       ╱       ╲
     (valid)  (invalid / max retries)
       │         │
     [END]    (fallback) ──> [END]
    """
    graph = StateGraph(DocuMindState)

    # 1. Add Nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("router", router_node)
    graph.add_node("text_rag", text_rag_node)
    graph.add_node("table_analysis", table_analysis_node)
    graph.add_node("image_analysis", image_analysis_node)
    graph.add_node("verify_answer", verify_answer_node)
    graph.add_node("fallback", fallback_node)

    # 2. Add Fixed Edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "router")

    # 3. Add Router Conditional Edge
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "text_rag": "text_rag",
            "table_analysis": "table_analysis",
            "image_analysis": "image_analysis"
        }
    )

    # 4. Connect analysis nodes to verification
    graph.add_edge("text_rag", "verify_answer")
    graph.add_edge("table_analysis", "verify_answer")
    graph.add_edge("image_analysis", "verify_answer")

    # 5. Add Verification Conditional Edge
    graph.add_conditional_edges(
        "verify_answer",
        verify_decision,
        {
            "end": END,
            "retry": "router",
            "fallback": "fallback"
        }
    )

    graph.add_edge("fallback", END)

    compiled_graph = graph.compile()
    return compiled_graph


# Pre-compiled workflow instance
documind_graph = build_documind_graph()


def run_documind_workflow(question: str, doc_id: str) -> Dict[str, Any]:
    """
    Helper function to invoke the compiled LangGraph workflow with an initial state.
    """
    initial_state: DocuMindState = {
        "question": question,
        "doc_id": doc_id,
        "context_chunks": [],
        "route": "text_rag",
        "answer": "",
        "sources": [],
        "verified": False,
        "retry_count": 0,
        "error": None
    }

    final_state = documind_graph.invoke(initial_state)
    return {
        "question": final_state["question"],
        "doc_id": final_state["doc_id"],
        "route": final_state.get("route", "text_rag"),
        "answer": final_state.get("answer", "I couldn't find this in the document."),
        "sources": final_state.get("sources", []),
        "verified": final_state.get("verified", False),
        "retry_count": final_state.get("retry_count", 0)
    }
