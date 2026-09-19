"""State definition for DocuMind LangGraph workflow."""

from typing import TypedDict, List, Dict, Any, Optional

class DocuMindState(TypedDict):
    """
    TypedDict state passed between all nodes in the DocuMind LangGraph pipeline.
    
    Fields:
        question: User query string.
        doc_id: Document ID for vector store filtering.
        context_chunks: Retrieved document chunks from ChromaDB.
        route: Decision made by router ('text_rag', 'table_analysis', 'image_analysis').
        answer: Generated LLM answer string.
        sources: Page citations and metadata corresponding to answer.
        verified: Boolean flag indicating if answer passed hallucination check.
        retry_count: Number of verification retries performed (max 2).
        error: Optional error tracking message.
    """
    question: str
    doc_id: str
    context_chunks: List[Dict[str, Any]]
    route: str
    answer: str
    sources: List[Dict[str, Any]]
    verified: bool
    retry_count: int
    error: Optional[str]
