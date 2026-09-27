"""State definition for DocuMind LangGraph workflow."""

from typing import TypedDict, List, Dict, Any, Optional

class DocuMindState(TypedDict):
    """
    TypedDict state passed between all nodes in the DocuMind LangGraph pipeline.

    Fields:
        conversation_id: Unique ID for the current conversation.
        active_docs: List of document IDs associated with this conversation.
        question: User query string.
        doc_id: Active document ID for vector store filtering.
        mode: Execution mode ('document_mode' or 'general_knowledge_mode').
        pdf_path: Path to active PDF file.
        filename: Name of active PDF file.
        context_chunks: Retrieved document chunks from ChromaDB.
        route: Decision made by router ('text_rag', 'table_analysis', 'image_analysis',
               'calculation', 'general_knowledge', 'hybrid', 'web_search').
        answer: Generated LLM answer string.
        sources: Page citations and metadata corresponding to answer.
        verified: Boolean flag indicating if answer passed hallucination check.
        retry_count: Number of verification retries performed.
        timings: Execution timing breakdown in seconds.
        error: Optional error tracking message.
        doc_relevance: Relevance score of retrieved chunks (0.0 to 1.0).
        uploaded_image_b64: Base64-encoded standalone image uploaded by user (PNG/JPG/WEBP).
        uploaded_image_ocr: OCR text extracted from the uploaded standalone image.
        uploaded_image_path: Original file path/name of the uploaded standalone image.
        web_search_results: List of web search result dicts with title/snippet/url/source.
    """
    conversation_id: Optional[str]
    active_docs: List[str]
    question: str
    doc_id: Optional[str]
    mode: str
    pdf_path: Optional[str]
    filename: Optional[str]
    context_chunks: List[Dict[str, Any]]
    route: str
    answer: str
    sources: List[Dict[str, Any]]
    verified: bool
    retry_count: int
    timings: Dict[str, float]
    error: Optional[str]
    # Relevance score of retrieved chunks to the question (0.0 = no match, 1.0 = perfect match)
    doc_relevance: float
    # Standalone image upload fields (optional — only set when user uploads an image directly)
    uploaded_image_b64: Optional[str]
    uploaded_image_ocr: Optional[str]
    uploaded_image_path: Optional[str]
    # Web search results (populated by web_search_node when triggered)
    web_search_results: List[Dict[str, Any]]
    # Evidence sufficiency analysis (evaluated after retrieval & reranking)
    evidence_sufficiency: Optional[Dict[str, Any]]


