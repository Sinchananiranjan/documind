"""LangGraph node implementations for DocuMind workflow."""

import logging
from typing import Dict, Any, List

from app.graph.state import DocuMindState
from app.rag.vector_store import VectorStoreManager
from app.models.llm import LocalLLMManager

logger = logging.getLogger(__name__)

# Singletons for nodes
vector_manager = VectorStoreManager()
llm_manager = LocalLLMManager()


# ============================================================================
# 1. RETRIEVAL NODE
# ============================================================================
def retrieve_node(state: DocuMindState) -> Dict[str, Any]:
    """
    LangGraph Node: Retrieves top relevant document chunks from ChromaDB.
    """
    logger.info(f"[LangGraph Node: RETRIEVE] Query: '{state['question']}' | Doc ID: '{state['doc_id']}'")
    
    chunks = vector_manager.search_similarity(
        query=state["question"],
        doc_id=state["doc_id"],
        k=6
    )
    
    sources = []
    for c in chunks:
        sources.append({
            "page_num": c["page_num"],
            "chunk_type": c["chunk_type"],
            "snippet": c["content"][:150] + "..." if len(c["content"]) > 150 else c["content"],
            "doc_id": c["doc_id"]
        })
        
    return {
        "context_chunks": chunks,
        "sources": sources
    }


# ============================================================================
# 2. ROUTER NODE
# ============================================================================
def router_node(state: DocuMindState) -> Dict[str, Any]:
    """
    LangGraph Node: Determines execution branch (text_rag, table_analysis, image_analysis).
    """
    q_lower = state["question"].lower()
    chunks = state.get("context_chunks", [])
    
    # Table detection keywords or table markdown presence
    table_keywords = ["table", "column", "row", "total", "stat", "metric", "revenue", "price", "figure", "summary"]
    has_table_keyword = any(k in q_lower for k in table_keywords)
    has_table_chunk = any(c.get("chunk_type") == "table" for c in chunks)

    # Image detection keywords or image presence
    image_keywords = ["image", "picture", "photo", "diagram", "chart", "figure", "logo", "illustration"]
    has_image_keyword = any(k in q_lower for k in image_keywords)
    has_image_chunk = any(c.get("chunk_type") == "image" for c in chunks)

    if (has_table_keyword or has_table_chunk) and not has_image_keyword:
        route = "table_analysis"
    elif (has_image_keyword or has_image_chunk):
        route = "image_analysis"
    else:
        route = "text_rag"

    logger.info(f"[LangGraph Node: ROUTER] Decision: '{route}'")
    return {"route": route}


# ============================================================================
# 3. TEXT RAG NODE
# ============================================================================
def text_rag_node(state: DocuMindState) -> Dict[str, Any]:
    """
    LangGraph Node: Generates answer for general textual queries.
    """
    logger.info("[LangGraph Node: TEXT_RAG] Generating text answer...")
    chunks = state.get("context_chunks", [])
    
    if not chunks:
        return {"answer": "I couldn't find this in the document."}

    formatted_context = "\n\n".join([
        f"[Page {c['page_num']} ({c['chunk_type'].upper()})]:\n{c['content']}"
        for c in chunks
    ])

    system_prompt = (
        "You are DocuMind, a helpful multimodal document AI assistant. "
        "Answer the user's question accurately using ONLY the provided document context. "
        "Always explicitly cite page numbers (e.g., 'According to Page 2...'). "
        "If the information is not present in the context, respond EXACTLY: "
        "'I couldn't find this in the document.'"
    )

    prompt = (
        f"DOCUMENT CONTEXT:\n{formatted_context}\n\n"
        f"USER QUESTION: {state['question']}\n\n"
        "ANSWER:"
    )

    answer = llm_manager.generate_text(prompt, system_prompt=system_prompt)
    if not answer or len(answer.strip()) == 0:
        answer = "I couldn't find this in the document."

    return {"answer": answer}


# ============================================================================
# 4. TABLE ANALYSIS NODE
# ============================================================================
def table_analysis_node(state: DocuMindState) -> Dict[str, Any]:
    """
    LangGraph Node: Specialized prompt engineering for tabular data analysis.
    """
    logger.info("[LangGraph Node: TABLE_ANALYSIS] Processing tabular chunks...")
    chunks = state.get("context_chunks", [])
    
    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
    if not table_chunks:
        table_chunks = chunks  # Fallback to all chunks if specific table chunk missing

    formatted_tables = "\n\n".join([
        f"[Page {c['page_num']} Table Data]:\n{c['content']}"
        for c in table_chunks
    ])

    system_prompt = (
        "You are DocuMind Table Analyzer. Analyze the provided Markdown table data. "
        "Extract exact numbers, values, and column comparisons. "
        "Always cite page numbers for table data. "
        "If the table does not contain the requested data, state: 'I couldn't find this in the document.'"
    )

    prompt = (
        f"TABLE CONTEXT:\n{formatted_tables}\n\n"
        f"QUESTION: {state['question']}\n\n"
        "TABLE ANALYSIS ANSWER:"
    )

    answer = llm_manager.generate_text(prompt, system_prompt=system_prompt)
    if not answer or len(answer.strip()) == 0:
        answer = "I couldn't find this in the document."

    return {"answer": answer}


# ============================================================================
# 5. IMAGE ANALYSIS NODE
# ============================================================================
def image_analysis_node(state: DocuMindState) -> Dict[str, Any]:
    """
    LangGraph Node: Processes image / visual element queries.
    """
    logger.info("[LangGraph Node: IMAGE_ANALYSIS] Processing visual element...")
    chunks = state.get("context_chunks", [])
    image_chunks = [c for c in chunks if c.get("chunk_type") == "image" and c.get("image_b64")]

    if image_chunks and llm_manager.has_vision_model():
        img_b64 = image_chunks[0]["image_b64"]
        page_num = image_chunks[0]["page_num"]
        vision_prompt = f"Describe what is shown in this image from Page {page_num} and answer: {state['question']}"
        answer = llm_manager.analyze_image(img_b64, vision_prompt)
        if answer:
            return {"answer": f"[Page {page_num} Image Vision Analysis]: {answer}"}

    # Fallback to textual description / OCR text extracted from image
    formatted_img_desc = "\n\n".join([
        f"[Page {c['page_num']} Image Info]:\n{c['content']}"
        for c in chunks
    ])

    prompt = (
        f"IMAGE METADATA & OCR TEXT:\n{formatted_img_desc}\n\n"
        f"QUESTION: {state['question']}\n\n"
        "Provide answer based on extracted image text and descriptions. Cite page numbers. "
        "If details are unavailable, answer: 'I couldn't find this in the document.'"
    )

    answer = llm_manager.generate_text(prompt)
    if not answer:
        answer = "I couldn't find this in the document."

    return {"answer": answer}


# ============================================================================
# 6. VERIFY ANSWER NODE
# ============================================================================
def verify_answer_node(state: DocuMindState) -> Dict[str, Any]:
    """
    LangGraph Node: Verifies if the answer is factual and grounded in context.
    Prevents hallucinations and increments retry counter if verification fails.
    """
    logger.info("[LangGraph Node: VERIFY_ANSWER] Validating answer groundedness...")
    answer = state.get("answer", "")
    chunks = state.get("context_chunks", [])
    retry_count = state.get("retry_count", 0)

    # Standard negative answer requires no further verification
    if "I couldn't find this in the document" in answer:
        return {"verified": True}

    formatted_context = "\n".join([c["content"] for c in chunks])
    is_valid = llm_manager.verify_groundedness(
        question=state["question"],
        context=formatted_context,
        answer=answer
    )

    logger.info(f"Verification result: Grounded={is_valid} | Retry Count: {retry_count}")

    return {
        "verified": is_valid,
        "retry_count": retry_count + 1 if not is_valid else retry_count
    }


# ============================================================================
# 7. FALLBACK NODE
# ============================================================================
def fallback_node(state: DocuMindState) -> Dict[str, Any]:
    """
    LangGraph Node: Triggered when answer fails verification after retries.
    Enforces safe fallback message.
    """
    logger.warning("[LangGraph Node: FALLBACK] Maximum retries reached or ungrounded answer. Returning fallback.")
    return {
        "answer": "I couldn't find this in the document.",
        "verified": True
    }
