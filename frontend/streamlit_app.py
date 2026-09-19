"""DocuMind Streamlit Frontend Application.

Runs standalone by directly invoking Python pipeline services or interfacing with FastAPI.
"""

import os
import sys
import tempfile
import streamlit as st

# Add parent directory to sys.path to enable direct imports of app components
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow
from app.models.llm import LocalLLMManager
from data.generate_sample import generate_sample_pdf

# Initialize App Config & Page Setup
st.set_page_config(
    page_title="DocuMind - Multimodal AI Document Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-route {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-verified {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-retry {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_dict=True)

# Singleton instances for direct Streamlit execution
@st.cache_resource
def get_services():
    processor = PDFProcessor()
    vector_mgr = VectorStoreManager()
    llm_mgr = LocalLLMManager()
    return processor, vector_mgr, llm_mgr

pdf_processor, vector_manager, llm_manager = get_services()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None


# Helper function to process and load document into session state
def process_and_set_document(file_path: str):
    with st.spinner("Processing PDF (Text, OCR, Tables, Images) & Indexing into ChromaDB..."):
        res = pdf_processor.process_pdf(file_path)
        summary = res["summary"]
        chunks = res["chunks"]
        vector_manager.add_document_chunks(chunks)
        
        st.session_state.current_doc_id = summary["doc_id"]
        st.session_state.doc_summary = summary
        st.session_state.messages = []  # Clear previous chat on new document
        st.success(f"Loaded '{summary['filename']}' ({summary['total_pages']} pages, {summary['tables_found']} tables, {summary['images_found']} images)")


# Sidebar Layout
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pdf-extract.png", width=64)
    st.markdown("## **DocuMind**")
    st.markdown("Multimodal AI Document Analyzer")
    st.divider()

    # Model Status Card
    st.markdown("### 🤖 Local AI Models")
    installed_models = llm_manager.get_available_models()
    active_model = llm_manager.select_active_model()
    has_vision = llm_manager.has_vision_model()

    st.info(f"**LLM Model:** `{active_model}`\n\n**Vision Support:** `{'Available (Moondream)' if has_vision else 'OCR Fallback'}`")

    st.divider()

    # Upload PDF Section
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Process Uploaded PDF", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            process_and_set_document(tmp_path)
            os.remove(tmp_path)

    # Sample Document Generator Button
    st.markdown("---")
    st.markdown("#### Test Sample PDF")
    if st.button("🧪 Generate & Load Sample PDF", use_container_width=True):
        sample_path = generate_sample_pdf()
        process_and_set_document(sample_path)

    st.divider()

    # Document Management & Switcher
    st.markdown("### 📚 Document Library")
    indexed_docs = vector_manager.list_indexed_documents()
    if indexed_docs:
        selected_doc = st.selectbox(
            "Select Active Document ID",
            options=indexed_docs,
            index=indexed_docs.index(st.session_state.current_doc_id) if st.session_state.current_doc_id in indexed_docs else 0
        )
        st.session_state.current_doc_id = selected_doc

        if st.button("🗑️ Delete Selected Document", use_container_width=True):
            vector_manager.delete_document(selected_doc)
            st.session_state.current_doc_id = None
            st.session_state.doc_summary = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No documents currently indexed in ChromaDB.")


# Main Application View
st.markdown('<div class="main-header">DocuMind – Multimodal Document Analyzer</div>', unsafe_allow_dict=True)
st.markdown('<div class="sub-header">Ask questions about PDFs containing text, scanned pages, tables, and images. Powered by LangGraph & Ollama.</div>', unsafe_allow_dict=True)

# Active Document Summary Header
if st.session_state.doc_summary:
    s = st.session_state.doc_summary
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Filename", s["filename"][:18] + "..." if len(s["filename"]) > 18 else s["filename"])
    col2.metric("Pages", s["total_pages"])
    col3.metric("Scanned", s["scanned_pages"])
    col4.metric("Tables", s["tables_found"])
    col5.metric("Images", s["images_found"])
    st.divider()

# Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            meta = msg["metadata"]
            st.markdown(
                f'<span class="badge-route">Route: {meta.get("route")}</span> '
                f'<span class="badge-verified">Verified: {meta.get("verified")}</span>',
                unsafe_allow_dict=True
            )
            if meta.get("sources"):
                with st.expander("📌 View Page Citations & Sources"):
                    for src in meta["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<b>Page {src.get("page_num")}</b> ({src.get("chunk_type").upper()})<br/>'
                            f'<i>Snippet:</i> {src.get("snippet")}'
                            f'</div>',
                            unsafe_allow_dict=True
                        )

# Question Input Handler
question = st.chat_input("Ask a question about the active document...")

if question:
    if not st.session_state.current_doc_id:
        st.warning("Please upload a PDF or generate the Sample PDF from the sidebar first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Assistant generation via LangGraph Workflow
        with st.chat_message("assistant"):
            with st.spinner("LangGraph Workflow executing (Retrieve -> Route -> Analyze -> Verify)..."):
                workflow_output = run_documind_workflow(
                    question=question,
                    doc_id=st.session_state.current_doc_id
                )

                answer = workflow_output["answer"]
                route = workflow_output["route"]
                verified = workflow_output["verified"]
                sources = workflow_output["sources"]

                st.markdown(answer)

                st.markdown(
                    f'<span class="badge-route">Route: {route}</span> '
                    f'<span class="badge-verified">Grounded Verified: {verified}</span>',
                    unsafe_allow_dict=True
                )

                if sources:
                    with st.expander("📌 View Page Citations & Sources"):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-box">'
                                f'<b>Page {src.get("page_num")}</b> ({src.get("chunk_type").upper()})<br/>'
                                f'<i>Snippet:</i> {src.get("snippet")}'
                                f'</div>',
                                unsafe_allow_dict=True
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "route": route,
                        "verified": verified,
                        "sources": sources
                    }
                })
