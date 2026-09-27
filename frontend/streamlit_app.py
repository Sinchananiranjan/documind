"""DocuMind Streamlit Frontend Application.

ChatGPT-Style Multimodal Document AI Application with:
- Dedicated Chat History Scroll Container
- Permanently Fixed Bottom Chat Composer with Inline '+' Attachment Popover
- Unpersisted Lazy Conversation Lifecycle (Saved ONLY upon message send or file upload)
- Conversation-Scoped Document Memory & Metadata Persistence
- Dynamic Document Reference Resolution ("first PDF", "second PDF", filename)
- Universal Single Auto-Routing Engine
- Clickable PDF Citations & Page Viewer
"""

import os
import sys
import base64
import tempfile
import io
import shutil
import hashlib
import streamlit as st
from PIL import Image

# Add parent directory to sys.path to enable direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow
from app.models.llm import LocalLLMManager
from app.conversation_manager import ConversationManager

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind - Multimodal AI Document Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Premium Aesthetics & CSS ───────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts & Root Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main App Header */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        padding: 20px 24px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #F8FAFC;
    }
    .app-header p {
        font-size: 0.92rem;
        color: #94A3B8;
        margin: 6px 0 0 0;
    }

    /* Badges & Indicators */
    .badge-route {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-verified-true {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-verified-false {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-timing {
        background-color: #F1F5F9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.80rem;
    }
    
    /* Document Cards */
    .doc-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 16px;
    }
    .doc-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #1E293B;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Citation Cards */
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .source-card-web {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .pdf-viewer-box {
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 16px;
        background-color: #EFF6FF;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.12);
    }
    
    /* Sidebar Styling */
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* ── Permanently Fixed Bottom Chat Composer ────────────────────────────── */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stChatInput"]) {
        position: fixed !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: min(880px, 92vw) !important;
        z-index: 99999 !important;
        background-color: #FFFFFF !important;
        padding: 12px 18px 16px 18px !important;
        border-top: 1px solid #E2E8F0 !important;
        box-shadow: 0 -4px 25px rgba(0, 0, 0, 0.08) !important;
        border-radius: 18px 18px 0 0 !important;
        align-items: center !important;
    }

    /* Reserve bottom padding in main container */
    .stMainBlockContainer {
        padding-bottom: 95px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Service Singletons ─────────────────────────────────────────────────
@st.cache_resource
def get_services():
    processor = PDFProcessor()
    vector_mgr = VectorStoreManager()
    llm_mgr = LocalLLMManager()
    conv_mgr = ConversationManager()
    return processor, vector_mgr, llm_mgr, conv_mgr

pdf_processor, vector_manager, llm_manager, conv_manager = get_services()


# ── Session State Setup ───────────────────────────────────────────────────────
def _init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_init_state("current_conv_id", None)
_init_state("preview_doc_id", None)
_init_state("preview_page", 1)
_init_state("requested_preview_page", None)
_init_state("show_pdf_preview", False)
_init_state("app_mode", "auto")

# LIFECYCLE: Whenever the user opens DocuMind, start with a fresh, empty conversation draft.
# This unpersisted draft is NOT saved to History until a message is sent or file uploaded.
if not st.session_state.current_conv_id:
    st.session_state.current_conv_id = conv_manager.create_conversation(mode="auto", save_immediately=False)

current_conv = conv_manager.get_conversation(st.session_state.current_conv_id)
if not current_conv:
    st.session_state.current_conv_id = conv_manager.create_conversation(mode="auto", save_immediately=False)
    current_conv = conv_manager.get_conversation(st.session_state.current_conv_id)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_conversation(conv_id: str):
    st.session_state.current_conv_id = conv_id
    st.session_state.preview_doc_id = None
    st.session_state.show_pdf_preview = False
    st.session_state.app_mode = "auto"
    st.rerun()

def process_uploaded_file(uploaded_file) -> bool:
    name = uploaded_file.name.lower()
    curr_conv_id = st.session_state.current_conv_id
    curr_conv = conv_manager.get_conversation(curr_conv_id)
    active_doc_ids = curr_conv.get("active_docs", []) if curr_conv else []

    if name.endswith(".pdf"):
        with st.spinner(f"Processing PDF '{uploaded_file.name}'..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                res = pdf_processor.process_pdf(tmp_path, original_filename=uploaded_file.name)
                doc_id = res["summary"]["doc_id"]
                page_count = res["summary"].get("num_pages", 1)
                
                if doc_id in active_doc_ids:
                    st.info(f"PDF '{uploaded_file.name}' is already attached to this conversation.")
                    return True

                conv_manager.add_active_doc(curr_conv_id, doc_id, filename=uploaded_file.name, page_count=page_count)
                vector_manager.add_document_chunks(res["chunks"], conversation_id=curr_conv_id)
                st.success(f"Attached PDF: {uploaded_file.name}")
                return True
            except Exception as e:
                st.error(f"Failed to process PDF {uploaded_file.name}: {e}")
                return False
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        with st.spinner(f"Processing Image '{uploaded_file.name}'..."):
            try:
                img = Image.open(uploaded_file)
                max_side = 1280
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                ocr_text = ""
                try:
                    import pytesseract
                    tess_path = shutil.which("tesseract")
                    if tess_path:
                        pytesseract.pytesseract.tesseract_cmd = tess_path
                    else:
                        for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
                            if os.path.exists(p):
                                pytesseract.pytesseract.tesseract_cmd = p
                                break
                    ocr_text = pytesseract.image_to_string(img).strip()
                except Exception:
                    pass

                doc_id = "img_" + hashlib.md5(img_b64.encode()).hexdigest()[:10]
                if doc_id in active_doc_ids:
                    st.info(f"Image '{uploaded_file.name}' is already attached to this conversation.")
                    return True

                chunk = {
                    "doc_id": doc_id,
                    "filename": uploaded_file.name,
                    "page_num": 1,
                    "chunk_type": "image",
                    "chunk_id": f"{doc_id}_img1",
                    "content": f"[Standalone Image Upload] {uploaded_file.name}\nOCR Text:\n{ocr_text}",
                    "image_b64": img_b64
                }
                
                conv_manager.add_active_doc(curr_conv_id, doc_id, filename=uploaded_file.name, page_count=1)
                vector_manager.add_document_chunks([chunk], conversation_id=curr_conv_id)
                st.success(f"Attached Image: {uploaded_file.name}")
                return True
            except Exception as e:
                st.error(f"Failed to process image {uploaded_file.name}: {e}")
                return False


# ── Sidebar Interface ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 **DocuMind AI**")
    
    # ➕ New Chat Primary Action Button
    if st.button("➕ New Chat", width="stretch", type="primary"):
        new_id = conv_manager.create_conversation(mode="auto", save_immediately=False)
        load_conversation(new_id)

    st.divider()

    # Searchable History
    st.markdown("### 💬 Chat History")
    search_query = st.text_input("Search history...", key="search_history", label_visibility="collapsed", placeholder="🔍 Search history...")

    saved_convs = conv_manager.list_conversations()
    if search_query.strip():
        saved_convs = [c for c in saved_convs if search_query.lower() in c.get("title", "").lower()]

    if not saved_convs:
        st.caption("No saved chat history yet. Send a message or attach a PDF to save a chat.")
    else:
        for c in saved_convs:
            c_id = c["id"]
            title_text = c.get("title", "Conversation")
            if len(title_text) > 22:
                title_text = title_text[:20] + "..."

            is_active = (c_id == st.session_state.current_conv_id)
            btn_type = "primary" if is_active else "secondary"
            
            col_btn, col_opt = st.columns([0.82, 0.18])
            with col_btn:
                if st.button(f"💬 {title_text}", key=f"conv_btn_{c_id}", width="stretch", type=btn_type):
                    load_conversation(c_id)
            with col_opt:
                with st.popover("⋮"):
                    new_title = st.text_input("Rename", value=c['title'], key=f"rename_input_{c_id}")
                    if st.button("Save Title", key=f"save_title_{c_id}", width="stretch"):
                        conv_manager.update_conversation(c_id, {"title": new_title})
                        st.rerun()
                    
                    if st.button("Delete Chat", key=f"del_btn_{c_id}", width="stretch"):
                        deleted_docs = conv_manager.delete_conversation(c_id)
                        for d_id in deleted_docs:
                            vector_manager.delete_document(d_id, conversation_id=c_id)
                        
                        if c_id == st.session_state.current_conv_id:
                            new_id = conv_manager.create_conversation(mode="auto", save_immediately=False)
                            load_conversation(new_id)
                        else:
                            st.rerun()

    st.divider()

    # AI Engine Status Widget
    st.markdown("### 🤖 System Status")
    is_online = llm_manager.check_connection()
    active_model = llm_manager.select_active_model()
    provider_name = getattr(llm_manager, "get_provider_name", lambda: "Groq Cloud API")()

    if is_online:
        st.success(f"⚡ {provider_name} Online")
        st.caption(f"**Model:** `{active_model}`\n\n**Engine:** `Auto Capability Routing`")
    else:
        st.error(f"❌ {provider_name} Offline. Check GROQ_API_KEY in environment.")


# ── Main Chat Workspace ───────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧠 DocuMind</h1>
    <p>Multimodal Document Intelligence — Single Auto-Routing Engine for PDFs, Math, Tables, Vision & General Knowledge</p>
</div>
""", unsafe_allow_html=True)

conv_docs = conv_manager.get_documents(st.session_state.current_conv_id)
active_docs = current_conv.get("active_docs", [])

# Display Compact Document Cards for Active Conversation
if conv_docs:
    cards_html = "".join([
        f'<div class="doc-card">📄 <b>{d.get("filename", d["doc_id"])}</b> <span style="color:#64748B;">({d.get("page_count", 1)} pgs)</span></div>'
        for d in conv_docs
    ])
    st.markdown(f'<div class="doc-card-container">{cards_html}</div>', unsafe_allow_html=True)
else:
    st.caption("📎 **No files attached to this chat.** Click **'+'** in the bottom composer to attach PDFs or images, or type your question directly.")


# ══════════════════════════════════════════════════════════════════════════════
# PDF PAGE VIEWER COMPONENT
# ══════════════════════════════════════════════════════════════════════════════
_prev_viewer_doc = st.session_state.get("_last_viewer_doc_id")
if st.session_state.preview_doc_id != _prev_viewer_doc:
    st.session_state.preview_page = 1
    if "viewer_page_input" in st.session_state:
        del st.session_state["viewer_page_input"]
    st.session_state["_last_viewer_doc_id"] = st.session_state.preview_doc_id

if st.session_state.requested_preview_page is not None:
    st.session_state["viewer_page_input"] = st.session_state.requested_preview_page
    st.session_state.preview_page = st.session_state.requested_preview_page
    st.session_state.requested_preview_page = None

if st.session_state.show_pdf_preview and st.session_state.preview_doc_id:
    viewer_doc_id = st.session_state.preview_doc_id
    viewer_page = st.session_state.preview_page

    page_dir = f"./data/pdf_pages/{viewer_doc_id}"
    max_pages = 1
    if os.path.isdir(page_dir):
        max_pages = len([f for f in os.listdir(page_dir) if f.endswith(".png")])

    viewer_display_name = viewer_doc_id
    for doc in conv_docs:
        if doc.get("doc_id") == viewer_doc_id:
            viewer_display_name = doc.get("filename", viewer_doc_id)
            break

    st.markdown('<div class="pdf-viewer-box">', unsafe_allow_html=True)
    st.markdown(f"#### 📖 PDF Document Viewer — `{viewer_display_name}`")

    if "viewer_page_input" not in st.session_state:
        st.session_state["viewer_page_input"] = viewer_page
        
    new_page = st.number_input(
        "Jump to page:",
        min_value=1,
        max_value=max(max_pages, 1),
        key="viewer_page_input",
        step=1
    )
    
    if new_page != st.session_state.preview_page:
        st.session_state.preview_page = new_page

    img_path = f"./data/pdf_pages/{viewer_doc_id}/page_{st.session_state.preview_page}.png"
    if os.path.exists(img_path):
        st.image(
            Image.open(img_path),
            caption=f"Page {st.session_state.preview_page}",
            width="stretch"
        )
    elif viewer_doc_id.startswith("img_"):
        st.info("Standalone image preview.")
    else:
        st.warning(f"⚠️ Preview image not found: `{img_path}`.")

    if st.button("✖ Close Viewer", key="close_viewer"):
        st.session_state.show_pdf_preview = False
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()


# ── Chat History Container (Scrollable Box for Messages) ──────────────────────
chat_scroll_box = st.container(height=520)
with chat_scroll_box:
    messages = current_conv.get("messages", [])
    if not messages:
        st.markdown(
            """
            <div style="text-align: center; padding: 40px 20px; color: #64748B;">
                <h3>👋 Welcome to DocuMind!</h3>
                <p>Ask questions about your documents, solve equations, analyze tables, or explore general knowledge.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for msg_idx, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if "metadata" in msg:
                    meta = msg["metadata"]
                    route = meta.get("route", "text_rag")
                    verified = meta.get("verified", False)
                    timings = meta.get("timings", {})
                    sources = meta.get("sources", [])

                    ver_class = "badge-verified-true" if verified else "badge-verified-false"
                    t_str = f"Total: {timings.get('total', 0)}s"

                    st.markdown(
                        f'<span class="badge-route">Route: {route}</span> '
                        f'<span class="{ver_class}">Verified: {verified}</span> '
                        f'<span class="badge-timing">⏱️ {t_str}</span>',
                        unsafe_allow_html=True
                    )

                    if sources:
                        with st.expander("📌 View Citations & Sources"):
                            for src_idx, src in enumerate(sources):
                                p_num = src.get("page_num", 1)
                                c_type = src.get("chunk_type", "text").upper()
                                snippet = src.get("snippet", "")
                                src_doc_id = src.get("doc_id", "")
                                chunk_id = src.get("chunk_id", f"c{src_idx}")
                                filename = src.get("filename", src_doc_id)
                                src_url = src.get("url", "")
                                src_title = src.get("title", "")

                                if c_type == "WEB" or src_doc_id == "web_search":
                                    display_title = src_title or filename
                                    url_link = f'<a href="{src_url}" target="_blank">{src_url}</a>' if src_url else ''
                                    st.markdown(
                                        f'<div class="source-card-web">'
                                        f'<b>🌐 {display_title}</b><br/>'
                                        f'<i>Snippet:</i> {snippet}<br/>'
                                        f'{url_link}'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="source-card">'
                                        f'<b>📄 {filename} - Page {p_num}</b> ({c_type})<br/>'
                                        f'<i>Snippet:</i> {snippet}'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )

                                    btn_key = f"cite_{st.session_state.current_conv_id}_{msg_idx}_{src_idx}_{src_doc_id}_{p_num}_{chunk_id}"
                                    if not src_doc_id.startswith("img_"):
                                        if st.button(f"🔍 Open Page {p_num} in Viewer", key=btn_key):
                                            st.session_state.preview_doc_id = src_doc_id
                                            st.session_state.requested_preview_page = p_num
                                            st.session_state.show_pdf_preview = True
                                            st.rerun()


# ── ChatGPT-Style Permanently Fixed Bottom Chat Composer ──────────────────────
composer_container = st.container()
with composer_container:
    c_plus, c_input = st.columns([0.07, 0.93])
    with c_plus:
        with st.popover("➕", help="Upload PDF, PNG, JPG, JPEG, WEBP to active conversation"):
            st.markdown("#### 📁 Upload File to Active Conversation")
            uploaded_files = st.file_uploader(
                "Upload files",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key=f"composer_uploader_{st.session_state.current_conv_id}",
                label_visibility="collapsed"
            )
            if st.button("Attach to Conversation", key=f"btn_attach_{st.session_state.current_conv_id}", width="stretch"):
                if uploaded_files:
                    for f in uploaded_files:
                        process_uploaded_file(f)
                    st.rerun()
    with c_input:
        question = st.chat_input("Ask a question about your files or any topic...")

if question:
    # 1. Add user message (this automatically persists conversation to History if it was an unpersisted draft)
    user_msg = {"role": "user", "content": question}
    conv_manager.add_message(st.session_state.current_conv_id, user_msg)
    
    # 2. Run workflow
    with st.spinner("Processing request..."):
        wf_output = run_documind_workflow(
            question=question,
            mode="auto",
            conversation_id=st.session_state.current_conv_id,
            active_docs=active_docs
        )

    answer = wf_output["answer"]
    route = wf_output["route"]
    verified = wf_output["verified"]
    sources = wf_output["sources"]
    timings = wf_output.get("timings", {})
    doc_relevance = wf_output.get("doc_relevance", 0.0)

    if route == "general_knowledge":
        sources = []
    elif route == "web_search":
        sources = [s for s in sources if s.get("snippet") or s.get("url")]

    # 3. Save assistant response to conversation history
    asst_msg = {
        "role": "assistant",
        "content": answer,
        "metadata": {
            "route": route,
            "verified": verified,
            "sources": sources,
            "timings": timings
        }
    }
    conv_manager.add_message(st.session_state.current_conv_id, asst_msg)

    # 4. Rerun so ALL messages render inside chat_scroll_box ABOVE the composer
    st.rerun()
