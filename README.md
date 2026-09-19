# DocuMind – Multimodal Document Analyzer

DocuMind is an open-source, 100% local AI application designed to analyze multimodal PDF documents containing **text, scanned OCR pages, tables, and images**. 

Built with **LangGraph**, **LangChain**, **FastAPI**, **Streamlit**, **PyMuPDF**, **Tesseract OCR**, **ChromaDB**, **Sentence Transformers**, and **Ollama**, DocuMind requires zero paid API keys and keeps all data locally on your computer.

---

## 🏗️ Tech Stack & Architecture

- **Orchestration:** LangGraph (`StateGraph`, typed state, conditional routing, hallucination verification, automated retry loop)
- **Framework & Models:** LangChain, SentenceTransformers (`all-MiniLM-L6-v2`), Ollama (`qwen2.5:3b` default, `qwen2.5:1.5b` fallback, `moondream` optional vision)
- **Document Processing:** PyMuPDF (`fitz`), PyTesseract OCR, PyMuPDF `find_tables()`
- **Vector Database:** ChromaDB (persistent local storage with metadata filtering)
- **Backend & Frontend:** FastAPI (REST API), Streamlit (Interactive Web Dashboard)
- **Dependency Management:** `uv`

### 🔄 LangGraph Workflow Diagram

```text
               [User Question + Doc ID]
                          │
                   (retrieve_node)
                          │  (fetches top-k chunks from ChromaDB)
                    (router_node)
                     ╱    │    ╲
                    ╱     │     ╲  (conditional route based on query & context)
                   v      v      v
           (text_rag) (table) (image)
                   ╲      │      ╱
                    ╲     │     ╱
                     v    v    v
                    (verify_answer) 
                       ╱     ╲
                      ╱       ╲
            (is_grounded?)     (unverified / retries < 2)
                 │                    │
                 v                    v
               [END]               (router)
                                      │  (if max retries reached)
                                      v
                                  (fallback) ──> [END]
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Windows OS
- Python 3.10+
- `uv` package manager (`pip install uv` or `winget install astral-sh.uv`)
- Ollama installed and running on Windows (`ollama --version`)
- Tesseract OCR installed (default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

### 2. Install Project Dependencies with `uv`
Run inside the project root directory:

```bash
uv sync
```

### 3. Pull Local Ollama LLM Model
Ensure Ollama service is running, then pull the lightweight `qwen2.5:3b` model (or `qwen2.5:1.5b` fallback):

```bash
ollama pull qwen2.5:3b
```

*(Optional for Vision analysis)*:
```bash
ollama pull moondream
```

---

## 🚀 Running DocuMind

### Option A: Launch Streamlit App Directly (Recommended)
You can run the Streamlit frontend directly without needing to launch FastAPI manually:

```bash
uv run streamlit run frontend/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Launch FastAPI Backend Server
If you wish to interact via REST API or Swagger UI:

```bash
uv run uvicorn app.api.main:app --reload --port 8000
```
Open Swagger docs at `http://127.0.0.1:8000/docs`.

---

## 🧪 Testing & Data Generation

### 1. Generate Sample PDF
Generate a multi-page test PDF containing text sections, a 4x4 benchmark table, and vector diagrams:

```bash
uv run python data/generate_sample.py
```
This generates `data/sample_document.pdf`. You can also generate this directly inside Streamlit using the sidebar button!

### 2. Run Automated Test Suite
Run the full pytest suite covering PDF processing, ChromaDB indexing, and LangGraph workflow state execution:

```bash
uv run pytest
```

---

## 💻 Hardware Optimizations for Windows (8GB RAM / Intel i3)

1. **Lightweight Embeddings:** Uses `all-MiniLM-L6-v2` (~90MB RAM footprint on CPU).
2. **Quantized Local LLM:** Default model `qwen2.5:3b` (~1.9GB RAM usage), smoothly running on 8GB RAM systems.
3. **Graceful Vision Fallback:** If `moondream` vision model is not installed, image queries seamlessly fall back to OCR text extraction and image metadata descriptions without crashing.
4. **Hallucination Protection:** If context is insufficient, the system returns: `"I couldn't find this in the document."`
