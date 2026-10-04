# DocuMind – Multimodal Document Analyzer


DocuMind is an open-source, 100% local AI application designed to analyze multimodal PDF documents containing **text, scanned OCR pages, tables, and images**. 

Optimized specifically for Windows systems with **8GB RAM, Intel i3 CPU, and zero GPU requirements**, DocuMind requires zero paid API keys and keeps all data locally on your computer.

---

## 🚀 Key Improvements & Features

- **⚡ Fast Python Heuristic Router:** 0ms extra LLM latency for query classification (routes to `text_rag`, `table_analysis`, or `image_analysis`).
- **🧮 Numerical & Formula Reasoning:** Handles mathematical calculations (e.g. SNR = $20 \log_{10}(S/N)$, $3.5\text{ mV} / 0.75\text{ mV} \approx 13.38\text{ dB}$) step-by-step using retrieved document values.
- **📐 LaTeX Math Rendering:** Automatically formats math equations using standard LaTeX syntax.
- **📄 Interactive Clickable Page Citations:** Click **"🔍 Click to open Page X in Viewer"** in citations to instantly open and view the exact rendered PDF page inside an interactive in-app page viewer.
- **🛡️ Hallucination Verification Bug Fix:** Fallback responses (*"I couldn't find this in the document."*) correctly report `Grounded Verified: False`.
- **⏱️ Performance Instrumentation:** Displays execution timing breakdown for retrieval, routing, LLM generation, and verification.
- **🤖 Lightweight Model Stack:** Defaults to `qwen2.5:1.5b` (1.1GB RAM) with `all-MiniLM-L6-v2` embeddings (~90MB RAM on CPU).

---

## 🔄 LangGraph Workflow Topology

```text
               [User Question + Doc ID]
                          │
                   (retrieve_node)       ──> [Top 3-5 Chunks from ChromaDB]
                          │
              (router_node - Fast Python)──> [0ms Classification]
                     ╱    │    ╲
                    ╱     │     ╲
                   v      v      v
           (text_rag) (table) (image)
                   ╲      │      ╱
                    ╲     │     ╱
                     v    v    v
                    (verify_answer) 
                       ╱     ╲
                      ╱       ╲
             (is_grounded)    (fallback / ungrounded)
                 │                    │
                 v                    v
               [END]               (fallback) ──> [END]
```

---

## ⚙️ Installation & Setup

### 1. Install Dependencies with `uv`
```bash
uv sync
```

### 2. Configure Groq API Key
Copy `.env.example` to `.env` and set your Groq API Key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running DocuMind

### 1. Launch Streamlit Web UI (Recommended)
```bash
uv run streamlit run frontend/streamlit_app.py
```
Open browser at `http://localhost:8501`.

### 2. Launch FastAPI Backend
```bash
uv run uvicorn app.api.main:app --reload --port 8000
```
Open API docs at `http://127.0.0.1:8000/docs`.

---

## 🧪 Testing

### Run Full Pytest Suite
```bash
uv run pytest
```
*Executes tests for Ollama detection, PDF parsing, ChromaDB indexing, fast routing, numerical formula calculation, and fallback verification.*
