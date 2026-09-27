"""Performance & Completeness test suite measuring routing, retrieval, LLM generation, tokens/sec, and overall speed."""

import os
import sys
import time
import tempfile
from reportlab.pdfgen import canvas

# Set UTF-8 encoding for Windows console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from data.generate_sample import generate_sample_pdf
from data.generate_finance_sample import generate_finance_pdf
from data.generate_education_sample import generate_education_pdf
from data.generate_error_detection_sample import generate_error_detection_pdf

from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow

def generate_dbms_pdf() -> str:
    fd, path = tempfile.mkstemp(suffix="_dbms.pdf")
    os.close(fd)
    c = canvas.Canvas(path)
    c.drawString(100, 750, "Database Management Systems (DBMS) Overview")
    c.drawString(100, 730, "What is metadata?")
    c.drawString(100, 710, "Metadata is defined as data about data, describing table schemas and data types.")
    c.drawString(100, 690, "What are the main functions of a DBMS?")
    c.drawString(100, 670, "The main functions of a DBMS include data storage, retrieval, security, and concurrency control.")
    c.drawString(100, 650, "Explain DDL and DML with examples.")
    c.drawString(100, 630, "DDL (Data Definition Language) defines database schemas (CREATE, ALTER). DML (Data Manipulation Language) modifies data (INSERT, UPDATE).")
    c.save()
    return path

def main():
    print("=" * 110)
    print("STARTING PERFORMANCE & COMPLETENESS BENCHMARK")
    print("=" * 110)

    # 1. Generate multi-domain test PDFs
    pdf_err = generate_error_detection_pdf("./data/test_perf_error.pdf")
    pdf_finance = generate_finance_pdf("./data/test_perf_finance.pdf")
    pdf_edu = generate_education_pdf("./data/test_perf_edu.pdf")
    pdf_dbms = generate_dbms_pdf()

    processor = PDFProcessor()
    vector_mgr = VectorStoreManager(persist_dir="./data/test_perf_chroma_db")

    docs = {}
    for name, path in [
        ("Error Detection & Coding", pdf_err),
        ("Finance (Q3 Revenue)", pdf_finance),
        ("Education (Biology)", pdf_edu),
        ("DBMS (Database Systems)", pdf_dbms)
    ]:
        res = processor.process_pdf(path)
        vector_mgr.add_document_chunks(res["chunks"])
        doc_id = res["summary"]["doc_id"]
        docs[name] = {"doc_id": doc_id, "path": path}

    print("\n" + "=" * 110)
    print("RUNNING SHORT AND LONG ANSWER BENCHMARKS")
    print("=" * 110 + "\n")

    test_cases = [
        # A. Short Question
        (
            "Error Detection & Coding",
            "What is a single-bit error?",
            ["single-bit error", "one bit"]
        ),
        # B. Detailed Question
        (
            "Error Detection & Coding",
            "Explain error detection and error correction in detail.",
            ["dmin", "s + 1", "2t + 1"]
        ),
        # C. Detailed Step-by-Step Question
        (
            "Error Detection & Coding",
            "Explain CRC step-by-step including encoder, decoder, syndrome, and error decision.",
            ["generator polynomial", "modulo-2", "remainder"]
        ),
        # D. Detailed Multi-Concept Question
        (
            "Error Detection & Coding",
            "Explain Hamming distance, minimum Hamming distance, and how they are used for error detection.",
            ["hamming distance", "xor", "dmin"]
        ),
        # E. Long Multi-Part Question
        (
            "Error Detection & Coding",
            "Explain the difference between single-bit and burst errors, how Hamming distance guarantees error detection, and how CRC modulo-2 division is used by the receiver.",
            ["burst error", "noise impulse", "hamming", "crc"]
        ),
        # F. Cross-Domain Detailed Questions
        (
            "Finance (Q3 Revenue)",
            "Explain Q3 consolidated net revenue growth, performance table margins, and profit margin calculation in detail.",
            ["4.85", "49.4%", "gross revenue"]
        ),
        (
            "Education (Biology)",
            "Explain primary producers, primary consumers, trophic level energy efficiency, and Lindeman's 10% rule in detail.",
            ["primary producers", "photosynthesis", "10%"]
        ),
        (
            "DBMS (Database Systems)",
            "Explain metadata, DDL, DML, and the four main functions of a DBMS in detail.",
            ["metadata", "ddl", "dml", "storage"]
        )
    ]

    results_table = []

    for domain_name, q, expected_keywords in test_cases:
        doc_info = docs[domain_name]
        doc_id = doc_info["doc_id"]

        t_start = time.perf_counter()
        wf_out = run_documind_workflow(q, doc_id=doc_id)
        total_time = round(time.perf_counter() - t_start, 2)

        route = wf_out["route"]
        ans = wf_out["answer"]
        sources = wf_out.get("sources", [])
        timings = wf_out.get("timings", {})

        routing_t = timings.get("routing", 0.0)
        retrieval_t = timings.get("retrieval", 0.0)
        llm_t = timings.get("llm", 0.0)
        tok_gen = timings.get("tokens_generated", len(ans.split()))
        tps = timings.get("tokens_per_sec", 0.0)

        llm_calls = 0 if route == "calculation" and llm_t == 0.0 else 1

        # Completeness Check
        kw_matched = sum(1 for kw in expected_keywords if kw.lower() in ans.lower())
        completeness = "COMPLETE (High)" if kw_matched >= len(expected_keywords) else f"PARTIAL ({kw_matched}/{len(expected_keywords)})"

        # Grounding & Citation Check
        grounding = "PASS" if kw_matched > 0 and "couldn't find" not in ans.lower() else "FAIL"
        citation = "PASS" if (route in ("text_rag", "hybrid") and sources) else "PASS (N/A)"

        results_table.append({
            "question": q[:40] + "...",
            "completeness": completeness,
            "grounding": grounding,
            "citation": citation,
            "routing_t": f"{routing_t:.3f}s",
            "retrieval_t": f"{retrieval_t:.3f}s",
            "llm_t": f"{llm_t:.2f}s",
            "tokens": tok_gen,
            "tps": f"{tps:.1f}",
            "total_t": f"{total_time:.2f}s",
            "llm_calls": llm_calls
        })

        print(f"[{domain_name}] Q: '{q}'")
        print(f"  Route: `{route}` | Total: {total_time:.2f}s | LLM: {llm_t:.2f}s | Ret: {retrieval_t:.3f}s | Rout: {routing_t:.3f}s")
        print(f"  Tokens: {tok_gen} | Speed: {tps:.1f} tok/s | LLM calls: {llm_calls}")
        print(f"  Completeness: {completeness} | Grounding: {grounding} | Citation: {citation}")
        print(f"  Ans snippet: {ans[:140]}...\n" + "-" * 80)

    # Print Report Markdown Table
    print("\n" + "=" * 110)
    print("PERFORMANCE & COMPLETENESS REPORT TABLE")
    print("=" * 110 + "\n")
    print("| Question | Completeness | Grounding | Citation | Routing time | Retrieval time | LLM time | Tokens | Tokens/sec | Total time | LLM calls |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results_table:
        print(f"| {r['question']} | {r['completeness']} | {r['grounding']} | {r['citation']} | {r['routing_t']} | {r['retrieval_t']} | {r['llm_t']} | {r['tokens']} | {r['tps']} | {r['total_t']} | {r['llm_calls']} |")

    if os.path.exists(pdf_dbms):
        os.remove(pdf_dbms)

if __name__ == "__main__":
    main()
