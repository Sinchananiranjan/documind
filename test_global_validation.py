"""Final global validation test suite verifying all 8 requirements of the final task prompt."""

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
from data.generate_or_sample import generate_or_pdf

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
    c.save()
    return path

def main():
    print("=" * 115)
    print("STARTING FINAL GLOBAL VALIDATION SUITE")
    print("=" * 115)

    # 1. Generate multi-domain test PDFs
    pdf_or = generate_or_pdf("./data/test_benchmark_or.pdf")
    pdf_err = generate_error_detection_pdf("./data/test_benchmark_error.pdf")
    pdf_finance = generate_finance_pdf("./data/test_benchmark_finance.pdf")
    pdf_edu = generate_education_pdf("./data/test_benchmark_edu.pdf")
    pdf_dbms = generate_dbms_pdf()

    processor = PDFProcessor()
    vector_mgr = VectorStoreManager(persist_dir="./data/test_benchmark_chroma_db")

    # 2. Index all PDFs
    docs = {}
    for name, path in [
        ("Operations Research & LP", pdf_or),
        ("Error Detection & Coding", pdf_err),
        ("Finance (Q3 Revenue)", pdf_finance),
        ("Education (Biology)", pdf_edu),
        ("DBMS (Database Systems)", pdf_dbms)
    ]:
        res = processor.process_pdf(path)
        vector_mgr.add_document_chunks(res["chunks"])
        doc_id = res["summary"]["doc_id"]
        docs[name] = {"doc_id": doc_id, "path": path}
        print(f"Indexed domain '{name}' -> doc_id: {doc_id} ({len(res['chunks'])} chunks)")

    print("\n" + "=" * 115)
    print("RUNNING BENCHMARK TEST SUITE")
    print("=" * 115 + "\n")

    test_cases = [
        # 1. Calculation Regression Tests (A, B, C)
        (
            "Operations Research & LP",
            "If x1=5 and x2=3, calculate Z=2x1+3x2",
            "document_mode",
            "calculation",
            ["19"],
        ),
        (
            "Operations Research & LP",
            "For Max Z=4x1+3x2, calculate Z at (4.5,2)",
            "document_mode",
            "calculation",
            ["24"],
        ),
        (
            "Operations Research & LP",
            "For Min Z=10x1+4x2, calculate Z at (6,21)",
            "document_mode",
            "calculation",
            ["144"],
        ),

        # 2. Long Answer Tests (1, 2, 3)
        (
            "Error Detection & Coding",
            "Explain Hamming distance in detail, including how it is calculated, its relationship with minimum Hamming distance, and how it is used for error detection and correction.",
            "document_mode",
            "long_answer",
            ["Hamming distance", "dmin", "detect", "correct"],
        ),
        (
            "Error Detection & Coding",
            "Explain error detection and error correction, including single-bit errors, burst errors, redundancy, coding, and the role of the receiver.",
            "document_mode",
            "long_answer",
            ["single-bit", "burst", "redundancy", "receiver"],
        ),
        (
            "Operations Research & LP",
            "Explain the complete phases of an Operations Research study.",
            "document_mode",
            "long_answer",
            ["Problem Formulation", "Mathematical Model", "Algorithm Execution", "Validation", "Implementation"],
        ),

        # 3. Cross-Domain Generalization Tests
        (
            "Operations Research & LP",
            "Why is (8,4) optimal in Max Z=80x1+55x2?",
            "document_mode",
            "domain_test",
            ["860", "highest", "corner point"],
        ),
        (
            "Operations Research & LP",
            "Explain the special cases in the graphical method with examples.",
            "document_mode",
            "domain_test",
            ["multiple optimal", "no optimal", "unbounded"],
        ),
        (
            "Operations Research & LP",
            "Does the document discuss neural networks for Operations Research?",
            "document_mode",
            "absence_test",
            ["does not provide", "does not mention", "not mention"],
        ),
        (
            "Error Detection & Coding",
            "What is CRC?",
            "document_mode",
            "domain_test",
            ["Cyclic Redundancy Check", "polynomial"],
        ),
        (
            "Error Detection & Coding",
            "How is Hamming distance different from syndrome?",
            "document_mode",
            "domain_test",
            ["differ", "syndrome", "vector"],
        ),
        (
            "Finance (Q3 Revenue)",
            "How much consolidated net revenue was generated in Q3 2026?",
            "document_mode",
            "domain_test",
            ["4.85"],
        ),
        (
            "Education (Biology)",
            "What energy level do primary producers have in the PDF, and how does energy flow in ecosystems generally?",
            "document_mode",
            "domain_test",
            ["10,000", "kcal"],
        ),
        (
            "DBMS (Database Systems)",
            "What is metadata and what are the main functions of a DBMS?",
            "document_mode",
            "domain_test",
            ["data about data", "data storage"],
        ),
        (
            "Operations Research & LP",
            "What are the rules of cricket?",
            "general_knowledge_mode",
            "gk_test",
            ["General Knowledge", "cricket"],
        ),
    ]

    results_table = []
    all_passed = True

    for idx, (domain_name, q, mode, category, expected_keywords) in enumerate(test_cases, 1):
        doc_info = docs[domain_name]
        doc_id = doc_info["doc_id"]

        t_start = time.perf_counter()
        wf_out = run_documind_workflow(q, doc_id=doc_id, mode=mode)
        total_time = round(time.perf_counter() - t_start, 2)

        route = wf_out["route"]
        ans = wf_out["answer"]
        verified = wf_out.get("verified", False)
        sources = wf_out.get("sources", [])
        timings = wf_out.get("timings", {})
        retrieval_time = timings.get("retrieval", 0.0)
        gen_time = timings.get("llm", 0.0)
        tokens_gen = timings.get("tokens_generated", 0)
        tokens_per_sec = timings.get("tokens_per_sec", 0.0)

        llm_calls = 0 if route == "calculation" and gen_time == 0.0 else 1

        pages_retrieved = sorted({s["page_num"] for s in sources}) if sources else []
        pages_str = ", ".join(f"Page {p}" for p in pages_retrieved) if pages_retrieved else "None"

        kw_match = any(kw.lower() in ans.lower() for kw in expected_keywords)
        is_absence = "does not provide" in ans.lower() or "does not mention" in ans.lower() or "not mention" in ans.lower() or "couldn't find" in ans.lower() or "could not find" in ans.lower()
        
        passed = (kw_match or (category == "absence_test" and is_absence))
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"

        ans_clean = ans.replace('\n', ' ').strip()
        ans_snippet = ans_clean[:120] + ("..." if len(ans_clean) > 120 else "")

        results_table.append({
            "idx": idx,
            "category": category,
            "domain": domain_name,
            "question": q,
            "route": route,
            "verified": verified,
            "pages": pages_str,
            "answer_snippet": ans_snippet,
            "status": status_str,
            "llm_calls": llm_calls,
            "t_retrieval": f"{retrieval_time}s",
            "t_gen": f"{gen_time}s",
            "tokens": tokens_gen,
            "tok_per_sec": f"{tokens_per_sec:.1f}",
            "t_total": f"{total_time}s"
        })

        print(f"[{idx}/{len(test_cases)}] [{category.upper()}] [{domain_name}] Query: '{q}'")
        print(f"  Route: `{route}` | Verified: {verified} | Status: {status_str} | Total: {total_time}s (Ret: {retrieval_time}s, Gen: {gen_time}s) | LLM calls: {llm_calls}")
        print(f"  Ans: {ans_snippet}")
        print("-" * 90)

    # Print Final Markdown Benchmark Report Table
    print("\n" + "=" * 115)
    print("FINAL VERIFICATION BENCHMARK REPORT TABLE")
    print("=" * 115 + "\n")
    print("| # | Category | Document | Question | Route | Verified | Pages | Answer Snippet | Status | LLM calls | Retrieval | Generation | Tokens | Tok/sec | Total |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results_table:
        print(f"| {r['idx']} | `{r['category']}` | {r['domain']} | {r['question']} | `{r['route']}` | {r['verified']} | {r['pages']} | {r['answer_snippet']} | {r['status']} | {r['llm_calls']} | {r['t_retrieval']} | {r['t_gen']} | {r['tokens']} | {r['tok_per_sec']} | {r['t_total']} |")

    if os.path.exists(pdf_dbms):
        os.remove(pdf_dbms)

    print("\n" + "=" * 115)
    if all_passed:
        print("RESULT: VALIDATION PASSED")
    else:
        print("RESULT: VALIDATION FAILED")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
