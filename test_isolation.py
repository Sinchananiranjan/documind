import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.rag.vector_store import VectorStoreManager
from data.generate_sample import generate_sample_pdf
import tempfile
from reportlab.pdfgen import canvas
from app.document.pdf_processor import PDFProcessor

def generate_dbms_pdf():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    c = canvas.Canvas(path)
    c.drawString(100, 750, "DBMS Module 2")
    c.drawString(100, 730, "What is the relational model?")
    c.drawString(100, 710, "The relational model represents data as relations or tables.")
    c.drawString(100, 690, "What is the degree or arity of a relation?")
    c.drawString(100, 670, "The degree or arity of a relation is the number of attributes it contains.")
    c.save()
    return path

def main():
    print("Initializing...")
    vsm = VectorStoreManager()
    processor = PDFProcessor()
    
    # Process sample communication PDF
    print("Processing Sample Communication PDF...")
    sample_path = generate_sample_pdf()
    res1 = processor.process_pdf(sample_path)
    vsm.add_document_chunks(res1["chunks"])
    doc1_id = res1["summary"]["doc_id"]
    print(f"Sample PDF ID: {doc1_id}")
    
    # Process DBMS PDF
    print("Processing DBMS PDF...")
    dbms_path = generate_dbms_pdf()
    res2 = processor.process_pdf(dbms_path)
    vsm.add_document_chunks(res2["chunks"])
    doc2_id = res2["summary"]["doc_id"]
    print(f"DBMS PDF ID: {doc2_id}")
    
    # Test Isolation
    print("\n--- Testing Retrieval Isolation ---")
    
    q1 = "What is the relational model?"
    print(f"Query: '{q1}' on Document 1 (Communication)")
    chunks1 = vsm.search_similarity(q1, doc_id=doc1_id)
    for c in chunks1:
        print(f" - {c['doc_id']}: {c['snippet']}")
        assert c['doc_id'] == doc1_id, "CROSS-DOCUMENT LEAKAGE!"
        
    print(f"Query: '{q1}' on Document 2 (DBMS)")
    chunks2 = vsm.search_similarity(q1, doc_id=doc2_id)
    for c in chunks2:
        print(f" - {c['doc_id']}: {c['snippet']}")
        assert c['doc_id'] == doc2_id, "CROSS-DOCUMENT LEAKAGE!"

    print("\nIsolation Test Passed Successfully!")
    
if __name__ == "__main__":
    main()
