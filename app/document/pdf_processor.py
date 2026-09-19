"""Main PDF Processor integrating PyMuPDF, OCR, tables, and image extraction."""

import fitz  # PyMuPDF
import os
import hashlib
import logging
from typing import List, Dict, Any

from app.document.ocr import run_ocr_on_pixmap
from app.document.table_extractor import extract_tables_from_page
from app.document.image_extractor import extract_images_from_page

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processes PDF documents to extract text, tables, images, and scanned OCR content."""

    def __init__(self, low_text_threshold: int = 50):
        self.low_text_threshold = low_text_threshold

    def generate_doc_id(self, file_path: str) -> str:
        """Generate a deterministic document ID based on filename and size."""
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        content_hash = hashlib.md5(f"{file_name}_{file_size}".encode()).hexdigest()[:10]
        clean_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(file_name)[0])
        return f"{clean_name}_{content_hash}"

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process PDF and return document metadata + structured pages/chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")

        doc_id = self.generate_doc_id(file_path)
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        document_chunks = []
        scanned_page_count = 0
        table_count = 0
        image_count = 0

        logger.info(f"Processing PDF '{file_path}' (Doc ID: {doc_id}, Pages: {total_pages})")

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Direct text extraction
            page_text = page.get_text("text").strip()

            # 2. Check for scanned / low-text page -> Run OCR
            is_scanned = len(page_text) < self.low_text_threshold
            if is_scanned:
                scanned_page_count += 1
                logger.info(f"Page {page_num} detected as scanned/low-text. Running OCR...")
                pix = page.get_pixmap(dpi=150)
                ocr_text = run_ocr_on_pixmap(pix)
                if ocr_text:
                    document_chunks.append({
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "chunk_type": "ocr",
                        "content": f"[Page {page_num} Scanned Text (OCR)]:\n{ocr_text}"
                    })
            
            if page_text:
                document_chunks.append({
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "chunk_type": "text",
                    "content": f"[Page {page_num} Text]:\n{page_text}"
                })

            # 3. Table extraction
            tables = extract_tables_from_page(page, page_num)
            for tab in tables:
                table_count += 1
                document_chunks.append({
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "chunk_type": "table",
                    "content": f"[Page {page_num} Table {tab['table_id']}]:\n{tab['markdown']}"
                })

            # 4. Image extraction
            images = extract_images_from_page(doc, page, page_num)
            for img in images:
                image_count += 1
                document_chunks.append({
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "chunk_type": "image",
                    "content": f"[Page {page_num} Image]: {img['description']}",
                    "image_b64": img["base64"]
                })

        doc.close()

        summary = {
            "doc_id": doc_id,
            "filename": os.path.basename(file_path),
            "total_pages": total_pages,
            "scanned_pages": scanned_page_count,
            "tables_found": table_count,
            "images_found": image_count,
            "total_chunks": len(document_chunks)
        }
        
        logger.info(f"Processing complete for {doc_id}: {summary}")

        return {
            "summary": summary,
            "chunks": document_chunks
        }
