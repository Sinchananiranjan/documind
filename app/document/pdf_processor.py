"""Main PDF Processor integrating PyMuPDF, OCR, tables, and page preview rendering."""

import fitz  # PyMuPDF
import os
import hashlib
import logging
from typing import List, Dict, Any

from app.document.ocr import run_ocr_on_pixmap
from app.document.table_extractor import extract_tables_from_page
from app.document.image_extractor import extract_images_from_page

logger = logging.getLogger(__name__)

CACHE_PAGE_DIR = "./data/pdf_pages"

class PDFProcessor:
    """Processes PDF documents during upload to extract text, tables, images, and render page previews."""

    def __init__(self, low_text_threshold: int = 50):
        self.low_text_threshold = low_text_threshold
        os.makedirs(CACHE_PAGE_DIR, exist_ok=True)

    def generate_doc_id(self, file_path: str, original_filename: Optional[str] = None) -> str:
        """Generate a deterministic document ID based on filename and size."""
        file_name = original_filename or os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 100
        content_hash = hashlib.md5(f"{file_name}_{file_size}".encode()).hexdigest()[:10]
        clean_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(file_name)[0])
        return f"{clean_name}_{content_hash}"

    def render_and_save_pages(self, doc, doc_id: str) -> str:
        """Render each page of PDF to PNG for instant preview rendering in frontend."""
        doc_page_dir = os.path.join(CACHE_PAGE_DIR, doc_id)
        os.makedirs(doc_page_dir, exist_ok=True)
        
        for i, page in enumerate(doc):
            page_num = i + 1
            img_path = os.path.join(doc_page_dir, f"page_{page_num}.png")
            if not os.path.exists(img_path):
                pix = page.get_pixmap(dpi=150)
                pix.save(img_path)
        
        return doc_page_dir

    def process_pdf(self, file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Process PDF during upload: extract text, OCR, tables, images, and render preview pages.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")

        doc_id = self.generate_doc_id(file_path, original_filename=original_filename)
        filename = original_filename or os.path.basename(file_path)
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        # Render PNG previews for interactive clickable citations
        page_dir = self.render_and_save_pages(doc, doc_id)

        document_chunks = []
        scanned_page_count = 0
        table_count = 0
        image_count = 0

        logger.info(f"Processing PDF '{file_path}' (Doc ID: {doc_id}, Pages: {total_pages})")

        chunk_counter = 1
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Direct text extraction
            page_text = page.get_text("text").strip()

            # 2. Check for scanned / low-text page -> Run OCR
            is_scanned = len(page_text) < self.low_text_threshold
            if is_scanned:
                scanned_page_count += 1
                logger.info(f"Page {page_num} detected as scanned. Running OCR...")
                pix = page.get_pixmap(dpi=150)
                ocr_text = run_ocr_on_pixmap(pix)
                if ocr_text:
                    document_chunks.append({
                        "doc_id": doc_id,
                        "filename": filename,
                        "page_num": page_num,
                        "chunk_type": "ocr",
                        "chunk_id": f"{doc_id}_p{page_num}_c{chunk_counter}",
                        "content": f"[Page {page_num} OCR Scanned Text]:\n{ocr_text}"
                    })
                    chunk_counter += 1
            
            if page_text:
                document_chunks.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "chunk_type": "text",
                    "chunk_id": f"{doc_id}_p{page_num}_c{chunk_counter}",
                    "content": f"[Page {page_num} Text]:\n{page_text}"
                })
                chunk_counter += 1

            # 3. Table extraction
            tables = extract_tables_from_page(page, page_num)
            for tab in tables:
                table_count += 1
                document_chunks.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "chunk_type": "table",
                    "chunk_id": f"{doc_id}_p{page_num}_tab{tab['table_id']}",
                    "content": f"[Page {page_num} Table {tab['table_id']}]:\n{tab['markdown']}"
                })
                chunk_counter += 1

            # 4. Image extraction
            images = extract_images_from_page(doc, page, page_num)
            for img in images:
                image_count += 1
                document_chunks.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "chunk_type": "image",
                    "chunk_id": f"{doc_id}_p{page_num}_{img['image_id']}",
                    "content": f"[Page {page_num} Image]: {img['description']}",
                    "image_b64": img["base64"]
                })
                chunk_counter += 1

        doc.close()

        summary = {
            "doc_id": doc_id,
            "filename": filename,
            "pdf_path": os.path.abspath(file_path),
            "page_preview_dir": page_dir,
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
