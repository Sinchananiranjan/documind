"""FastAPI route definitions for DocuMind API."""

import os
import shutil
import tempfile
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel

from app.document.pdf_processor import PDFProcessor
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow

logger = logging.getLogger(__name__)

router = APIRouter()
pdf_processor = PDFProcessor()
vector_manager = VectorStoreManager()


class QueryRequest(BaseModel):
    doc_id: str
    question: str


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        result = pdf_processor.process_pdf(temp_path)
        summary = result["summary"]
        chunks = result["chunks"]

        # Index in ChromaDB
        indexed_count = vector_manager.add_document_chunks(chunks)

        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}'",
            "doc_id": summary["doc_id"],
            "summary": summary,
            "indexed_chunks": indexed_count
        }

    except Exception as e:
        logger.error(f"Error processing PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/query")
async def query_document(payload: QueryRequest):
    """Execute question against uploaded document via LangGraph workflow."""
    if not payload.doc_id or not payload.question:
        raise HTTPException(status_code=400, detail="Both 'doc_id' and 'question' are required.")

    try:
        response = run_documind_workflow(question=payload.question, doc_id=payload.doc_id)
        return {
            "status": "success",
            "data": response
        }
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.get("/documents")
async def list_documents():
    """List indexed document IDs in vector store."""
    docs = vector_manager.list_indexed_documents()
    return {"status": "success", "documents": docs}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document from vector store."""
    success = vector_manager.delete_document(doc_id)
    if success:
        return {"status": "success", "message": f"Document '{doc_id}' deleted."}
    else:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found or delete failed.")
