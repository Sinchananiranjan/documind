"""Vector store manager using persistent ChromaDB and LangChain documents."""

import os
import logging
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "documind_collection"


class VectorStoreManager:
    """Manages document chunking, indexing, and persistent storage in ChromaDB."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_function = get_embedding_model()
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_dir
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def add_document_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Takes raw extracted chunks from PDFProcessor, splits long texts,
        attaches metadata (doc_id, page_num, chunk_type), and stores them in ChromaDB.
        """
        documents_to_add = []

        for chunk in chunks:
            doc_id = chunk["doc_id"]
            page_num = chunk["page_num"]
            chunk_type = chunk["chunk_type"]
            content = chunk["content"]

            # For tables or short image descriptions, keep intact without splitting
            if chunk_type in ["table", "image"]:
                doc = Document(
                    page_content=content,
                    metadata={
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "chunk_type": chunk_type,
                        "image_b64": chunk.get("image_b64", "")
                    }
                )
                documents_to_add.append(doc)
            else:
                # Text / OCR chunks are split into manageable RAG pieces
                sub_chunks = self.text_splitter.split_text(content)
                for sub in sub_chunks:
                    doc = Document(
                        page_content=sub,
                        metadata={
                            "doc_id": doc_id,
                            "page_num": page_num,
                            "chunk_type": chunk_type,
                            "image_b64": ""
                        }
                    )
                    documents_to_add.append(doc)

        if documents_to_add:
            self.vector_store.add_documents(documents_to_add)
            logger.info(f"Added {len(documents_to_add)} vector chunks to ChromaDB.")
        
        return len(documents_to_add)

    def search_similarity(
        self,
        query: str,
        doc_id: Optional[str] = None,
        k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in ChromaDB filtered by doc_id and optionally chunk_type.
        Returns formatted list of matching chunks with metadata.
        """
        where_clause = {}
        if doc_id:
            where_clause["doc_id"] = doc_id
        if filter_type:
            where_clause["chunk_type"] = filter_type

        kwargs = {"k": k}
        if where_clause:
            kwargs["filter"] = where_clause

        try:
            results = self.vector_store.similarity_search_with_score(query, **kwargs)
        except Exception as e:
            logger.warning(f"Similarity search failed with filter {where_clause}: {e}. Retrying without metadata filter...")
            results = self.vector_store.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "page_num": doc.metadata.get("page_num", 1),
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "chunk_type": doc.metadata.get("chunk_type", "text"),
                "content": doc.page_content,
                "score": float(score),
                "image_b64": doc.metadata.get("image_b64", "")
            })

        return formatted_results

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks associated with doc_id from ChromaDB."""
        try:
            collection = self.vector_store._collection
            collection.delete(where={"doc_id": doc_id})
            logger.info(f"Deleted doc_id '{doc_id}' from vector store.")
            return True
        except Exception as e:
            logger.error(f"Error deleting doc_id '{doc_id}': {e}")
            return False

    def list_indexed_documents(self) -> List[str]:
        """Get unique list of doc_ids stored in ChromaDB."""
        try:
            collection = self.vector_store._collection
            metadata_list = collection.get(include=["metadatas"])["metadatas"]
            doc_ids = set(m.get("doc_id") for m in metadata_list if m and "doc_id" in m)
            return list(doc_ids)
        except Exception as e:
            logger.warning(f"Could not list documents: {e}")
            return []
