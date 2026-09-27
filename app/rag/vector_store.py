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

_vector_store_instance = None


class VectorStoreManager:
    """Manages document chunking, indexing, and persistent storage in ChromaDB (loaded once)."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        global _vector_store_instance
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_function = get_embedding_model()
        
        if _vector_store_instance is None:
            logger.info("Initializing ChromaDB persistent collection...")
            _vector_store_instance = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_dir
            )
        self.vector_store = _vector_store_instance

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=550,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def add_document_chunks(self, chunks: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> int:
        """
        Splits text chunks while strictly preserving metadata:
        conversation_id, doc_id, filename, page_num, chunk_type, chunk_id.
        """
        documents_to_add = []

        for chunk in chunks:
            doc_id = chunk["doc_id"]
            conv_id = conversation_id or chunk.get("conversation_id", "")
            filename = chunk.get("filename", "")
            page_num = chunk["page_num"]
            chunk_type = chunk["chunk_type"]
            chunk_id = chunk.get("chunk_id", f"{doc_id}_p{page_num}")
            content = chunk["content"]
            upload_order = chunk.get("upload_order", 1)

            if chunk_type in ["table", "image"]:
                doc = Document(
                    page_content=content,
                    metadata={
                        "doc_id": doc_id,
                        "conversation_id": conv_id,
                        "filename": filename,
                        "page_num": page_num,
                        "chunk_type": chunk_type,
                        "chunk_id": chunk_id,
                        "upload_order": upload_order,
                        "image_b64": chunk.get("image_b64", "")
                    }
                )
                documents_to_add.append(doc)
            else:
                sub_chunks = self.text_splitter.split_text(content)
                for idx, sub in enumerate(sub_chunks):
                    doc = Document(
                        page_content=sub,
                        metadata={
                            "doc_id": doc_id,
                            "conversation_id": conv_id,
                            "filename": filename,
                            "page_num": page_num,
                            "chunk_type": chunk_type,
                            "chunk_id": f"{chunk_id}_s{idx+1}",
                            "upload_order": upload_order,
                            "image_b64": ""
                        }
                    )
                    documents_to_add.append(doc)

        if documents_to_add:
            self.vector_store.add_documents(documents_to_add)
            logger.info(f"Indexed {len(documents_to_add)} vector chunks into ChromaDB collection '{COLLECTION_NAME}' (persist_dir: '{self.persist_dir}').")
            if logger.isEnabledFor(logging.DEBUG):
                doc_ids_added = list(set(d.metadata.get("doc_id") for d in documents_to_add if d.metadata))
                logger.debug(f"[DEBUG INDEX] Collection='{COLLECTION_NAME}' | DocIDs={doc_ids_added} | Total Documents={len(documents_to_add)}")
        
        return len(documents_to_add)

    def search_similarity(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        k: int = 3,  # Default reduced from 4 → 3 for speed
        filter_type: Optional[str] = None,
        active_docs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in ChromaDB strictly filtered by conversation_id and active_docs.
        Returns top matching sources with metadata preserved.
        """
        where_clause = {}
        conditions = []

        if conversation_id:
            conditions.append({"conversation_id": conversation_id})
        
        if active_docs:
            if len(active_docs) == 1:
                conditions.append({"doc_id": active_docs[0]})
            elif len(active_docs) > 1:
                conditions.append({"doc_id": {"$in": active_docs}})
        elif doc_id:
            conditions.append({"doc_id": doc_id})

        if filter_type:
            conditions.append({"chunk_type": filter_type})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        kwargs = {"k": k}
        if where_clause:
            kwargs["filter"] = where_clause

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[DEBUG SEARCH] Collection='{COLLECTION_NAME}' | Query='{query}' | ActiveDocs={active_docs} | DocID={doc_id} | Filter={where_clause} | k={k}")

        try:
            results = self.vector_store.similarity_search_with_score(query, **kwargs)
        except Exception as e:
            logger.error(f"Similarity search failed with filter {where_clause}: {e}.")
            return []

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[DEBUG SEARCH RESULTS] Raw retrieved count: {len(results)} for query: '{query}'")

        formatted_results = []
        seen_contents = set()

        for doc, score in results:
            content_snippet = doc.page_content.strip()
            # Deduplicate very similar sub-chunks
            if content_snippet in seen_contents:
                continue
            seen_contents.add(content_snippet)

            res_item = {
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "filename": doc.metadata.get("filename", ""),
                "page_num": doc.metadata.get("page_num", 1),
                "chunk_type": doc.metadata.get("chunk_type", "text"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "snippet": content_snippet[:160] + "..." if len(content_snippet) > 160 else content_snippet,
                "score": float(score),
                "image_b64": doc.metadata.get("image_b64", "")
            }
            formatted_results.append(res_item)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  -> Chunk doc_id='{res_item['doc_id']}' page={res_item['page_num']} score={res_item['score']:.4f} snippet='{res_item['snippet'][:80]}'")

            if len(formatted_results) >= k:
                break

        return formatted_results

    def delete_document(self, doc_id: str, conversation_id: Optional[str] = None) -> bool:
        """Delete all chunks associated with doc_id (and optional conversation_id) from ChromaDB."""
        try:
            collection = self.vector_store._collection
            if conversation_id:
                collection.delete(where={"$and": [{"doc_id": doc_id}, {"conversation_id": conversation_id}]})
            else:
                collection.delete(where={"doc_id": doc_id})
            logger.info(f"Deleted doc_id '{doc_id}' (conversation_id='{conversation_id}') from vector store.")
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
