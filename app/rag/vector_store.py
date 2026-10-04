"""Vector store manager using persistent ChromaDB and Haystack hybrid retrieval pipeline."""

import os
import logging
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model
from app.rag.haystack_retriever import HaystackHybridRetriever

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "documind_collection"

_vector_store_instance = None
_haystack_instance = None


class VectorStoreManager:
    """Manages document chunking, ChromaDB persistence, and Haystack hybrid retrieval."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        global _vector_store_instance, _haystack_instance
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

        if _haystack_instance is None:
            _haystack_instance = HaystackHybridRetriever()
        self.haystack_retriever = _haystack_instance

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=550,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def add_document_chunks(self, chunks: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> int:
        """
        Splits text chunks while strictly preserving metadata:
        conversation_id, doc_id, filename, page_num, chunk_type, chunk_id.
        Indexes both in ChromaDB (vector persistence) and Haystack (hybrid retrieval).
        """
        documents_to_add = []
        raw_chunks_to_haystack = []

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
                raw_chunks_to_haystack.append({
                    "doc_id": doc_id,
                    "conversation_id": conv_id,
                    "filename": filename,
                    "page_num": page_num,
                    "chunk_type": chunk_type,
                    "chunk_id": chunk_id,
                    "content": content,
                    "upload_order": upload_order,
                    "image_b64": chunk.get("image_b64", "")
                })
            else:
                sub_chunks = self.text_splitter.split_text(content)
                for idx, sub in enumerate(sub_chunks):
                    sub_cid = f"{chunk_id}_s{idx+1}"
                    doc = Document(
                        page_content=sub,
                        metadata={
                            "doc_id": doc_id,
                            "conversation_id": conv_id,
                            "filename": filename,
                            "page_num": page_num,
                            "chunk_type": chunk_type,
                            "chunk_id": sub_cid,
                            "upload_order": upload_order,
                            "image_b64": ""
                        }
                    )
                    documents_to_add.append(doc)
                    raw_chunks_to_haystack.append({
                        "doc_id": doc_id,
                        "conversation_id": conv_id,
                        "filename": filename,
                        "page_num": page_num,
                        "chunk_type": chunk_type,
                        "chunk_id": sub_cid,
                        "content": sub,
                        "upload_order": upload_order,
                        "image_b64": ""
                    })

        if documents_to_add:
            self.vector_store.add_documents(documents_to_add)
            self.haystack_retriever.index_chunks(raw_chunks_to_haystack, conversation_id=conversation_id)
            logger.info(f"Indexed {len(documents_to_add)} chunks into ChromaDB and Haystack retriever.")

        return len(documents_to_add)

    def search_similarity(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        k: int = 8,
        filter_type: Optional[str] = None,
        active_docs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in ChromaDB strictly filtered by conversation_id and active_docs.
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

        try:
            results = self.vector_store.similarity_search_with_score(query, **kwargs)
        except Exception as e:
            logger.error(f"Similarity search failed with filter {where_clause}: {e}.")
            return []

        formatted_results = []
        seen_contents = set()

        for doc, score in results:
            content_snippet = doc.page_content.strip()
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

            if len(formatted_results) >= k:
                break

        return formatted_results

    def search_hybrid(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        k: int = 10,
        active_docs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Haystack Hybrid Retrieval:
        1. Syncs stored ChromaDB documents into Haystack if not already present.
        2. Retrieves dense similarity matches from ChromaDB.
        3. Retrieves BM25 lexical matches from Haystack's InMemoryBM25Retriever.
        4. Fuses candidates using RRF (Reciprocal Rank Fusion).
        5. Applies neighboring chunk and page window expansion so multi-part answers are complete.
        """
        target_docs = active_docs or ([doc_id] if doc_id else [])

        # Ensure Haystack has the document chunks loaded from ChromaDB if needed
        self._ensure_haystack_synced(target_docs=target_docs, conversation_id=conversation_id)

        # 1. Dense retrieval from ChromaDB
        dense_results = self.search_similarity(
            query=query,
            conversation_id=conversation_id,
            doc_id=doc_id,
            k=k * 2,
            active_docs=target_docs
        )

        # 2. BM25 retrieval from Haystack
        bm25_results = self.haystack_retriever.bm25_search(
            query=query,
            conversation_id=conversation_id,
            active_docs=target_docs,
            top_k=k * 2
        )

        # 3. Fuse Dense + BM25 using Reciprocal Rank Fusion (RRF)
        fused_candidates = self.haystack_retriever.fuse_dense_and_bm25(dense_results, bm25_results)

        # 4. Context expansion: neighboring chunks & adjacent page chunks
        expanded_candidates = self.haystack_retriever.expand_neighboring_chunks_and_pages(
            candidate_chunks=fused_candidates[:k],
            conversation_id=conversation_id,
            active_docs=target_docs,
            max_total_chunks=k + 4
        )

        return expanded_candidates

    def _ensure_haystack_synced(self, target_docs: List[str], conversation_id: Optional[str] = None):
        """Syncs chunks stored in ChromaDB into Haystack DocumentStore if missing."""
        if not target_docs:
            return
        
        # Check if target_docs already exist in haystack internal map
        loaded_doc_ids = set(c.get("doc_id") for c in self.haystack_retriever._chunks_by_id.values())
        missing_docs = [d for d in target_docs if d not in loaded_doc_ids]

        if missing_docs:
            logger.info(f"[HAYSTACK SYNC] Loading missing docs {missing_docs} from ChromaDB into Haystack...")
            chroma_chunks = self.get_all_chunks_for_docs(active_docs=missing_docs, conversation_id=conversation_id, max_chunks=300)
            if chroma_chunks:
                self.haystack_retriever.index_chunks(chroma_chunks, conversation_id=conversation_id)

    def delete_document(self, doc_id: str, conversation_id: Optional[str] = None) -> bool:
        """Delete all chunks associated with doc_id from ChromaDB and Haystack."""
        try:
            collection = self.vector_store._collection
            if conversation_id:
                collection.delete(where={"$and": [{"doc_id": doc_id}, {"conversation_id": conversation_id}]})
            else:
                collection.delete(where={"doc_id": doc_id})
            
            self.haystack_retriever.delete_document(doc_id, conversation_id=conversation_id)
            logger.info(f"Deleted doc_id '{doc_id}' from vector store and Haystack retriever.")
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

    def get_all_chunks_for_docs(
        self,
        active_docs: List[str],
        conversation_id: Optional[str] = None,
        max_chunks: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ALL stored chunks for the given active_docs by metadata filter.
        """
        try:
            collection = self.vector_store._collection
            conditions = []
            if active_docs:
                if len(active_docs) == 1:
                    conditions.append({"doc_id": active_docs[0]})
                else:
                    conditions.append({"doc_id": {"$in": active_docs}})
            if conversation_id:
                conditions.append({"conversation_id": conversation_id})

            where_clause = {}
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}

            kwargs = {"include": ["documents", "metadatas"]}
            if where_clause:
                kwargs["where"] = where_clause

            raw = collection.get(**kwargs)
            docs = raw.get("documents") or []
            metas = raw.get("metadatas") or []

            results = []
            for content, meta in zip(docs, metas):
                if not content or not meta:
                    continue
                results.append({
                    "doc_id": meta.get("doc_id", "unknown"),
                    "filename": meta.get("filename", ""),
                    "page_num": meta.get("page_num", 1),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "chunk_id": meta.get("chunk_id", ""),
                    "content": content,
                    "snippet": content[:160] + "..." if len(content) > 160 else content,
                    "score": 0.5,
                    "image_b64": meta.get("image_b64", "")
                })

            results.sort(key=lambda x: (x.get("doc_id", ""), x.get("page_num", 1)))
            return results[:max_chunks]

        except Exception as e:
            logger.error(f"[LEXICAL FETCH] Failed to fetch all chunks for docs {active_docs}: {e}")
            return []
