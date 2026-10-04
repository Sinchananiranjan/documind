"""
Haystack 2.x/3.x Hybrid Retrieval Engine for DocuMind.
Integrates BM25 lexical retrieval, Dense semantic retrieval, result fusion (RRF/Combined),
neighboring chunk/page window expansion, and conversation-isolated document indexing.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

logger = logging.getLogger(__name__)


class HaystackHybridRetriever:
    """
    Manages hybrid BM25 + Dense retrieval using Haystack 2.x/3.x components
    with neighboring chunk/page window expansion and RRF candidate fusion.
    """

    def __init__ (self):
        self.doc_store = InMemoryDocumentStore()
        # Keep an internal dictionary for fast metadata / neighborhood lookups
        # Key: chunk_id or (doc_id, page_num, chunk_index)
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self._chunks_by_doc_page: Dict[str, List[Dict[str, Any]]] = {}

    def index_chunks(self, chunks: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> int:
        """
        Indexes document chunks into Haystack's InMemoryDocumentStore and internal structures.
        Preserves all metadata: doc_id, conversation_id, filename, page_num, chunk_type, chunk_id.
        """
        haystack_docs = []
        count = 0

        for chunk in chunks:
            doc_id = chunk["doc_id"]
            conv_id = conversation_id or chunk.get("conversation_id", "")
            filename = chunk.get("filename", "")
            page_num = chunk.get("page_num", 1)
            chunk_type = chunk.get("chunk_type", "text")
            chunk_id = chunk.get("chunk_id", f"{doc_id}_p{page_num}")
            content = chunk.get("content", "")
            upload_order = chunk.get("upload_order", 1)
            image_b64 = chunk.get("image_b64", "")

            meta = {
                "doc_id": doc_id,
                "conversation_id": conv_id,
                "filename": filename,
                "page_num": page_num,
                "chunk_type": chunk_type,
                "chunk_id": chunk_id,
                "upload_order": upload_order,
                "image_b64": image_b64
            }

            chunk_item = {
                "doc_id": doc_id,
                "conversation_id": conv_id,
                "filename": filename,
                "page_num": page_num,
                "chunk_type": chunk_type,
                "chunk_id": chunk_id,
                "content": content,
                "snippet": content[:160] + "..." if len(content) > 160 else content,
                "image_b64": image_b64,
                "meta": meta
            }

            self._chunks_by_id[chunk_id] = chunk_item
            
            doc_page_key = f"{doc_id}_p{page_num}"
            if doc_page_key not in self._chunks_by_doc_page:
                self._chunks_by_doc_page[doc_page_key] = []
            self._chunks_by_doc_page[doc_page_key].append(chunk_item)

            h_doc = HaystackDocument(content=content, meta=meta, id=chunk_id)
            haystack_docs.append(h_doc)
            count += 1

        if haystack_docs:
            self.doc_store.write_documents(haystack_docs)
            logger.info(f"[HAYSTACK RETRIEVER] Indexed {len(haystack_docs)} chunks into Haystack DocumentStore.")

        return count

    def delete_document(self, doc_id: str, conversation_id: Optional[str] = None):
        """Deletes all chunks for doc_id from Haystack DocumentStore and internal maps."""
        to_delete_ids = []
        for cid, item in list(self._chunks_by_id.items()):
            if item["doc_id"] == doc_id:
                if conversation_id is None or item["conversation_id"] == conversation_id:
                    to_delete_ids.append(cid)
                    del self._chunks_by_id[cid]

        for key in list(self._chunks_by_doc_page.keys()):
            if key.startswith(f"{doc_id}_p"):
                del self._chunks_by_doc_page[key]

        if to_delete_ids:
            try:
                self.doc_store.delete_documents(to_delete_ids)
            except Exception as e:
                logger.warning(f"[HAYSTACK RETRIEVER] Error deleting docs from Haystack store: {e}")

    def bm25_search(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        active_docs: Optional[List[str]] = None,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 search over indexed Haystack documents with conversation and active_docs filters.
        """
        bm25_retriever = InMemoryBM25Retriever(document_store=self.doc_store, top_k=top_k * 2)

        # Build filter dict for Haystack
        filters = None
        conditions = []

        if conversation_id:
            conditions.append({"field": "meta.conversation_id", "operator": "==", "value": conversation_id})

        if active_docs:
            if len(active_docs) == 1:
                conditions.append({"field": "meta.doc_id", "operator": "==", "value": active_docs[0]})
            elif len(active_docs) > 1:
                conditions.append({"field": "meta.doc_id", "operator": "in", "value": active_docs})

        if len(conditions) == 1:
            filters = conditions[0]
        elif len(conditions) > 1:
            filters = {"operator": "AND", "conditions": conditions}

        try:
            res = bm25_retriever.run(query=query, filters=filters, top_k=top_k)
            retrieved_docs = res.get("documents", [])
        except Exception as e:
            logger.error(f"[HAYSTACK BM25] Search failed: {e}")
            retrieved_docs = []

        results = []
        for rank, d in enumerate(retrieved_docs):
            meta = d.meta or {}
            score = d.score if d.score is not None else max(0.1, 1.0 - (rank * 0.05))
            results.append({
                "doc_id": meta.get("doc_id", "unknown"),
                "filename": meta.get("filename", ""),
                "page_num": meta.get("page_num", 1),
                "chunk_type": meta.get("chunk_type", "text"),
                "chunk_id": d.id or meta.get("chunk_id", ""),
                "content": d.content or "",
                "snippet": (d.content or "")[:160] + "...",
                "bm25_score": float(score),
                "score": float(score),
                "image_b64": meta.get("image_b64", "")
            })

        return results

    def expand_neighboring_chunks_and_pages(
        self,
        candidate_chunks: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        active_docs: Optional[List[str]] = None,
        max_total_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Window expansion: For retrieved candidate chunks, automatically incorporates
        neighboring sub-chunks (e.g. _s1, _s2) and adjacent page chunks from the same document.
        This ensures answers spanning across chunk boundaries or pages (like lists of 5 ingredients
        or 3 dimensions) are complete and unbroken in the retrieved evidence.
        """
        if not candidate_chunks:
            return []

        expanded_map: Dict[str, Dict[str, Any]] = {}
        ordered_keys = []

        for c in candidate_chunks:
            cid = c.get("chunk_id", "")
            doc_id = c.get("doc_id", "")
            page_num = c.get("page_num", 1)

            if cid not in expanded_map:
                expanded_map[cid] = c
                ordered_keys.append(cid)

            # 1. Expand same-page sub-chunks
            doc_page_key = f"{doc_id}_p{page_num}"
            same_page_chunks = self._chunks_by_doc_page.get(doc_page_key, [])
            for sp_chunk in same_page_chunks:
                sp_id = sp_chunk.get("chunk_id", "")
                if sp_id not in expanded_map:
                    # Inherit parent relevance score with slight decay
                    c_copy = dict(sp_chunk)
                    c_copy["combined_score"] = max(0.15, c.get("combined_score", 0.3) * 0.85)
                    c_copy["score"] = max(0.15, c.get("score", 0.3) * 0.85)
                    expanded_map[sp_id] = c_copy
                    ordered_keys.append(sp_id)

            # 2. Expand adjacent page chunks (page_num + 1)
            next_page_key = f"{doc_id}_p{page_num + 1}"
            next_page_chunks = self._chunks_by_doc_page.get(next_page_key, [])
            for np_chunk in next_page_chunks[:2]:
                np_id = np_chunk.get("chunk_id", "")
                if np_id not in expanded_map:
                    c_copy = dict(np_chunk)
                    c_copy["combined_score"] = max(0.10, c.get("combined_score", 0.3) * 0.75)
                    c_copy["score"] = max(0.10, c.get("score", 0.3) * 0.75)
                    expanded_map[np_id] = c_copy
                    ordered_keys.append(np_id)

        # Sort expanded pool by doc_id, page_num, and chunk_id for sequential reading
        expanded_list = list(expanded_map.values())
        expanded_list.sort(key=lambda x: (x.get("doc_id", ""), x.get("page_num", 1), x.get("chunk_id", "")))

        logger.info(f"[HAYSTACK EXPAND] Expanded candidate set from {len(candidate_chunks)} to {len(expanded_list)} chunks.")
        return expanded_list[:max_total_chunks]

    def fuse_dense_and_bm25(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) combining dense vector search scores and Haystack BM25 scores.
        RRF_score(doc) = 1/(k + rank_dense) + 1/(k + rank_bm25)
        """
        fusion_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for rank, c in enumerate(dense_results):
            cid = c.get("chunk_id", "") or (c["doc_id"] + str(c["page_num"]) + c["content"][:40])
            chunk_map[cid] = c
            fusion_scores[cid] = fusion_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Process BM25 results
        for rank, c in enumerate(bm25_results):
            cid = c.get("chunk_id", "") or (c["doc_id"] + str(c["page_num"]) + c["content"][:40])
            if cid not in chunk_map:
                chunk_map[cid] = c
            fusion_scores[cid] = fusion_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Build fused list
        fused = []
        max_rrf = max(fusion_scores.values()) if fusion_scores else 1.0

        for cid, rrf_val in fusion_scores.items():
            item = dict(chunk_map[cid])
            # Normalize RRF score to 0.0 - 1.0 range
            norm_rrf = round(rrf_val / max_rrf, 4)
            item["rrf_score"] = norm_rrf
            
            # Combine with raw distance/score if present
            raw_dense_score = max(0.0, 1.0 - (item.get("score", 1.0) / 2.0))
            combined_score = max(norm_rrf, round(0.5 * norm_rrf + 0.5 * raw_dense_score, 3))
            item["combined_score"] = combined_score
            fused.append(item)

        fused.sort(key=lambda x: x["combined_score"], reverse=True)
        return fused
