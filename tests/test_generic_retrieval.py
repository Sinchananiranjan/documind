"""Generic retrieval test suite for PDF indexing, vector storage, metadata filtering, multi-document isolation, and evidence sufficiency.

No subject-specific hardcoding is used.
"""

import unittest
from app.rag.vector_store import VectorStoreManager
from app.graph.workflow import run_documind_workflow


class TestGenericRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vector_manager = VectorStoreManager()
        
        # Test Doc 1 (Generic Document A)
        cls.doc1_chunks = [
            {
                "doc_id": "doc_alpha_999",
                "filename": "document_alpha.pdf",
                "page_num": 1,
                "chunk_type": "text",
                "chunk_id": "alpha_c1",
                "content": "Document Alpha defines System X: System X operates with distributed consensus and 256-bit encryption."
            },
            {
                "doc_id": "doc_alpha_999",
                "filename": "document_alpha.pdf",
                "page_num": 2,
                "chunk_type": "table",
                "chunk_id": "alpha_t1",
                "content": "| Parameter | Target Value |\n| Latency | 5 ms |\n| Throughput | 10000 req/s |"
            }
        ]
        
        # Test Doc 2 (Generic Document B)
        cls.doc2_chunks = [
            {
                "doc_id": "doc_beta_888",
                "filename": "document_beta.pdf",
                "page_num": 1,
                "chunk_type": "text",
                "chunk_id": "beta_c1",
                "content": "Document Beta defines System Y: System Y uses centralized queueing with token bucket rate limiting."
            }
        ]

        cls.vector_manager.add_document_chunks(cls.doc1_chunks)
        cls.vector_manager.add_document_chunks(cls.doc2_chunks)

    @classmethod
    def tearDownClass(cls):
        cls.vector_manager.delete_document("doc_alpha_999")
        cls.vector_manager.delete_document("doc_beta_888")

    def test_1_indexing_and_storage(self):
        """Test 1: Verify chunks are properly embedded and indexed in persistent ChromaDB collection."""
        indexed_docs = self.vector_manager.list_indexed_documents()
        self.assertIn("doc_alpha_999", indexed_docs)
        self.assertIn("doc_beta_888", indexed_docs)

    def test_2_retrieval_and_relevance(self):
        """Test 2: Querying retrieves relevant evidence chunks with preserved metadata."""
        results = self.vector_manager.search_similarity(
            query="distributed consensus and encryption",
            active_docs=["doc_alpha_999"]
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "doc_alpha_999")
        self.assertIn("encryption", results[0]["content"].lower())

    def test_3_metadata_filtering(self):
        """Test 3: Single and multi-doc filtering properly constrains ChromaDB retrieval results."""
        # Single doc filter
        res_alpha = self.vector_manager.search_similarity(
            query="system",
            active_docs=["doc_alpha_999"]
        )
        for r in res_alpha:
            self.assertEqual(r["doc_id"], "doc_alpha_999")

        # Multi-doc filter
        res_multi = self.vector_manager.search_similarity(
            query="system",
            active_docs=["doc_alpha_999", "doc_beta_888"]
        )
        doc_ids_returned = {r["doc_id"] for r in res_multi}
        self.assertTrue(doc_ids_returned.issubset({"doc_alpha_999", "doc_beta_888"}))

    def test_4_multi_document_isolation(self):
        """Test 4: Strict document isolation prevents retrieving content from unselected documents."""
        # Querying doc_beta_888 for content only present in doc_alpha_999
        res = self.vector_manager.search_similarity(
            query="256-bit encryption",
            active_docs=["doc_beta_888"]
        )
        # Should not return doc_alpha_999 content
        for r in res:
            self.assertNotEqual(r["doc_id"], "doc_alpha_999")

    def test_5_insufficient_evidence_response(self):
        """Test 5: Querying for completely absent concept in Document Mode yields standard fallback."""
        res = run_documind_workflow(
            question="What is quantum gravity warp drive protocol?",
            doc_id="doc_alpha_999",
            active_docs=["doc_alpha_999"],
            mode="document_mode"
        )
        self.assertTrue(
            "couldn't find" in res["answer"].lower() or "not provide" in res["answer"].lower() or "not contain" in res["answer"].lower()
        )
        self.assertFalse(res["verified"])


if __name__ == "__main__":
    unittest.main()
