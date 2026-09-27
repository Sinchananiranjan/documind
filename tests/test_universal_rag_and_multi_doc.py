import unittest
import os
import shutil
import tempfile
from app.conversation_manager import ConversationManager
from app.rag.vector_store import VectorStoreManager
from app.graph.nodes import (
    _resolve_conversational_references,
    _compress_context,
    retrieve_node,
    router_node,
    text_rag_node,
    general_knowledge_node
)
from app.graph.state import DocuMindState

class TestUniversalRAGAndMultiDoc(unittest.TestCase):
    """
    Validation test suite for:
    - Follow-up reference resolution ("those two", "these functions", "explain that", etc.)
    - Multi-document balanced retrieval & grouped context formatting
    - Independent question fallback to General Knowledge vs Document-specific queries
    """

    def setUp(self):
        from app.graph.nodes import conv_manager, vector_manager
        self.conv_mgr = conv_manager
        self.vector_mgr = vector_manager

    def test_01_reference_resolution_generic(self):
        conv_id = self.conv_mgr.create_conversation()
        
        # User message 1
        self.conv_mgr.add_message(conv_id, {"role": "user", "content": "Which two Pandas functions are used for discretization?"})
        # Asst message 1
        self.conv_mgr.add_message(conv_id, {"role": "assistant", "content": "The two main discretization functions are **cut()** and **qcut()**."})
        
        # Follow-up question 1
        enriched_q, refs = _resolve_conversational_references("What is the difference between those two?", conv_id)
        self.assertTrue(len(refs) > 0)
        self.assertTrue(any("cut" in r.lower() or "qcut" in r.lower() for r in refs))
        self.assertIn("cut", enriched_q.lower())

    def test_02_multi_document_balanced_retrieval(self):
        conv_id = self.conv_mgr.create_conversation()
        
        # Add doc 1
        self.conv_mgr.add_active_doc(conv_id, "doc_mod_3", filename="Module_3.pdf")
        chunk_mod_3 = {
            "doc_id": "doc_mod_3",
            "filename": "Module_3.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "m3_c1",
            "content": "Module 3 covers data cleaning, handling missing values, and discretization using cut() and qcut()."
        }
        self.vector_mgr.add_document_chunks([chunk_mod_3], conversation_id=conv_id)

        # Add doc 2
        self.conv_mgr.add_active_doc(conv_id, "doc_mod_5", filename="Module_5.pdf")
        chunk_mod_5 = {
            "doc_id": "doc_mod_5",
            "filename": "Module_5.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "m5_c1",
            "content": "Module 5 covers model evaluation, confusion matrix, precision, recall, and ROC curves."
        }
        self.vector_mgr.add_document_chunks([chunk_mod_5], conversation_id=conv_id)

        state: DocuMindState = {
            "question": "What is the difference between Module 3 and Module 5?",
            "conversation_id": conv_id,
            "active_docs": ["doc_mod_3", "doc_mod_5"],
            "mode": "auto"
        }

        ret_res = retrieve_node(state)
        chunks = ret_res["context_chunks"]
        
        # Verify chunks from BOTH documents are present
        doc_ids_retrieved = {c["doc_id"] for c in chunks}
        self.assertIn("doc_mod_3", doc_ids_retrieved)
        self.assertIn("doc_mod_5", doc_ids_retrieved)

        # Verify compressed context groups by document filename
        formatted_ctx = _compress_context(chunks)
        self.assertIn("=== DOCUMENT: Module_3.pdf ===", formatted_ctx)
        self.assertIn("=== DOCUMENT: Module_5.pdf ===", formatted_ctx)

    def test_03_independent_question_general_knowledge_fallback(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "doc_mod_3", filename="Module_3.pdf")
        
        # User asks independent question "Who invented Python?" while Module_3 is attached
        state: DocuMindState = {
            "question": "Who invented Python?",
            "conversation_id": conv_id,
            "active_docs": ["doc_mod_3"],
            "mode": "auto"
        }

        ret_res = retrieve_node(state)
        state.update(ret_res)
        
        route_res = router_node(state)
        self.assertEqual(route_res["route"], "general_knowledge")

if __name__ == "__main__":
    unittest.main()
