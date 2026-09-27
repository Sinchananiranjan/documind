"""Unit tests validating document list queries, fallback message preservation, and derived math verification."""

import unittest
from unittest.mock import patch, MagicMock

from app.graph.state import DocuMindState
from app.graph.nodes import (
    text_rag_node,
    verify_answer_node,
    fallback_node,
    _is_document_list_query,
    _resolve_conversational_references
)
from app.conversation_manager import ConversationManager


class TestChatGPTUploadAndNotFound(unittest.TestCase):

    def test_is_document_list_query_detection(self):
        """Test detection of queries asking for available or uploaded documents list."""
        self.assertTrue(_is_document_list_query("Which documents are available?"))
        self.assertTrue(_is_document_list_query("What documents are uploaded?"))
        self.assertTrue(_is_document_list_query("List the uploaded PDFs"))
        self.assertFalse(_is_document_list_query("What is the algorithm on page 5?"))

    def test_document_list_query_returns_actual_filenames(self):
        """Test 'Which documents are available?' returns actual uploaded PDF filenames, not chunks."""
        from app.graph.nodes import conv_manager
        conv_id = conv_manager.create_conversation()
        conv_manager.add_active_doc(conv_id, "doc_101", filename="annual_report.pdf", page_count=10)
        conv_manager.add_active_doc(conv_id, "doc_102", filename="budget_summary.pdf", page_count=5)

        state: DocuMindState = {
            "question": "Which documents are available?",
            "conversation_id": conv_id,
            "mode": "auto",
            "active_docs": ["doc_101", "doc_102"],
            "context_chunks": [],
            "timings": {}
        }
        res = text_rag_node(state)
        self.assertTrue(res["verified"])
        self.assertIn("annual_report.pdf", res["answer"])
        self.assertIn("budget_summary.pdf", res["answer"])
        self.assertNotIn("doc_101_p1_s1", res["answer"])  # Not raw chunk IDs

    def test_derived_math_numbers_pass_verification(self):
        """Derived/calculated math numbers like '70.57' should pass verification without hallucination flags."""
        state: DocuMindState = {
            "question": "Calculate the percentage change between 100 and 170.57",
            "answer": "The calculated percentage change is 70.57%.",
            "route": "text_rag",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Value increased from 100 to 170.57."}],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertTrue(res["verified"])

    def test_fallback_node_preserves_explicit_state_messages(self):
        """Fallback node must preserve explicit missing document state messages."""
        state: DocuMindState = {
            "question": "What is mentioned about quantum computing?",
            "answer": "The selected document does not provide information about this topic.",
            "route": "text_rag",
            "context_chunks": [],
            "timings": {}
        }
        res = fallback_node(state)
        self.assertEqual(res["answer"], "The selected document does not provide information about this topic.")


if __name__ == "__main__":
    unittest.main()
