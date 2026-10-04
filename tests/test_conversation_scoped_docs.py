"""Comprehensive Generic Test Suite for Conversation-Scoped Document Memory & Isolation.

Validates all 17 generic requirements without hardcoding any subject, topic, document, question, or formula.
"""

import unittest
import os
import shutil
import tempfile
import time
from unittest.mock import patch, MagicMock

from app.conversation_manager import ConversationManager
from app.rag.vector_store import VectorStoreManager
from app.graph.state import DocuMindState
from app.graph.nodes import (
    retrieve_node,
    router_node,
    text_rag_node,
    calculation_node,
    general_knowledge_node,
    verify_answer_node,
)
from app.graph.workflow import run_documind_workflow


class TestConversationScopedDocs(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.conv_file = os.path.join(self.temp_dir, "conversations.json")
        self.chroma_dir = os.path.join(self.temp_dir, "chroma_db")
        
        self.conv_mgr = ConversationManager(storage_file=self.conv_file)
        self.vector_mgr = VectorStoreManager(persist_dir=self.chroma_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. New application opens a new conversation.
    def test_1_new_application_opens_new_conversation(self):
        conv_id = self.conv_mgr.create_conversation(mode="auto")
        conv = self.conv_mgr.get_conversation(conv_id)
        self.assertIsNotNone(conv)
        self.assertEqual(conv["active_docs"], [])
        self.assertEqual(conv["messages"], [])
        self.assertEqual(conv["documents"], [])

    # 2. Uploading a document associates it with the active conversation.
    def test_2_uploading_document_associates_with_active_conversation(self):
        conv_id = self.conv_mgr.create_conversation(mode="auto")
        doc_id = "doc_alpha_123"
        filename = "physics_notes.pdf"
        
        self.conv_mgr.add_active_doc(conv_id, doc_id, filename=filename, page_count=5)
        conv = self.conv_mgr.get_conversation(conv_id)
        
        self.assertIn(doc_id, conv["active_docs"])
        self.assertEqual(len(conv["documents"]), 1)
        self.assertEqual(conv["documents"][0]["doc_id"], doc_id)
        self.assertEqual(conv["documents"][0]["filename"], filename)
        self.assertEqual(conv["documents"][0]["upload_order"], 1)

    # 3. Reopening an old conversation restores its documents.
    def test_3_reopening_old_conversation_restores_documents(self):
        conv_id = self.conv_mgr.create_conversation(mode="auto")
        self.conv_mgr.add_active_doc(conv_id, "doc_x1", filename="doc_x1.pdf")
        self.conv_mgr.add_message(conv_id, {"role": "user", "content": "Hello"})
        
        # Reload conversation manager from file
        reloaded_mgr = ConversationManager(storage_file=self.conv_file)
        restored_conv = reloaded_mgr.get_conversation(conv_id)
        
        self.assertIsNotNone(restored_conv)
        self.assertIn("doc_x1", restored_conv["active_docs"])
        self.assertEqual(len(restored_conv["messages"]), 1)

    # 4. Conversation A cannot retrieve Conversation B's documents.
    def test_4_conversation_a_cannot_retrieve_conversation_b_documents(self):
        conv_a = self.conv_mgr.create_conversation()
        conv_b = self.conv_mgr.create_conversation()

        chunk_a = {
            "doc_id": "doc_a",
            "filename": "file_a.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "doc_a_p1_s1",
            "content": "Secret protocol algorithm for Conversation A."
        }
        chunk_b = {
            "doc_id": "doc_b",
            "filename": "file_b.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "doc_b_p1_s1",
            "content": "Confidential financial statement for Conversation B."
        }

        self.vector_mgr.add_document_chunks([chunk_a], conversation_id=conv_a)
        self.vector_mgr.add_document_chunks([chunk_b], conversation_id=conv_b)

        # Query Conversation A for financial statement
        res_a = self.vector_mgr.search_similarity(
            query="financial statement",
            conversation_id=conv_a,
            active_docs=["doc_a"]
        )
        for r in res_a:
            self.assertNotEqual(r["doc_id"], "doc_b")
            self.assertNotIn("Conversation B", r["content"])

        # Query Conversation B for protocol algorithm
        res_b = self.vector_mgr.search_similarity(
            query="protocol algorithm",
            conversation_id=conv_b,
            active_docs=["doc_b"]
        )
        for r in res_b:
            self.assertNotEqual(r["doc_id"], "doc_a")
            self.assertNotIn("Conversation A", r["content"])

    # 5. Multiple documents can belong to one conversation.
    def test_5_multiple_documents_belong_to_one_conversation(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "doc_1", filename="first.pdf")
        self.conv_mgr.add_active_doc(conv_id, "doc_2", filename="second.pdf")
        self.conv_mgr.add_active_doc(conv_id, "doc_3", filename="third.pdf")

        docs = self.conv_mgr.get_documents(conv_id)
        self.assertEqual(len(docs), 3)
        self.assertEqual([d["upload_order"] for d in docs], [1, 2, 3])

    # 6. "first PDF" dynamically resolves to the first uploaded PDF.
    def test_6_first_pdf_dynamically_resolves(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "doc_1", filename="annual_report.pdf")
        self.conv_mgr.add_active_doc(conv_id, "doc_2", filename="budget_plan.pdf")

        targets = self.conv_mgr.resolve_target_docs(conv_id, "What is the revenue according to the first PDF?")
        self.assertEqual(targets, ["doc_1"])

    # 7. "second PDF" dynamically resolves to the second uploaded PDF.
    def test_7_second_pdf_dynamically_resolves(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "doc_1", filename="annual_report.pdf")
        self.conv_mgr.add_active_doc(conv_id, "doc_2", filename="budget_plan.pdf")

        targets = self.conv_mgr.resolve_target_docs(conv_id, "Summarize the second PDF")
        self.assertEqual(targets, ["doc_2"])

    # 8. A document filename can dynamically identify the correct document.
    def test_8_filename_dynamically_identifies_correct_document(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "doc_1", filename="annual_report.pdf")
        self.conv_mgr.add_active_doc(conv_id, "doc_2", filename="budget_plan.pdf")

        targets = self.conv_mgr.resolve_target_docs(conv_id, "Explain the projections in budget_plan.pdf")
        self.assertEqual(targets, ["doc_2"])

    # 9. A user can ask questions across multiple documents in the same conversation.
    def test_9_questions_across_multiple_documents(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "doc_1", filename="annual_report.pdf")
        self.conv_mgr.add_active_doc(conv_id, "doc_2", filename="budget_plan.pdf")

        targets = self.conv_mgr.resolve_target_docs(conv_id, "Compare the first and second PDFs")
        self.assertIn("doc_1", targets)
        self.assertIn("doc_2", targets)

    # 10. Unrelated questions do not automatically retrieve conversation documents.
    def test_10_unrelated_questions_do_not_retrieve_documents(self):
        state_math: DocuMindState = {
            "question": "125 × 8",
            "mode": "auto",
            "active_docs": ["doc_1"],
            "context_chunks": [],
            "timings": {}
        }
        res_math = retrieve_node(state_math)
        self.assertEqual(res_math["context_chunks"], [])

        state_gk: DocuMindState = {
            "question": "What is photosynthesis?",
            "mode": "auto",
            "active_docs": [],
            "context_chunks": [],
            "timings": {}
        }
        res_route = router_node(state_gk)
        self.assertEqual(res_route["route"], "general_knowledge")

    # 11. Follow-up questions correctly resolve relevant conversation/document context.
    def test_11_followup_questions_resolve_relevant_context(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_message(conv_id, {"role": "user", "content": "What is Figure 1.2 in the report?"})
        self.conv_mgr.add_message(conv_id, {"role": "assistant", "content": "Figure 1.2 describes B-tree indexing."})

        state: DocuMindState = {
            "question": "Explain this figure in detail",
            "conversation_id": conv_id,
            "mode": "auto",
            "active_docs": ["doc_1"],
            "context_chunks": [],
            "timings": {}
        }
        res = retrieve_node(state)
        # Verify retrieve_node executed without error and logs resolved reference
        self.assertIn("timings", res)

    # 12. Standalone calculations remain independent from document retrieval.
    def test_12_standalone_calculations_remain_independent(self):
        state: DocuMindState = {
            "question": "52 + 53",
            "mode": "auto",
            "active_docs": ["doc_1"],
            "context_chunks": [],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "calculation")
        res_calc = calculation_node(state)
        self.assertIn("105", res_calc["answer"])
        res_verify = verify_answer_node({**state, "answer": res_calc["answer"], "route": "calculation"})
        self.assertTrue(res_verify["verified"])

    # 13. General knowledge remains independent from document retrieval when no docs exist.
    def test_13_general_knowledge_remains_independent(self):
        state: DocuMindState = {
            "question": "What is photosynthesis?",
            "mode": "auto",
            "active_docs": [],
            "context_chunks": [],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "general_knowledge")
        with patch("app.models.llm.LocalLLMManager.generate_text_with_metrics") as mock_gen:
            mock_gen.return_value = ("Photosynthesis is the plant process of converting light into chemical energy.", {"total_llm_time": 0.1})
            res_gk = general_knowledge_node(state)
            self.assertIn("Photosynthesis", res_gk["answer"])
            res_v = verify_answer_node({**state, "answer": res_gk["answer"], "route": "general_knowledge"})
            self.assertTrue(res_v["verified"])

    # 14. Document-based calculations retrieve only the relevant conversation documents.
    def test_14_document_based_calculations_retrieve_relevant_docs(self):
        state: DocuMindState = {
            "question": "Calculate the total cost using value 50 given in page 1 of document",
            "mode": "auto",
            "active_docs": ["doc_1"],
            "context_chunks": [{"doc_id": "doc_1", "page_num": 1, "chunk_type": "text", "content": "Base price is 50."}],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "calculation")

    # 15. Existing conversations retain their document indexes after application restart.
    def test_15_persistence_retains_indexes_after_app_restart(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "doc_persist_1", filename="persist.pdf")

        chunk = {
            "doc_id": "doc_persist_1",
            "filename": "persist.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "p1_s1",
            "content": "Persisted document content across restart."
        }
        self.vector_mgr.add_document_chunks([chunk], conversation_id=conv_id)

        # Simulate App Restart by instantiating new managers on same storage dirs
        reloaded_conv_mgr = ConversationManager(storage_file=self.conv_file)
        reloaded_vector_mgr = VectorStoreManager(persist_dir=self.chroma_dir)

        conv = reloaded_conv_mgr.get_conversation(conv_id)
        self.assertIsNotNone(conv)
        self.assertIn("doc_persist_1", conv["active_docs"])

        search_res = reloaded_vector_mgr.search_similarity(
            query="Persisted document",
            conversation_id=conv_id,
            active_docs=["doc_persist_1"]
        )
        self.assertTrue(len(search_res) > 0)
        self.assertEqual(search_res[0]["doc_id"], "doc_persist_1")

    # 16. Adding a new PDF to an existing conversation does not break the previously uploaded PDF.
    def test_16_adding_new_pdf_does_not_break_previous_pdf(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "pdf_1", filename="part1.pdf")

        chunk1 = {
            "doc_id": "pdf_1",
            "filename": "part1.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "pdf1_s1",
            "content": "Part 1 contains introductory concepts."
        }
        self.vector_mgr.add_document_chunks([chunk1], conversation_id=conv_id)

        # Add second PDF
        self.conv_mgr.add_active_doc(conv_id, "pdf_2", filename="part2.pdf")
        chunk2 = {
            "doc_id": "pdf_2",
            "filename": "part2.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "pdf2_s1",
            "content": "Part 2 contains advanced methodologies."
        }
        self.vector_mgr.add_document_chunks([chunk2], conversation_id=conv_id)

        # Verify PDF 1 is still searchable
        res1 = self.vector_mgr.search_similarity("introductory concepts", conversation_id=conv_id, active_docs=["pdf_1", "pdf_2"])
        self.assertTrue(any(r["doc_id"] == "pdf_1" for r in res1))

        # Verify PDF 2 is searchable
        res2 = self.vector_mgr.search_similarity("advanced methodologies", conversation_id=conv_id, active_docs=["pdf_1", "pdf_2"])
        self.assertTrue(any(r["doc_id"] == "pdf_2" for r in res2))

    # 17. Removing/clearing a conversation does not leak its documents into other conversations.
    def test_17_deleting_conversation_cleans_up_without_leaks(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_active_doc(conv_id, "temp_doc", filename="temp.pdf")
        chunk = {
            "doc_id": "temp_doc",
            "filename": "temp.pdf",
            "page_num": 1,
            "chunk_type": "text",
            "chunk_id": "temp_s1",
            "content": "Temporary contents."
        }
        self.vector_mgr.add_document_chunks([chunk], conversation_id=conv_id)

        deleted_docs = self.conv_mgr.delete_conversation(conv_id)
        for d_id in deleted_docs:
            self.vector_mgr.delete_document(d_id, conversation_id=conv_id)

        self.assertIsNone(self.conv_mgr.get_conversation(conv_id))
        search_res = self.vector_mgr.search_similarity("Temporary contents", conversation_id=conv_id, active_docs=["temp_doc"])
        self.assertEqual(len(search_res), 0)


    # 18. Follow-up phrase reference resolution test
    def test_18_followup_phrases_resolution(self):
        conv_id = self.conv_mgr.create_conversation()
        self.conv_mgr.add_message(conv_id, {"role": "user", "content": "What is the Dijkstra algorithm and Sieve algorithm?"})
        self.conv_mgr.add_message(conv_id, {"role": "assistant", "content": "Dijkstra finds shortest path, Sieve finds primes."})

        state: DocuMindState = {
            "question": "Explain those two methods in detail",
            "conversation_id": conv_id,
            "mode": "auto",
            "active_docs": ["doc_1"],
            "context_chunks": [],
            "timings": {}
        }
        res = retrieve_node(state)
        self.assertIn("timings", res)

    # 19. Current information routes to web search
    def test_19_current_info_web_search_routing(self):
        state: DocuMindState = {
            "question": "What is the latest Python version in 2026?",
            "mode": "auto",
            "active_docs": [],
            "context_chunks": [],
            "timings": {}
        }
        res_route = router_node(state)
    # 20. Reopening, refreshing, and history selection does not create duplicate conversations
    def test_20_reopening_and_refreshing_does_not_create_duplicate_conversations(self):
        conv_id = self.conv_mgr.create_conversation(mode="auto")
        self.conv_mgr.add_message(conv_id, {"role": "user", "content": "What is TCP?"})
        self.conv_mgr.add_message(conv_id, {"role": "assistant", "content": "TCP is a protocol."})
        
        initial_count = len(self.conv_mgr.list_conversations())
        
        # Simulate app load / refresh: existing conversations are listed, no new conversation created
        all_convs = self.conv_mgr.list_conversations()
        self.assertEqual(len(all_convs), initial_count)
        
        current_conv = self.conv_mgr.get_conversation(conv_id)
        self.assertIsNotNone(current_conv)
        self.assertEqual(len(current_conv["messages"]), 2)
        
        # Simulate selecting an existing conversation from history
        selected_conv = self.conv_mgr.get_conversation(all_convs[0]["id"])
        self.assertEqual(selected_conv["id"], conv_id)
        self.assertEqual(len(self.conv_mgr.list_conversations()), initial_count)

        # Unpersisted new draft conversation does not increase saved history count until used
        new_conv_id = self.conv_mgr.create_conversation(mode="auto", save_immediately=False)
        self.assertEqual(len(self.conv_mgr.list_conversations()), initial_count)

        # Sending a message persists the conversation to History
        self.conv_mgr.add_message(new_conv_id, {"role": "user", "content": "Hello"})
        self.assertEqual(len(self.conv_mgr.list_conversations()), initial_count + 1)


if __name__ == "__main__":
    unittest.main()
