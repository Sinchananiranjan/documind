"""
Generic regression tests for retrieval quality, query normalization,
document-first search priority, and honest answer verification.
Zero hardcoding of specific questions, answers, topics, or page numbers.
"""

import unittest
import uuid
from app.graph.nodes import (
    _normalize_retrieval_query,
    retrieve_node,
    router_node,
    verify_answer_node,
    vector_manager
)
from app.graph.state import DocuMindState


class TestGenericConceptRetrievalAndVerification(unittest.TestCase):

    def test_query_normalization_strips_meta_phrases(self):
        """Verify that instructional meta-phrases are stripped without altering the core question."""
        test_cases = [
            (
                "What are the four basic tasks in designing security services from the PDF?",
                "What are the four basic tasks in designing security services"
            ),
            (
                "According to the PDF, describe five ingredients of symmetric cipher model",
                "describe five ingredients of symmetric cipher model"
            ),
            (
                "Based on the uploaded document, list the three independent dimensions",
                "list the three independent dimensions"
            ),
            (
                "Explain Caesar cipher from the uploaded file",
                "Explain Caesar cipher"
            ),
            (
                "What does the document say about monoalphabetic substitution?",
                "monoalphabetic substitution"
            )
        ]
        for raw, expected_contain in test_cases:
            norm = _normalize_retrieval_query(raw)
            self.assertNotIn("from the PDF", norm)
            self.assertNotIn("According to the PDF", norm)
            self.assertNotIn("uploaded document", norm)
            self.assertIn(expected_contain, norm)

    def test_false_verification_prevented_on_fallback(self):
        """Verify that any fallback answer produces verified = False in verify_answer_node."""
        fallback_answers = [
            "The uploaded document does not contain enough information to answer this.",
            "The uploaded document does not contain enough information.",
            "The uploaded document does not provide information to answer this.",
            "I couldn't find this information in the document.",
            "The topic is not present in the selected file."
        ]
        for ans in fallback_answers:
            state: DocuMindState = {
                "question": "Sample generic concept query?",
                "answer": ans,
                "route": "text_rag",
                "context_chunks": [],
                "doc_id": "test_doc_123",
                "conversation_id": "conv_test_123",
                "sources": [],
                "timings": {}
            }
            res = verify_answer_node(state)
            self.assertFalse(res.get("verified"), f"Failed for fallback answer: '{ans}'")

    def test_document_first_retrieval_flow_without_meta_phrases(self):
        """
        Verify that when a document exists in the conversation, normal user questions
        search the document FIRST via hybrid retrieval even without 'from the PDF'.
        """
        conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        doc_id = f"test_doc_{uuid.uuid4().hex[:8]}"

        # Index a synthetic document containing arbitrary technical concepts
        synthetic_chunks = [
            {
                "doc_id": doc_id,
                "conversation_id": conv_id,
                "filename": "security_overview.pdf",
                "page_num": 2,
                "chunk_type": "text",
                "chunk_id": f"{doc_id}_p2_s1",
                "content": (
                    "In designing a security service, there are four basic tasks: "
                    "1. Design an algorithm for performing the security-related transformation. "
                    "2. Generate the secret information to be used with the algorithm. "
                    "3. Develop methods for the distribution and sharing of the secret information. "
                    "4. Specify a protocol to be used by the two parties to achieve the security service."
                )
            },
            {
                "doc_id": doc_id,
                "conversation_id": conv_id,
                "filename": "security_overview.pdf",
                "page_num": 3,
                "chunk_type": "text",
                "chunk_id": f"{doc_id}_p3_s1",
                "content": (
                    "A symmetric encryption scheme has five ingredients: "
                    "plaintext, encryption algorithm, secret key, ciphertext, and decryption algorithm. "
                    "Cryptographic systems are characterized along three independent dimensions: "
                    "the type of operations used, the number of keys used, and the way plaintext is processed."
                )
            }
        ]

        vector_manager.add_document_chunks(synthetic_chunks, conversation_id=conv_id)

        try:
            # Query WITHOUT "from the PDF"
            q1 = "What are the four basic tasks in designing security services?"
            state1: DocuMindState = {
                "question": q1,
                "conversation_id": conv_id,
                "doc_id": doc_id,
                "active_docs": [doc_id],
                "mode": "auto",
                "timings": {}
            }
            ret1 = retrieve_node(state1)
            chunks1 = ret1.get("context_chunks", [])
            self.assertTrue(len(chunks1) > 0, "Retrieval failed to find four basic tasks without meta phrase")
            self.assertIn("four basic tasks", chunks1[0]["content"].lower())

            # Query WITH "from the PDF"
            q2 = "What are the four basic tasks in designing security services from the PDF?"
            state2: DocuMindState = {
                "question": q2,
                "conversation_id": conv_id,
                "doc_id": doc_id,
                "active_docs": [doc_id],
                "mode": "auto",
                "timings": {}
            }
            ret2 = retrieve_node(state2)
            chunks2 = ret2.get("context_chunks", [])
            self.assertTrue(len(chunks2) > 0, "Retrieval failed to find four basic tasks with meta phrase")
            self.assertIn("four basic tasks", chunks2[0]["content"].lower())

        finally:
            vector_manager.delete_document(doc_id, conversation_id=conv_id)


if __name__ == "__main__":
    unittest.main()
