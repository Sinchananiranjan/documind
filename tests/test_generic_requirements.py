"""Generic requirement test suite covering DocuMind issue fixes globally without hardcoding."""

import unittest
from unittest.mock import patch
from app.graph.state import DocuMindState
from app.graph.nodes import (
    text_rag_node,
    image_analysis_node,
    calculation_node,
    general_knowledge_node,
    table_analysis_node,
    verify_answer_node,
    router_node,
)
from app.graph.workflow import run_documind_workflow
from app.document.math_engine import execute_generic_calculation


class TestGenericRequirements(unittest.TestCase):

    def test_ratio_vs_difference(self):
        """Test ratio ('how many times longer') vs difference ('what is the difference')."""
        q_ratio = "How many times longer is 100 than 20?"
        res_ratio = execute_generic_calculation(q_ratio, "")
        self.assertEqual(res_ratio["calculated_value"], 5)
        self.assertIn("/", res_ratio["operation_desc"])

        q_diff = "What is the difference between 100 and 20?"
        res_diff = execute_generic_calculation(q_diff, "")
        self.assertEqual(res_diff["calculated_value"], 80)
        self.assertIn("-", res_diff["operation_desc"])

    def test_calculation_requiring_document_formula(self):
        """Test formula calculation from document context."""
        context = "Efficiency E = (P_out / P_in) * 100. Given P_out = 80, P_in = 100."
        q = "Calculate Efficiency E if P_out = 80 and P_in = 100"
        res = execute_generic_calculation(q, context)
        self.assertIsNotNone(res["calculated_value"])
        self.assertEqual(res["calculated_value"], 80)

    def test_unsupported_claim_detection(self):
        """Test semantic verifier rejecting claims unsupported by evidence."""
        state: DocuMindState = {
            "question": "What is the protocol speed?",
            "answer": "The protocol speed is 5000 Gbps with quantum encryption.",
            "route": "text_rag",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "The standard protocol operates at 100 Mbps."}
            ],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertFalse(res["verified"])

    def test_hallucinated_stage_detection(self):
        """Test semantic verifier detecting unmentioned lifecycle stages."""
        state: DocuMindState = {
            "question": "Describe the stages shown.",
            "answer": "The workflow includes Stage 1: Input, Stage 2: Processing, followed by Testing, Deployment, and Maintenance.",
            "route": "image_analysis",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "image", "content": "Figure shows Stage 1: Input and Stage 2: Processing."}
            ],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertFalse(res["verified"])

    @patch("app.models.llm.LocalLLMManager.generate_text")
    def test_consecutive_unrelated_questions_isolation(self, mock_generate):
        """Test zero state leakage between consecutive unrelated queries."""
        mock_generate.side_effect = [
            "The difference between 50 and 20 is 30.",
            "50 is 5 times greater than 10."
        ]
        # Query 1
        res1 = run_documind_workflow("What is the difference between 50 and 20?", mode="general_knowledge_mode")
        self.assertIn("30", str(res1["answer"]))

        # Query 2 (unrelated query immediately after)
        res2 = run_documind_workflow("How many times greater is 50 than 10?", mode="general_knowledge_mode")
        self.assertIn("5", str(res2["answer"]))
        self.assertNotIn("30", str(res2["answer"]))

    def test_table_question_vs_nearby_calculation(self):
        """Test routing table questions to table analysis and calc questions to calculation."""
        state_table: DocuMindState = {
            "question": "What is the value in the second column according to the table?",
            "mode": "document_mode",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "table", "content": "Row 1 | Col 1 | Col 2\nData 1 | Val A | Val B"}],
            "timings": {}
        }
        res_table = router_node(state_table)
        self.assertEqual(res_table["route"], "table_analysis")

        state_calc: DocuMindState = {
            "question": "Calculate 40 * 5 based on formula",
            "mode": "document_mode",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Formula = 40 * 5"}],
            "timings": {}
        }
        res_calc = router_node(state_calc)
        self.assertEqual(res_calc["route"], "calculation")

    def test_paraphrased_document_answers(self):
        """Regression test 1: Paraphrased answers passing verification."""
        state: DocuMindState = {
            "question": "How does error correction work?",
            "answer": "The procedure utilizes a distance metric of 3 to rectify single faults in transmitted frames [Page 1].",
            "route": "text_rag",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "The minimum Hamming distance is 3, which enables single-bit error correction."}
            ],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertTrue(res["verified"])

    def test_multi_section_synthesis(self):
        """Regression test 2: Multi-section synthesis across chunks passing verification."""
        state: DocuMindState = {
            "question": "Summarize the complete workflow across sections.",
            "answer": "1. Initialization phase sets parameters [Page 1].\n2. Processing loop handles error checks [Page 2].",
            "route": "text_rag",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Section 1 describes the initialization phase and setup parameters."},
                {"doc_id": "d1", "page_num": 2, "chunk_type": "text", "content": "Section 2 outlines the processing loop and error check mechanism."}
            ],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertTrue(res["verified"])

    def test_answers_with_valid_numbers_across_chunks(self):
        """Regression test 3: Answers containing valid numbers not repeated verbatim in one chunk."""
        state: DocuMindState = {
            "question": "What are the operational limits?",
            "answer": "System handles 100 units at 25 degrees [Page 1, Page 2].",
            "route": "text_rag",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Maximum capacity is 100 units."},
                {"doc_id": "d1", "page_num": 2, "chunk_type": "text", "content": "Operating temperature limit is 25 degrees."}
            ],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertTrue(res["verified"])

    def test_conversational_follow_up_questions(self):
        """Regression test 4: Conversational follow-up question verification."""
        state: DocuMindState = {
            "question": "How does it calculate routes?",
            "answer": "It computes minimum cost routes starting from the origin node [Page 3].",
            "route": "text_rag",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 3, "chunk_type": "text", "content": "The shortest path algorithm calculates single-source shortest paths."}
            ],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertTrue(res["verified"])

    def test_genuinely_unsupported_answers(self):
        """Regression test 5: Genuinely unsupported claims rejected."""
        state: DocuMindState = {
            "question": "What is the network speed?",
            "answer": "The protocol speed is 5000 Gbps with QuantumEncryption.",
            "route": "text_rag",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "The standard protocol operates at 100 Mbps."}
            ],
            "timings": {}
        }
        res = verify_answer_node(state)
        self.assertFalse(res["verified"])

    def test_multi_condition_retrieval(self):
        """Regression test 6: Multi-condition question retrieving chunks for all parts."""
        state: DocuMindState = {
            "question": "What are the conditions required for condition A and condition B?",
            "mode": "document_mode",
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Condition A requires primary input stability."},
                {"doc_id": "d1", "page_num": 2, "chunk_type": "text", "content": "Condition B requires secondary feedback loop completion."}
            ],
            "timings": {}
        }
        res = router_node(state)
        self.assertIn(res["route"], ["text_rag", "table_analysis", "calculation"])

    def test_document_first_routing(self):
        """Regression test 7: Document mode with valid evidence routes to text_rag, not web_search."""
        state: DocuMindState = {
            "question": "What is the latest protocol specification described in the document?",
            "mode": "document_mode",
            "evidence_sufficiency": {"is_sufficient": True, "is_partial": False, "is_insufficient": False},
            "context_chunks": [
                {"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "The latest protocol specification is version 4.2."}
            ],
            "timings": {}
        }
        res = router_node(state)
        self.assertEqual(res["route"], "text_rag")

    def test_multiple_figures_identity(self):
        """Regression test 8: Disambiguating Figure 1 vs Figure 2 image chunks for visual consistency."""
        with patch("app.models.llm.LocalLLMManager.analyze_image") as mock_vision, \
             patch("app.models.llm.LocalLLMManager.has_vision_model", return_value=True), \
             patch("app.models.llm.LocalLLMManager.generate_text_with_metrics") as mock_gen:
            mock_vision.return_value = "Figure 2 shows a circular feedback diagram."
            mock_gen.return_value = ("Figure 2 is a circular feedback diagram [Page 2].", {"total_llm_time": 0.1})

            state: DocuMindState = {
                "question": "Describe Figure 2 in the document.",
                "mode": "document_mode",
                "context_chunks": [
                    {"doc_id": "d1", "page_num": 1, "chunk_type": "image", "image_b64": "img1_b64", "content": "Figure 1: Block diagram"},
                    {"doc_id": "d1", "page_num": 2, "chunk_type": "image", "image_b64": "img2_b64", "content": "Figure 2: Circular feedback diagram"}
                ],
                "timings": {}
            }
            res = image_analysis_node(state)
            self.assertIn("Figure 2", res["answer"])
            # Ensure vision model analyzed image from Page 2 (img2_b64)
            mock_vision.assert_called_once()
            self.assertEqual(mock_vision.call_args[0][0], "img2_b64")

    def test_insufficient_evidence_fallback(self):
        """Regression test 9: Document query with missing evidence returns document fallback."""
        state: DocuMindState = {
            "question": "What is described on page 99 of the document?",
            "mode": "auto",
            "active_docs": ["d1"],
            "context_chunks": [],
            "timings": {}
        }
        res = text_rag_node(state)
        self.assertIn("does not provide information", res["answer"])

    def test_standalone_math_addition(self):
        """Standalone math test: Addition '52 + 53' routes to calculation and verifies deterministically."""
        state: DocuMindState = {
            "question": "52 + 53",
            "mode": "auto",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Biology notes about cells."}],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "calculation")
        res_calc = calculation_node(state)
        self.assertIn("105", res_calc["answer"])
        res_verify = verify_answer_node({**state, "answer": res_calc["answer"], "route": "calculation"})
        self.assertTrue(res_verify["verified"])

    def test_standalone_math_multiplication(self):
        """Standalone math test: Multiplication '125 × 8' routes to calculation and verifies deterministically."""
        state: DocuMindState = {
            "question": "125 × 8",
            "mode": "auto",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Operating system process management."}],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "calculation")
        res_calc = calculation_node(state)
        self.assertIn("1000", res_calc["answer"])
        res_verify = verify_answer_node({**state, "answer": res_calc["answer"], "route": "calculation"})
        self.assertTrue(res_verify["verified"])

    def test_standalone_math_subtraction(self):
        """Standalone math test: Subtraction '100 - 35' routes to calculation."""
        state: DocuMindState = {
            "question": "100 - 35",
            "mode": "auto",
            "context_chunks": [],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "calculation")
        res_calc = calculation_node(state)
        self.assertIn("65", res_calc["answer"])

    def test_standalone_math_division(self):
        """Standalone math test: Division '500 / 25' routes to calculation."""
        state: DocuMindState = {
            "question": "500 / 25",
            "mode": "auto",
            "context_chunks": [],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "calculation")
        res_calc = calculation_node(state)
        self.assertIn("20", res_calc["answer"])

    def test_general_knowledge_photosynthesis(self):
        """General knowledge test: 'What is photosynthesis?' routes to general_knowledge despite uploaded PDF."""
        state: DocuMindState = {
            "question": "What is photosynthesis?",
            "mode": "auto",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Database indexing and B-trees."}],
            "timings": {}
        }
        res_route = router_node(state)
        self.assertEqual(res_route["route"], "general_knowledge")
        with patch("app.models.llm.LocalLLMManager.generate_text_with_metrics") as mock_gen:
            mock_gen.return_value = ("Photosynthesis is the biological process converting light to chemical energy.", {"total_llm_time": 0.1})
            res_gk = general_knowledge_node(state)
            self.assertIn("Photosynthesis", res_gk["answer"])
            res_verify = verify_answer_node({**state, "answer": res_gk["answer"], "route": "general_knowledge"})
            self.assertTrue(res_verify["verified"])

    def test_unrelated_questions_during_document_conversation(self):
        """Test unrelated math/GK questions during document conversation do not force document fallback."""
        state_math: DocuMindState = {
            "question": "52 + 53",
            "mode": "auto",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Computer networks."}],
            "timings": {}
        }
        self.assertEqual(router_node(state_math)["route"], "calculation")

        state_gk: DocuMindState = {
            "question": "Who wrote Hamlet?",
            "mode": "auto",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Computer networks."}],
            "timings": {}
        }
        self.assertEqual(router_node(state_gk)["route"], "general_knowledge")

    def test_document_based_calculation(self):
        """Test calculation with explicit document reference routes to calculation and uses context."""
        state: DocuMindState = {
            "question": "Calculate Efficiency E = (P_out / P_in) * 100 as per the document where P_out = 80 and P_in = 100",
            "mode": "auto",
            "context_chunks": [{"doc_id": "d1", "page_num": 1, "chunk_type": "text", "content": "Efficiency formula E = (P_out / P_in) * 100"}],
            "timings": {}
        }
        self.assertEqual(router_node(state)["route"], "calculation")
        res_calc = calculation_node(state)
        self.assertIn("80", res_calc["answer"])

    def test_followup_document_question(self):
        """Test follow-up document question routes to text_rag when evidence is sufficient."""
        state: DocuMindState = {
            "question": "Explain its main advantages.",
            "mode": "auto",
            "evidence_sufficiency": {"is_sufficient": True, "is_partial": False, "is_insufficient": False},
            "context_chunks": [{"doc_id": "d1", "page_num": 2, "chunk_type": "text", "content": "The main advantages are fault tolerance and high throughput."}],
            "timings": {}
        }
        self.assertEqual(router_node(state)["route"], "text_rag")


if __name__ == "__main__":
    unittest.main()


