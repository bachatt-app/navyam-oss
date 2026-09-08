"""Regression tests for the public chat/inference response contract."""

import importlib.util
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_module(name, relative_path):
    path = os.path.join(ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serve = load_module("navya_serve_contract", "code/06-inference/serve.py")
consumer = load_module("navya_consumer_contract", "apps/consumer/server.py")
build_reasoning = load_module(
    "navya_build_reasoning_contract",
    "code/13-reasoning/build_reasoning_traces.py")
format_reasoning = load_module(
    "navya_format_reasoning_contract",
    "code/13-reasoning/format_reasoning_sft.py")


class IdentityGuardTests(unittest.TestCase):
    def test_explicit_identity_queries_match(self):
        queries = [
            "Who are you?",
            "What is your name?",
            "Tell me about yourself",
            "tum kaun ho?",
            "aapka naam kya hai?",
            "आप कौन हैं?",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(serve.is_identity_query([
                    {"role": "user", "content": query},
                ]))

    def test_contextual_finance_followups_do_not_match(self):
        queries = [
            "achha, lekin phir dono alag kyun hain?",
            "toh mere liye kaun sa better hai?",
            "what are the tax implications?",
            "who pays the GST here?",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertFalse(serve.is_identity_query([
                    {"role": "user", "content": "Explain SIP vs lumpsum"},
                    {"role": "assistant", "content": "They differ in timing."},
                    {"role": "user", "content": query},
                ]))

    def test_tool_router_does_not_reintroduce_identity_false_positive(self):
        self.assertIsNone(serve.tool_router.route([
            {"role": "user", "content": "Explain SIP vs lumpsum"},
            {"role": "assistant", "content": "They differ in timing."},
            {"role": "user", "content":
             "achha, lekin phir dono alag kyun hain?"},
        ]))

    def test_conceptual_repo_rate_question_reaches_the_model(self):
        self.assertIsNone(serve.tool_router.route([
            {"role": "user", "content":
             "If repo rate rises, why can a floating loan tenure increase?"},
        ]))


class ReasoningContractTests(unittest.TestCase):
    def test_inference_splits_complete_think_block(self):
        answer, reasoning = serve.split_reasoning_output(
            "<think>Compare timing and cash flow.</think>\n\nSIP is periodic."
        )
        self.assertEqual(answer, "SIP is periodic.")
        self.assertEqual(reasoning, "Compare timing and cash flow.")

    def test_consumer_supports_legacy_inline_reasoning(self):
        answer, reasoning = consumer.split_reasoning(
            "<think>Check the rule.</think>\n\nHere is the answer."
        )
        self.assertEqual(answer, "Here is the answer.")
        self.assertEqual(reasoning, "Check the rule.")

    def test_incomplete_reasoning_is_not_silently_hidden(self):
        raw = "<think>This generation was truncated"
        self.assertEqual(serve.split_reasoning_output(raw), (raw, ""))
        self.assertEqual(consumer.split_reasoning(raw), (raw, ""))

    def test_consumer_reports_upstream_model_and_reasoning(self):
        result = consumer.completion_result({
            "model": "checkpoint-that-actually-answered",
            "choices": [{"message": {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Generated rationale",
            }}],
        })
        self.assertEqual(result, {
            "model": "checkpoint-that-actually-answered",
            "reply": "Final answer",
            "reasoning": "Generated rationale",
        })

    def test_reasoning_sft_uses_canonical_not_teacher_final(self):
        conversation = format_reasoning.make_conversation({
            "prompt": [{"role": "user", "content": "What is PPF?"}],
            "reasoning": "Teacher rationale",
            "answer": "Incorrect teacher final",
            "canonical": "Canonical source answer",
        })
        self.assertEqual(
            conversation["messages"][1]["content"],
            "<think>Teacher rationale</think>\n\nCanonical source answer")

    def test_reasoning_trace_rejects_script_mismatch(self):
        ok, reason = build_reasoning.acceptable(
            "Mujhe PPF samjhao", "पीपीएफ एक बचत योजना है।")
        self.assertFalse(ok)
        self.assertEqual(reason, "language mismatch")


if __name__ == "__main__":
    unittest.main()
