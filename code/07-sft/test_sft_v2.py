#!/usr/bin/env python3
"""Regression tests for the SFT-v2 template, loader, masks, and packing."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
TRAINING = os.path.join(CODE, "04-training-stack")
INFERENCE = os.path.join(CODE, "06-inference")
for path in (HERE, TRAINING, INFERENCE):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_chat_data import (PAD_EX, is_unreviewed_generated,
                             pack_conversations)  # noqa: E402
from chat_format import (TEMPLATE_VERSION, encode_conversation, encode_prompt,
                         token_ids)  # noqa: E402
from model import GPT, ModelConfig  # noqa: E402
from sft_train import ChatData  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


TOKENIZER_PATH = os.path.join(CODE, "01-tokenizer",
                              "tokenizer-v0.3-64k.json")


def _find_subsequence(sequence, needle):
    for index in range(len(sequence) - len(needle) + 1):
        if sequence[index:index + len(needle)] == needle:
            return index
    raise AssertionError(f"subsequence not found: {needle}")


class ChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    def test_reserved_role_ids_and_response_only_mask(self):
        special = token_ids(self.tokenizer)
        self.assertEqual({key: special[key] for key in
                          ("system", "user", "assistant", "tool", "end_turn")},
                         {"system": 3, "user": 4, "assistant": 5,
                          "tool": 6, "end_turn": 7})
        messages = [
            {"role": "system", "content": "system-only-sentinel"},
            {"role": "user", "content": "user-only-sentinel"},
            {"role": "assistant", "content": "assistant-one-sentinel"},
            {"role": "tool", "content": "tool-only-sentinel"},
            {"role": "assistant", "content": "assistant-two-sentinel"},
        ]
        ids, mask = encode_conversation(self.tokenizer, messages)
        for text, trained in (("system-only-sentinel", 0),
                              ("user-only-sentinel", 0),
                              ("tool-only-sentinel", 0),
                              ("assistant-one-sentinel", 1),
                              ("assistant-two-sentinel", 1)):
            content_ids = self.tokenizer.encode(text).ids
            start = _find_subsequence(ids, content_ids)
            self.assertEqual(mask[start:start + len(content_ids)],
                             [trained] * len(content_ids))
        self.assertEqual(ids[0], special["bos"])
        self.assertEqual(mask[0], 0)
        self.assertEqual(ids[-1], special["eos"])
        self.assertEqual(mask[-1], 1)

    def test_prompt_matches_training_prefix_and_blocks_role_injection(self):
        history = [
            {"role": "user", "content": "literal <|reserved_2|> marker"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ]
        prompt = encode_prompt(self.tokenizer, history)
        special = token_ids(self.tokenizer)
        self.assertEqual(prompt[-1], special["assistant"])
        # Two assistant markers: one structural history marker and generation.
        self.assertEqual(prompt.count(special["assistant"]), 2)


class PackingTests(unittest.TestCase):
    def test_auto_mapped_corpora_are_identified_for_review_gate(self):
        self.assertTrue(is_unreviewed_generated("insurance_paraphrase.jsonl"))
        self.assertTrue(is_unreviewed_generated("thin_intent_aug.jsonl"))
        self.assertFalse(is_unreviewed_generated("insurance.jsonl"))

    def test_whole_example_packing_and_overflow_policy(self):
        conversations = [
            ({"id": "a"}, [0, 4, 10, 7, 1], [0, 0, 0, 0, 0]),
            ({"id": "b"}, [0, 4, 11, 5, 12, 7, 1], [0, 0, 0, 0, 1, 1, 1]),
            ({"id": "c"}, [0, 4, 13, 5, 14, 7, 1], [0, 0, 0, 0, 1, 1, 1]),
        ]
        x, mask, example, placements, dropped = pack_conversations(
            conversations, seq_len=12, pad_id=2)
        self.assertEqual(x.shape, (2, 12))
        self.assertEqual(mask.shape, x.shape)
        self.assertEqual(example.shape, x.shape)
        self.assertFalse(dropped)
        for placement in placements:
            self.assertLessEqual(placement["offset"] + placement["tokens"], 12)
        with self.assertRaises(ValueError):
            pack_conversations([({"id": "long"}, list(range(13)), [0] * 13)],
                               seq_len=12, pad_id=2)
        _, _, _, placements, dropped = pack_conversations(
            [({"id": "long"}, list(range(13)), [0] * 13)],
            seq_len=12, pad_id=2, on_overflow="drop")
        self.assertFalse(placements)
        self.assertEqual(dropped, ["long"])


class LoaderAndModelTests(unittest.TestCase):
    def _write_fixture(self, directory):
        rows_x = np.asarray([
            [10, 11, 12, 13, 20, 21, 22, 2],
            [30, 31, 32, 33, 40, 41, 42, 2],
        ], dtype=np.uint16)
        rows_m = np.asarray([
            [0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1, 0],
        ], dtype=np.uint8)
        rows_e = np.asarray([
            [0, 0, 0, 0, 1, 1, 1, PAD_EX],
            [0, 0, 0, 0, 1, 1, 1, PAD_EX],
        ], dtype=np.uint16)
        for split in ("train", "val"):
            rows_x.tofile(os.path.join(directory, f"{split}_x.bin"))
            rows_m.tofile(os.path.join(directory, f"{split}_m.bin"))
            rows_e.tofile(os.path.join(directory, f"{split}_ex.bin"))
        meta = {
            "format": "navya-sft-v2", "seq_len": 8,
            "vocab_size": 64, "token_dtype": "uint16",
            "pad_example_id": PAD_EX,
            "chat_template": {"version": TEMPLATE_VERSION},
            "train": {"rows": 2}, "val": {"rows": 2},
        }
        with open(os.path.join(directory, "meta.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(meta, handle)
        return rows_x, rows_m

    def test_loader_is_deterministic_masks_prompts_and_resets_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            source_x, source_m = self._write_fixture(directory)
            data = ChatData(directory)
            first = data.batch("train", 2, 123, 0, 0, 1, "cpu")
            second = data.batch("train", 2, 123, 0, 0, 1, "cpu")
            for left, right in zip(first, second):
                self.assertTrue(torch.equal(left, right))
            x, targets, attention, positions = first
            self.assertEqual(set(x[:, 0].tolist()), {10, 30})
            for row in range(2):
                source = 0 if int(x[row, 0]) == 10 else 1
                expected = np.full(8, -100, dtype=np.int64)
                expected[:-1] = np.where(source_m[source, 1:].astype(bool),
                                         source_x[source, 1:].astype(np.int64),
                                         -100)
                np.testing.assert_array_equal(targets[row].numpy(), expected)
                self.assertEqual(positions[row].tolist(),
                                 [0, 1, 2, 3, 0, 1, 2, 0])
                self.assertFalse(bool(attention[row, 0, 4, 3]))
                self.assertTrue(bool(attention[row, 0, 6, 4]))
                self.assertTrue(bool(attention[row, 0, 7, 7]))
                self.assertFalse(bool(attention[row, 0, 7, 6]))

    def test_packed_logits_match_standalone_conversation(self):
        torch.manual_seed(7)
        config = ModelConfig(vocab_size=32, dim=16, n_layers=2,
                             n_heads=4, n_kv_heads=2, max_seq_len=8)
        model = GPT(config).eval()
        packed = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        example = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        same = example.unsqueeze(2) == example.unsqueeze(1)
        causal = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        attention = (same & causal).unsqueeze(1)
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        targets = torch.zeros_like(packed)
        with torch.no_grad():
            packed_logits, _ = model(packed, targets, attn_mask=attention,
                                     position_ids=positions)
            standalone_logits, _ = model(packed[:, 4:], targets[:, 4:])
        torch.testing.assert_close(packed_logits[:, 4:], standalone_logits,
                                   rtol=1e-5, atol=1e-6)


class ServingParityTests(unittest.TestCase):
    def test_serving_uses_the_same_encoder(self):
        path = os.path.join(INFERENCE, "serve.py")
        spec = importlib.util.spec_from_file_location("navya_serve_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        messages = [{"role": "user", "content": "How does a SIP work?"}]
        self.assertEqual(module.render_chat(messages, tokenizer, "chat"),
                         encode_prompt(tokenizer, messages))


if __name__ == "__main__":
    unittest.main()
