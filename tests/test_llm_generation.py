import queue
import unittest
from unittest.mock import patch

import torch

from app import llm


class _FakeStreamer:
    def __init__(self, *_args, **_kwargs):
        self._queue = queue.Queue()

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item

    def end(self):
        self._queue.put(None)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class LlmGenerationTests(unittest.TestCase):
    def test_transformers_worker_error_is_propagated_without_hanging(self):
        class BrokenModel:
            device = "cpu"

            def generate(self, **_kwargs):
                raise ValueError("simulated generation failure")

        state = {"backend": "transformers"}
        with (
            patch.object(llm, "_state", state),
            patch.object(llm, "ensure", return_value=(BrokenModel(), _FakeTokenizer())),
            patch.object(llm, "build_prompt", return_value="prompt"),
            patch.object(llm, "context_length", return_value=1024),
            patch.object(llm, "TextIteratorStreamer", _FakeStreamer),
        ):
            with self.assertRaisesRegex(RuntimeError, "LLM generation failed") as error:
                list(llm.generate_stream([{"role": "user", "content": "hello"}]))

        self.assertIsInstance(error.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
