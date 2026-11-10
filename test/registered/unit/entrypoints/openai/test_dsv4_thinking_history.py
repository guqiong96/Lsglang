"""A thinking turn must replay as itself, or the cache freezes at the prompt boundary.

The radix tree stores the ids the model was actually fed, and for a thinking turn
those are: the assistant marker, the reasoning block opened, its text, the block
closed, the answer. A later turn reuses anything past that turn only if replaying it
reproduces those ids. Two ways this used to go wrong, both measured on V4:

* drop_thinking removed the reasoning of an earlier assistant turn even though the
  client had sent it, leaving a turn that matches nothing;
* with the reasoning dropped, the marker emitted for that turn was the closing one,
  which is what a turn that never thought looks like -- turn 1 ended in the opening
  marker and turn 2 had the closing one at the same id (128821 vs 128822).

Either way the cached length stops at the previous prompt while the prompt keeps
growing. What these tests pin is the property the cache needs rather than any
particular token: turn 2's prompt starts with turn 1's prompt, byte for byte. Both
encoders are checked because the rendering is copied between them.

Run: python -m unittest -v test_dsv4_thinking_history
"""

import importlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SYSTEM = {"role": "system", "content": "You are a careful repair assistant."}
U1 = {"role": "user", "content": "Why does my treadmill belt drift left?"}
U2 = {"role": "user", "content": "And what is second?"}
REASONING = "The pulleys are probably out of alignment; check tension second."
ANSWER = "Check the pulley alignment first."


class _Encodings(unittest.TestCase):
    """Shared body, run once per encoder; subclasses only pick the module."""

    module = None

    @classmethod
    def setUpClass(cls):
        if cls.module is None:
            raise unittest.SkipTest("shared body, run through a subclass")
        cls.enc = importlib.import_module(f"sglang.srt.entrypoints.openai.{cls.module}")

    def _encode(self, messages, thinking_mode="thinking", drop_thinking=True):
        # drop_thinking defaults to the server's default, not to what the fix wants:
        # a test that had to ask for drop_thinking=False would skip the reported path.
        return self.enc.encode_messages(
            messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking
        )

    def _turn1(self, **kw):
        return self._encode([SYSTEM, U1], **kw)

    def _turn2(self, assistant, **kw):
        return self._encode([SYSTEM, U1, assistant, U2], **kw)

    def test_next_turn_starts_with_this_turns_prompt(self):
        first, second = self._turn1(), self._turn2(
            {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING}
        )
        self.assertTrue(
            second.startswith(first),
            "turn 2 must contain turn 1 unchanged, or nothing past the assistant turn "
            "can be reused",
        )

    def test_reasoning_block_is_opened_and_closed(self):
        second = self._turn2(
            {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING}
        )
        self.assertIn(
            self.enc.thinking_start_token + REASONING + self.enc.thinking_end_token,
            second,
            "the reasoning must sit inside a block that was opened and closed; the "
            "prompt's own instructions mention both markers, so match the block whole",
        )

    def test_the_sent_reasoning_survives_drop_thinking(self):
        """The reported path: server default drop_thinking, client echoed reasoning."""
        second = self._turn2(
            {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING},
            drop_thinking=True,
        )
        self.assertIn(REASONING, second, "drop_thinking removed a replayable turn")

    def test_chat_mode_history_is_replayable_too(self):
        first = self._turn1(thinking_mode="chat")
        second = self._turn2(
            {"role": "assistant", "content": ANSWER}, thinking_mode="chat"
        )
        self.assertTrue(second.startswith(first))

    def test_losing_the_reasoning_breaks_the_prefix(self):
        """Teeth: a client that forgets the reasoning really does break the match."""
        first, second = self._turn1(), self._turn2({"role": "assistant", "content": ANSWER})
        self.assertFalse(
            second.startswith(first),
            "if a turn without reasoning still matched, the test above proves nothing",
        )

    def test_an_empty_reasoning_block_is_still_a_block(self):
        second = self._turn2(
            {"role": "assistant", "content": ANSWER, "reasoning_content": ""}
        )
        self.assertTrue(second.startswith(self._turn1()))


class TestDeepSeekV4(_Encodings):
    module = "encoding_dsv4"


class TestDeepSeekV41(_Encodings):
    module = "encoding_dsv41"


if __name__ == "__main__":
    unittest.main()
