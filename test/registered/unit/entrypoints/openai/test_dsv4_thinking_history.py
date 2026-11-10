"""A thinking turn must replay as itself, or the cache freezes at the prompt boundary.

The radix tree stores the ids the model was actually fed, and for a thinking turn
those are: assistant marker, the reasoning block opened, its text, the block closed,
the answer. A later turn reuses anything past that turn only if replaying it
reproduces those ids. Two ways this used to go wrong, both measured:

* drop_thinking removed the reasoning of an earlier assistant turn even though the
  client had sent it, leaving a turn that matches nothing;
* with the reasoning dropped, the marker emitted for that turn was the closing one,
  which is what a turn that never thought looks like.

Either way the cached length stops at the previous prompt while the prompt keeps
growing. What these tests pin is the property the cache needs rather than any
particular token: turn 2's prompt starts with turn 1's prompt, byte for byte.

Run: python -m unittest -v test_dsv4_thinking_history
"""

import unittest

from sglang.srt.entrypoints.openai import encoding_dsv4 as enc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SYSTEM = {"role": "system", "content": "You are a careful repair assistant."}
U1 = {"role": "user", "content": "Why does my treadmill belt drift left?"}
REASONING = "The pulleys are probably out of alignment; check tension second."
ANSWER = "Check the pulley alignment first."
U2 = {"role": "user", "content": "And what is second?"}


def _encode(messages, thinking_mode="thinking", drop_thinking=True):
    # drop_thinking defaults to the server's default, not to what the fix wants: a
    # test that had to ask for drop_thinking=False would not cover the reported path.
    return enc.encode_messages(
        messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking
    )


def _turn1(**kw):
    return _encode([SYSTEM, U1], **kw)


def _turn2(assistant, **kw):
    return _encode([SYSTEM, U1, assistant, U2], **kw)


THINKING_ASSISTANT = {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING}
FORGOTTEN_ASSISTANT = {"role": "assistant", "content": ANSWER}


class TestThinkingHistoryIsReplayable(unittest.TestCase):
    def test_next_turn_starts_with_this_turns_prompt(self):
        first, second = _turn1(), _turn2(THINKING_ASSISTANT)
        self.assertTrue(
            second.startswith(first),
            "turn 2 must contain turn 1 unchanged, or nothing past the assistant turn "
            "can be reused",
        )

    def test_reasoning_block_is_opened_and_closed(self):
        second = _turn2(THINKING_ASSISTANT)
        self.assertIn(
            enc.thinking_start_token + REASONING + enc.thinking_end_token,
            second,
            "the reasoning must sit inside a block that was opened and closed; the "
            "prompt's own instructions mention both markers, so match the block whole",
        )

    def test_the_sent_reasoning_survives_drop_thinking(self):
        """The reported path: server default drop_thinking, client echoed reasoning."""
        second = _turn2(THINKING_ASSISTANT, drop_thinking=True)
        self.assertIn(REASONING, second, "drop_thinking removed a turn that was replayable")

    def test_chat_mode_history_is_replayable_too(self):
        first = _turn1(thinking_mode="chat")
        second = _turn2(FORGOTTEN_ASSISTANT, thinking_mode="chat")
        self.assertTrue(second.startswith(first))

    def test_losing_the_reasoning_breaks_the_prefix(self):
        """Teeth: a client that forgets the reasoning really does break the match."""
        first, second = _turn1(), _turn2(FORGOTTEN_ASSISTANT)
        self.assertFalse(
            second.startswith(first),
            "if a turn without reasoning still matched, the test above proves nothing",
        )

    def test_an_empty_reasoning_block_is_still_a_block(self):
        second = _turn2({"role": "assistant", "content": ANSWER, "reasoning_content": ""})
        self.assertTrue(second.startswith(_turn1()))


if __name__ == "__main__":
    unittest.main()
