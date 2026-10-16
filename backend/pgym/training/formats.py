"""Shared multiple-choice format for local-model scoring *and* SFT.

Using the exact same format for training targets and evaluation is
deliberate: evaluation reads the model's *next-token distribution* over
option letters, so

* the measured interpretation distribution is the model's actual predictive
  distribution, not a verbalized number, and
* reply style (length, hedging, apologies) cannot influence the score at all.

Option order is permuted (seeded) so neither training nor evaluation can
exploit position. Evaluation averages over all cyclic permutations.
"""
from __future__ import annotations

import hashlib
import random

from pgym.models.prompts import SYSTEM, render_conversation
from pgym.scenarios.schema import Observable

LETTERS = "ABCDEFGH"
FORMAT_VERSION = "mc-v1"


def permutation(n: int, key: str) -> list[int]:
    r = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    idx = list(range(n))
    r.shuffle(idx)
    return idx


def cyclic(n: int, shift: int) -> list[int]:
    return [(i + shift) % n for i in range(n)]


def mc_user_prompt(obs: Observable, kind: str, order: list[int], include_context: bool = True) -> str:
    """kind = 'interpret' | 'act'. `order[k]` = index of the option shown as letter k."""
    if include_context:
        convo = render_conversation(obs)
    else:  # 'isolated' ablation: the utterance alone, no relationship or history
        last = obs.history[-1]
        convo = f"Conversation:\n{last.sender}: {last.text}"
    if kind == "interpret":
        opts = obs.interpretations
        lines = [f"{LETTERS[k]}. {opts[i].label} — {opts[i].description}" for k, i in enumerate(order)]
        q = f"{obs.question}\n" + "\n".join(lines) + "\nAnswer with a single letter."
    else:
        opts = obs.actions
        lines = [f"{LETTERS[k]}. {opts[i].label}: \"{opts[i].example}\"" for k, i in enumerate(order)]
        q = "What should you do next?\n" + "\n".join(lines) + "\nAnswer with a single letter."
    return f"{convo}\n\n{q}"


def chat_messages(obs: Observable, kind: str, order: list[int], answer: str | None = None, include_context: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": mc_user_prompt(obs, kind, order, include_context)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs
