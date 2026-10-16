"""Prompt construction from the *observable* view only.

`build_interpretation_prompt` takes an `Observable` — not a `Scenario` — so it
is structurally impossible for hidden state, reference labels or factor ids to
reach the model. `tests/test_leakage.py` additionally scans rendered prompts
for any latent strings.
"""
from __future__ import annotations

import hashlib
import json

from pgym.scenarios.schema import Observable

PROMPT_VERSION = "interp-prompt-v2"

SYSTEM = (
    "You are helping someone understand a personal conversation. Read the context and messages carefully. "
    "People often mean more (or less) than they literally say, but indirect language does not always carry a hidden meaning. "
    "Reason about what this specific person most likely means given everything you know, express your uncertainty honestly as "
    "probabilities, and recommend what the listener should do next."
)


def render_conversation(obs: Observable) -> str:
    lines = []
    channel = {"private": "Private messages", "group_chat": "Group chat", "in_person": "In-person conversation"}[obs.channel]
    who = ", ".join(obs.participants) if obs.channel == "group_chat" else obs.speaker
    lines.append(f"Setting: {channel} between you and {who} ({obs.relationship_label}).")
    if obs.context:
        lines.append("\nWhat you know:")
        lines += [f"- {c}" for c in obs.context]
    lines.append("\nConversation:")
    for m in obs.history:
        t = f" [{m.time}]" if m.time else ""
        lines.append(f"{m.sender}{t}: {m.text}")
    return "\n".join(lines)


def build_interpretation_prompt(obs: Observable) -> str:
    opts = "\n".join(f'- "{o.id}": {o.label} — {o.description}' for o in obs.interpretations)
    acts = "\n".join(f'- "{a.id}": {a.label} — {a.description} (e.g. "{a.example}")' for a in obs.actions)
    return (
        f"{render_conversation(obs)}\n\n"
        f"Question: {obs.question}\n\n"
        f"Possible interpretations:\n{opts}\n\n"
        f"Possible next moves for you:\n{acts}\n\n"
        "Return JSON with: `interpretation_probs` (a probability for every interpretation id, summing to 1), "
        "`most_likely` (an interpretation id), `rationale` (2-3 sentences citing the specific context that drove your judgement), "
        "`action` (one move id) and `reply` (the exact message you'd send, if any)."
    )


def response_schema(obs: Observable) -> dict:
    ids = [o.id for o in obs.interpretations]
    return {
        "type": "object",
        "properties": {
            "interpretation_probs": {
                "type": "object",
                "properties": {i: {"type": "number"} for i in ids},
                "required": ids,
                "additionalProperties": False,
            },
            "most_likely": {"type": "string", "enum": ids},
            "rationale": {"type": "string"},
            "action": {"type": "string", "enum": [a.id for a in obs.actions]},
            "reply": {"type": "string"},
        },
        "required": ["interpretation_probs", "most_likely", "rationale", "action", "reply"],
        "additionalProperties": False,
    }


def prompt_hash(system: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([system, prompt]).encode()).hexdigest()[:16]
