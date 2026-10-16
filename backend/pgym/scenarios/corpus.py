"""Load the hand-written corpus and expand its counterfactual variants."""
from __future__ import annotations

from functools import lru_cache

import yaml

from pgym.config import DATA_DIR
from pgym.scenarios.schema import ActionOption, Message, Option, Reference, Scenario
from pgym.social.domains import get_domain

CORPUS_VERSION = "handwritten-v0.3"


def _messages(rows, speaker: str) -> list[Message]:
    return [Message(sender=s, text=t, time=tm, is_you=(s == "You")) for s, t, tm in rows]


def _normalize(d: dict[str, float]) -> dict[str, float]:
    z = sum(d.values())
    return {k: v / z for k, v in d.items()}


def _build(raw: dict, *, sid: str, context, history, reference, rationale, cf_of=None, change=None, hidden=None) -> Scenario:
    dom = get_domain(raw["domain"])
    opts = raw.get("options", {})
    interpretations = [
        Option(id=i.id, label=opts.get(i.id, i.label), description=opts.get(i.id, i.description) if i.id in opts else i.description, literal=i.literal)
        for i in dom.interpretations
    ]
    actions = [ActionOption(**{**a.__dict__, "style": list(a.style)}) for a in dom.actions]
    if raw["domain"] == "reference":
        for a in actions:
            if a.id == "assume_a" and "referent_a" in opts:
                a.label, a.example = f"Go with: {opts['referent_a']}", "Yes! Let's do it."
            if a.id == "assume_b" and "referent_b" in opts:
                a.label, a.example = f"Go with: {opts['referent_b']}", "Yes! Let's do it."
    missing = set(dom.intent_ids) - set(reference)
    if missing:
        raise ValueError(f"{sid}: reference missing {missing}")
    speaker = raw["speaker"]
    return Scenario(
        id=sid,
        family_id=raw["id"],
        domain=raw["domain"],
        capability=raw["capability"],
        capabilities=raw.get("capabilities", [raw["capability"]]),
        title=raw["title"],
        source="handwritten",
        split="eval_handwritten",
        speaker=speaker,
        relationship_label=raw["relationship"],
        channel=raw.get("channel", "private"),
        participants=raw.get("participants", [speaker]),
        context=context,
        history=_messages(history, speaker),
        question=dom.question.replace("{S}", speaker if speaker != "The group" else "the group"),
        interpretations=interpretations,
        actions=actions,
        reference=Reference(distribution=_normalize(reference), source="authored-hypothesis",
                            version=CORPUS_VERSION, rationale=rationale),
        hidden=hidden if hidden is not None else raw.get("hidden", {}),
        why_hidden_matters=raw.get("why_hidden_matters", ""),
        counterfactual_of=cf_of,
        counterfactual_change=change,
    )


@lru_cache(maxsize=1)
def load_corpus() -> tuple[Scenario, ...]:
    raws = yaml.safe_load((DATA_DIR / "scenarios" / "handwritten.yaml").read_text())
    out: list[Scenario] = []
    for raw in raws:
        base = _build(raw, sid=raw["id"], context=raw["context"], history=raw["history"],
                      reference=raw["reference"], rationale=raw["rationale"])
        out.append(base)
        for cf in raw.get("counterfactuals", []):
            ctx = list(cf.get("context_replace", raw["context"])) + list(cf.get("context_add", []))
            hist = cf.get("history_replace", raw["history"])
            out.append(_build(raw, sid=f"{raw['id']}__{cf['id']}", context=ctx, history=hist,
                              reference=cf["reference"], rationale=cf["rationale"],
                              cf_of=raw["id"], change=cf["change"],
                              hidden={"note": "Counterfactual variant: the hidden state of the base scenario is not assumed to carry over."}))
    return tuple(out)


def get_scenario(sid: str) -> Scenario:
    for s in load_corpus():
        if s.id == sid:
            return s
    raise KeyError(sid)
