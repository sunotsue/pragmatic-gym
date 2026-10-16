"""Latent Social State Model (LSSM).

A transparent, log-linear *hypothesis* about how observable context shifts the
distribution over a speaker's intent:

    logit(intent | context) = base[family][intent]
                              + sum_f effect[domain][f][level][intent]
                              + sum_interactions delta[intent]
    P(intent | context) = softmax(logit)

Uses of the LSSM inside Pragmatic Gym:

1. Counterfactual Lab: the *hypothesised* reference distribution for each
   cell, against which model behaviour is compared ("does the model move in
   the direction the hypothesis predicts, and by how much?").
2. Social RL environments: the prior from which the hidden intent is sampled.
3. Training targets: the posterior given *observable* context — never the
   sampled hidden intent — so SFT targets teach calibrated inference rather
   than overconfident guessing of a realised latent.

The numbers are authored by the researcher (version-pinned below). They are
not estimates from human data. The project's methodology is explicitly
designed so that the LSSM can be *replaced* by distributions fitted to human
annotations (see data/eval/human_annotations.schema.json) without changing any
downstream code.
"""
from __future__ import annotations

import math

LSSM_VERSION = "lssm-v0.3-authored"

E = dict[str, float]

EFFECTS: dict[str, dict[str, dict[str, E]]] = {
    "invitation": {
        "relationship": {
            "romantic_interest": {"keeping_options": 0.4, "undecided": 0.1},
            "acquaintance": {"soft_decline": 0.4, "follow_through": -0.2},
            "family_member": {"follow_through": 0.2, "undecided": 0.2},
        },
        "familiarity": {"stranger": {"soft_decline": 0.3, "keeping_options": 0.3}, "established": {"follow_through": 0.2}},
        "shared_history": {
            "repeated_disappointment": {"soft_decline": 1.0, "follow_through": -0.6},
            "established_trust": {"follow_through": 1.0, "soft_decline": -0.5},
            "prior_conflict": {"soft_decline": 0.6, "undecided": 0.2},
        },
        "stakes": {"low": {"undecided": 0.2}, "high": {"soft_decline": 0.3, "undecided": 0.2}},
        "comm_style": {
            "direct": {"undecided": 0.6, "soft_decline": -0.6},
            "indirect": {"soft_decline": 0.5},
            "avoidant": {"soft_decline": 0.6, "keeping_options": 0.3},
        },
        "prior_pattern": {"once": {"soft_decline": 0.4}, "repeated": {"soft_decline": 1.2, "follow_through": -0.5}},
        "audience": {"group_chat": {"soft_decline": 0.4, "undecided": 0.1}, "public": {"soft_decline": 0.5}},
        "norms": {"indirect_norm": {"soft_decline": 1.5}, "literal_norm": {"undecided": 1.2, "soft_decline": -0.8}},
        "info_asymmetry": {"asymmetric": {"keeping_options": 0.5, "soft_decline": 0.2}},
        "commitment": {"casual": {"follow_through": 0.2}, "explicit": {"follow_through": 0.6}},
        "directness": {"explicit": {"soft_decline": 4.0}},
        "social_pressure": {"medium": {"soft_decline": 0.2}, "high": {"soft_decline": 0.5}},
    },
    "relationship": {
        "familiarity": {"stranger": {"prefers_status_quo": 0.3}, "established": {"content_open": 0.2}},
        "shared_history": {
            "repeated_disappointment": {"avoiding_talk": 0.8, "prefers_status_quo": 0.4},
            "established_trust": {"content_open": 0.8},
            "prior_conflict": {"avoiding_talk": 0.5},
        },
        "stakes": {"low": {"content_open": 0.3}, "high": {"hesitant_wants_more": 0.3}},
        "comm_style": {
            "direct": {"content_open": 0.7, "avoiding_talk": -0.5},
            "indirect": {"hesitant_wants_more": 0.4, "prefers_status_quo": 0.3},
            "avoidant": {"avoiding_talk": 0.9},
        },
        "prior_pattern": {
            "once": {"avoiding_talk": 0.5, "prefers_status_quo": 0.3},
            "repeated": {"avoiding_talk": 1.2, "prefers_status_quo": 0.6, "content_open": -0.6},
        },
        "audience": {"group_chat": {"avoiding_talk": 0.4}, "public": {"avoiding_talk": 0.4, "content_open": 0.2}},
        "norms": {"indirect_norm": {"prefers_status_quo": 0.4, "avoiding_talk": 0.4}, "literal_norm": {"content_open": 1.0}},
        "info_asymmetry": {"asymmetric": {"prefers_status_quo": 0.4, "avoiding_talk": 0.3}},
        "commitment": {"casual": {"content_open": 0.2, "hesitant_wants_more": 0.3}, "explicit": {"content_open": 0.9, "prefers_status_quo": -0.6}},
        "directness": {"explicit": {"prefers_status_quo": 4.0}},
        "social_pressure": {"medium": {"avoiding_talk": 0.3}, "high": {"avoiding_talk": 0.6}},
    },
    "emotional": {
        "relationship": {
            "romantic_interest": {"hurt_face_saving": 0.4},
            "acquaintance": {"genuinely_fine": 0.5},
            "family_member": {"disappointed_understanding": 0.2},
        },
        "familiarity": {"stranger": {"genuinely_fine": 0.4}},
        "shared_history": {
            "repeated_disappointment": {"frustrated_pattern": 1.0, "hurt_face_saving": 0.4},
            "established_trust": {"genuinely_fine": 0.7, "disappointed_understanding": 0.3},
            "prior_conflict": {"hurt_face_saving": 0.5, "frustrated_pattern": 0.5},
        },
        "stakes": {"low": {"genuinely_fine": 0.8}, "high": {"hurt_face_saving": 0.8, "disappointed_understanding": 0.5, "genuinely_fine": -0.7}},
        "comm_style": {
            "direct": {"genuinely_fine": 0.9, "hurt_face_saving": -0.6},
            "indirect": {"hurt_face_saving": 0.5, "disappointed_understanding": 0.2},
            "avoidant": {"hurt_face_saving": 0.7, "frustrated_pattern": 0.3},
        },
        "prior_pattern": {"once": {"disappointed_understanding": 0.3, "frustrated_pattern": 0.4}, "repeated": {"frustrated_pattern": 1.4, "genuinely_fine": -0.6}},
        "audience": {"group_chat": {"hurt_face_saving": 0.4}, "public": {"hurt_face_saving": 0.4}},
        "norms": {"indirect_norm": {"hurt_face_saving": 0.6, "disappointed_understanding": 0.3}, "literal_norm": {"genuinely_fine": 1.2}},
        "info_asymmetry": {"asymmetric": {"frustrated_pattern": 0.5, "hurt_face_saving": 0.3}},
        "commitment": {"casual": {"disappointed_understanding": 0.3}, "explicit": {"frustrated_pattern": 0.7, "hurt_face_saving": 0.4}},
        "directness": {"explicit": {"disappointed_understanding": 4.0}},
    },
    "family": {
        "shared_history": {
            "repeated_disappointment": {"mild_reproach": 0.8, "indirect_request": 0.4},
            "established_trust": {"neutral_observation": 0.6, "concern_wellbeing": 0.4},
            "prior_conflict": {"mild_reproach": 0.7},
        },
        "stakes": {"low": {"neutral_observation": 0.8}, "high": {"indirect_request": 0.8, "concern_wellbeing": 0.2}},
        "comm_style": {
            "direct": {"neutral_observation": 0.8, "indirect_request": -0.4},
            "indirect": {"indirect_request": 0.7},
            "avoidant": {"indirect_request": 0.4, "mild_reproach": 0.2},
        },
        "prior_pattern": {"once": {"indirect_request": 0.4}, "repeated": {"indirect_request": 0.8, "mild_reproach": 0.6}},
        "audience": {"group_chat": {"neutral_observation": 0.2, "mild_reproach": 0.2}, "public": {"mild_reproach": 0.6}},
        "norms": {"indirect_norm": {"indirect_request": 0.9}, "literal_norm": {"neutral_observation": 1.0}},
        "info_asymmetry": {"asymmetric": {"indirect_request": 0.7}},
        "directness": {"explicit": {"indirect_request": 4.0}},
        "social_pressure": {"medium": {"mild_reproach": 0.3}, "high": {"mild_reproach": 0.6, "indirect_request": 0.3}},
    },
}

# Authored interaction hypotheses: (conditions, delta)
INTERACTIONS: dict[str, list[tuple[dict[str, str], E]]] = {
    "invitation": [
        ({"relationship": "close_friend", "shared_history": "established_trust"}, {"follow_through": 0.4}),
        ({"audience": "group_chat", "prior_pattern": "repeated"}, {"soft_decline": 0.4}),
        ({"relationship": "romantic_interest", "stakes": "high"}, {"keeping_options": 0.4}),
    ],
    "relationship": [({"comm_style": "avoidant", "prior_pattern": "repeated"}, {"avoiding_talk": 0.5})],
    "emotional": [
        ({"relationship": "romantic_interest", "stakes": "high"}, {"hurt_face_saving": 0.5}),
        ({"comm_style": "direct", "prior_pattern": "repeated"}, {"frustrated_pattern": 0.6}),
    ],
    "family": [({"norms": "indirect_norm", "stakes": "high"}, {"indirect_request": 0.5})],
}


def softmax(logits: dict[str, float]) -> dict[str, float]:
    m = max(logits.values())
    ex = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def logits(domain: str, base: dict[str, float], levels: dict[str, str]) -> dict[str, float]:
    out = dict(base)
    eff = EFFECTS.get(domain, {})
    for f, lv in levels.items():
        for intent, d in eff.get(f, {}).get(lv, {}).items():
            out[intent] = out.get(intent, 0.0) + d
    for cond, delta in INTERACTIONS.get(domain, []):
        if all(levels.get(k) == v for k, v in cond.items()):
            for intent, d in delta.items():
                out[intent] = out.get(intent, 0.0) + d
    return out


def posterior(domain: str, base: dict[str, float], levels: dict[str, str]) -> dict[str, float]:
    return softmax(logits(domain, base, levels))


def explain(domain: str, base: dict[str, float], levels: dict[str, str]) -> list[dict]:
    """Audit trail: every term that contributed to the hypothesised logits."""
    terms = [{"term": "base", "delta": dict(base)}]
    eff = EFFECTS.get(domain, {})
    for f, lv in levels.items():
        d = eff.get(f, {}).get(lv)
        if d:
            terms.append({"term": f"{f}={lv}", "delta": d})
    for cond, delta in INTERACTIONS.get(domain, []):
        if all(levels.get(k) == v for k, v in cond.items()):
            terms.append({"term": " × ".join(f"{k}={v}" for k, v in cond.items()), "delta": delta})
    return terms


def entropy(dist: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def normalized_entropy(dist: dict[str, float]) -> float:
    k = len(dist)
    return entropy(dist) / math.log(k) if k > 1 else 0.0
