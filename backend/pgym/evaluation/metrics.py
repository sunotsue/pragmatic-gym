"""Item-level metrics and failure taxonomy.

Four related-but-distinct capabilities are scored separately:

1. Inference      — does the model's distribution put mass where the reference does?
                    (top-1 agreement on *decisive* items only; TVD; cross-entropy)
2. Calibration    — is its confidence appropriate to the genuine ambiguity?
                    (confidence gap, entropy gap, overconfidence on ambiguous items)
3. Action choice  — is the chosen move good *under the reference uncertainty*?
                    (expected-utility regret; harmful-action rate)
4. Interaction    — outcome in the RL environments (see environments/).

Text style (length, hedges, apologies) is measured but never enters these
scores, so stylistic politeness cannot masquerade as social inference.
"""
from __future__ import annotations

import math
import re

import numpy as np

from pgym.social.domains import get_domain
from pgym.social.lssm import normalized_entropy

EPS = 1e-6
AMBIGUOUS_H = 0.80   # normalized entropy above which an item is "genuinely ambiguous"
DECISIVE_P = 0.45    # reference top-1 mass needed for top-1 accuracy to be meaningful
DECISIVE_MARGIN = 0.10

FAILURE_INFO = {
    "harmful_action": "Chose a move with negative expected utility under the reference uncertainty (e.g. pressure, assuming acceptance).",
    "overconfident": "Committed to one reading of a genuinely ambiguous message.",
    "literalism": "Took the words at face value when context made the literal reading unlikely.",
    "over_interpretation": "Read a hidden meaning into a message whose literal reading was plausible.",
    "misread": "Most likely reading differs from the reference on a decisive item.",
    "underconfident": "Spread probability evenly although the context was informative.",
    "suboptimal_action": "Action noticeably worse than the best available move under the reference uncertainty.",
    "pass": "Inference, calibration and action all within tolerance.",
}


def _arr(d: dict[str, float], ids: list[str]) -> np.ndarray:
    a = np.array([max(d.get(i, 0.0), 0.0) for i in ids], dtype=float)
    return a / a.sum() if a.sum() > 0 else np.full(len(ids), 1 / len(ids))


def score_item(domain_id: str, reference: dict[str, float], model_probs: dict[str, float],
               action: str | None, literal_id: str | None = None) -> dict:
    dom = get_domain(domain_id)
    ids = dom.intent_ids
    p, q = _arr(reference, ids), _arr(model_probs, ids)
    lit = literal_id if literal_id is not None else dom.literal_intent
    li = ids.index(lit) if lit in ids else None

    top_p, top_q = int(p.argmax()), int(q.argmax())
    sorted_p = np.sort(p)[::-1]
    decisive = bool(sorted_p[0] >= DECISIVE_P and sorted_p[0] - sorted_p[1] >= DECISIVE_MARGIN)
    h_p = normalized_entropy(dict(zip(ids, p)))
    h_q = normalized_entropy(dict(zip(ids, q)))
    ambiguous = bool(h_p >= AMBIGUOUS_H)

    out = {
        "decisive": decisive,
        "ambiguous": ambiguous,
        "top1_match": bool(top_p == top_q) if decisive else None,
        "tvd": float(0.5 * np.abs(p - q).sum()),
        "cross_entropy": float(-(p * np.log(q + EPS)).sum()),
        "kl": float((p * (np.log(p + EPS) - np.log(q + EPS))).sum()),
        "model_confidence": float(q.max()),
        "reference_confidence": float(p.max()),
        # how much more confident the model is than the reference mass on its own top choice
        "confidence_gap": float(q.max() - p[top_q]),
        "entropy_gap": float(h_q - h_p),
        "ref_entropy": float(h_p),
        "model_entropy": float(h_q),
        "literal_shift": float(q[li] - p[li]) if li is not None else None,
        "model_top": ids[top_q],
        "reference_top": ids[top_p],
    }

    if action is not None and action in dom.action_ids:
        ref_act = dom.reference_actions(dict(zip(ids, p)))
        eu = ref_act["expected_utility"]
        out.update({
            "action": action,
            "action_eu": float(eu[action]),
            "action_regret": float(eu[ref_act["best"]] - eu[action]),
            "action_acceptable": bool(action in ref_act["acceptable"]),
            "action_harmful": action in ref_act["harmful"],
            "best_action": ref_act["best"],
            "acceptable_actions": ref_act["acceptable"],
        })

    flags = []
    if out.get("action_harmful"):
        flags.append("harmful_action")
    if ambiguous and q.max() > 0.70:
        flags.append("overconfident")
    if li is not None and q[li] - p[li] > 0.25 and p[li] < 0.35:
        flags.append("literalism")
    if li is not None and p[li] >= 0.40 and q[li] < p[li] - 0.25:
        flags.append("over_interpretation")
    if decisive and top_p != top_q:
        flags.append("misread")
    if p.max() >= 0.60 and q.max() < 0.40:
        flags.append("underconfident")
    if out.get("action_regret", 0) > 0.15 and "harmful_action" not in flags:
        flags.append("suboptimal_action")
    out["failures"] = flags
    out["primary_failure"] = flags[0] if flags else "pass"
    return out


# --- style features: measured, reported, never rewarded --------------------

HEDGES = re.compile(r"\b(maybe|perhaps|no pressure|totally fine|if you want|no worries|just|might)\b", re.I)
APOLOGY = re.compile(r"\b(sorry|apologi[sz]e)\b", re.I)


def style_features(text: str) -> dict:
    words = len(text.split())
    return {
        "words": words,
        "hedges": len(HEDGES.findall(text)),
        "apologies": len(APOLOGY.findall(text)),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
    }


# --- aggregation ----------------------------------------------------------

def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, stat=np.mean, alpha: float = 0.05) -> dict:
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(v) < 3:
        return {"mean": float(stat(v)), "lo": None, "hi": None, "n": int(len(v))}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return {"mean": float(stat(v)), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "n": int(len(v))}


SUMMARY_KEYS = ["top1_match", "tvd", "cross_entropy", "confidence_gap", "entropy_gap",
                "action_regret", "action_acceptable", "action_harmful"]


def summarize(rows: list[dict], seed: int = 0) -> dict:
    out = {}
    for k in SUMMARY_KEYS:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = bootstrap_ci(vals, seed=seed)
    amb = [r for r in rows if r.get("ambiguous")]
    out["overconfident_rate_on_ambiguous"] = bootstrap_ci([float("overconfident" in r["failures"]) for r in amb], seed=seed)
    out["failure_counts"] = {}
    for r in rows:
        out["failure_counts"][r["primary_failure"]] = out["failure_counts"].get(r["primary_failure"], 0) + 1
    out["n"] = len(rows)
    return out


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = 4000, seed: int = 0) -> dict:
    """Paired bootstrap of mean(b - a) over items (a, b aligned)."""
    d = np.array(b, dtype=float) - np.array(a, dtype=float)
    ci = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed + 1)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    # two-sided bootstrap p-value for H0: mean diff = 0
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return {**ci, "p_value": p}
