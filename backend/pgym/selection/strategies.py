"""Data-selection strategies at a matched token budget.

All strategies draw from the *same* candidate pool and stop at the same
training-token budget, so arms differ only in *which experiences* they
contain. Every selected example carries a human-readable `reason`.

random                 uniform sample                                  (control)
diversity              k-center greedy over TF-IDF of the observable text
failure_targeted       highest base-model loss, optionally focused on one failure mode
counterfactual_contrast whole contrast clusters (anchor + one-factor neighbours), ranked by
                       how much the single changed factor moves the reference
isolated               the random arm's items with relationship/history stripped (ablation:
                       "isolated refusal examples" vs context-rich ones)
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from pgym.scenarios.schema import Observable
from pgym.selection.pool import POOL_DIR, read_jsonl
from pgym.training.formats import mc_user_prompt

STRATEGIES = ["random", "diversity", "failure_targeted", "counterfactual_contrast", "isolated"]
STRATEGY_INFO = {
    "random": "Uniform sample from the candidate pool. The control arm.",
    "diversity": "Greedy farthest-point (k-center) selection over TF-IDF vectors of the observable conversation: maximises coverage of surface variety.",
    "failure_targeted": "Highest base-model loss against the reference (cross-entropy + action regret), optionally focused on one diagnosed failure mode, at most two per contrast cluster.",
    "counterfactual_contrast": "Selects whole contrast clusters — the same conversation with one social factor changed — ranked by how far that single change moves the reference interpretation.",
    "isolated": "Ablation: the random arm's experiences with relationship context and history removed (utterance only). Tests whether context-rich examples teach more than isolated ones at the same token budget.",
}
SEQS_PER_ITEM = 5  # 4 interpretation samples (different option orders) + 1 action example


@lru_cache(maxsize=1)
def pool() -> tuple[dict, ...]:
    return tuple(read_jsonl(POOL_DIR / "train_pool.jsonl"))


@lru_cache(maxsize=1)
def base_scores() -> dict[str, dict]:
    p = POOL_DIR / "base_scores.jsonl"
    if not p.exists():
        return {}
    return {r["id"]: r for r in read_jsonl(p)}


def _text(it: dict, include_context: bool = True) -> str:
    return mc_user_prompt(Observable(**it["observable"]), "interpret", list(range(len(it["observable"]["interpretations"]))), include_context)


@lru_cache(maxsize=4096)
def _cost_cached(item_id: str, include_context: bool) -> int:
    it = next(x for x in pool() if x["id"] == item_id)
    # ~3.6 chars/token for Qwen-style BPE on English chat text, + chat-template overhead
    return int(len(_text(it, include_context)) / 3.6 + 90) * SEQS_PER_ITEM


def token_cost(it: dict, include_context: bool = True) -> int:
    return _cost_cached(it["id"], include_context)


def _take_until(items, budget: int, include_context: bool = True):
    out, used = [], 0
    for it, reason in items:
        c = token_cost(it, include_context)
        if used + c > budget:
            continue
        out.append((it, reason))
        used += c
    return out, used


def select(strategy: str, budget: int = 60000, seed: int = 0, target_failure: str | None = None,
           domains: list[str] | None = None) -> dict:
    P = [it for it in pool() if not domains or it["domain"] in domains]
    rng = random.Random(seed)
    include_context = strategy != "isolated"
    if strategy in ("random", "isolated"):
        order = P[:]
        rng.shuffle(order)
        chosen, used = _take_until([(it, "Uniformly sampled (control)." if strategy == "random" else "Uniformly sampled; context and history stripped (ablation).") for it in order], budget, include_context)
    elif strategy == "diversity":
        chosen, used = _diversity(P, budget, rng)
    elif strategy == "failure_targeted":
        chosen, used = _failure(P, budget, target_failure)
    elif strategy == "counterfactual_contrast":
        chosen, used = _contrast(P, budget)
    else:
        raise ValueError(strategy)
    rows = []
    for it, reason in chosen:
        bs = base_scores().get(it["id"])
        rows.append({"item": it, "reason": reason, "base": bs, "tokens": token_cost(it, include_context)})
    sel_id = hashlib.sha1(json.dumps([strategy, budget, seed, target_failure, domains]).encode()).hexdigest()[:10]
    return {"selection_id": sel_id, "strategy": strategy, "budget": budget, "tokens_used": used, "seed": seed,
            "target_failure": target_failure, "include_context": include_context, "n": len(rows), "rows": rows,
            "info": STRATEGY_INFO[strategy]}


def _diversity(P, budget, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [_text(it) for it in P]
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    X = X.toarray().astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    first = rng.randrange(len(P))
    sel = [first]
    mind = 1 - X @ X[first]
    chosen, used = [], 0
    chosen.append((P[first], "Seed point for farthest-point traversal."))
    used += token_cost(P[first])
    while used < budget:
        mind[sel] = -1
        j = int(mind.argmax())
        if mind[j] <= 0:
            break
        c = token_cost(P[j])
        if used + c > budget:
            mind[j] = -1
            if (mind > 0).sum() == 0:
                break
            continue
        chosen.append((P[j], f"Farthest-point pick: cosine distance {mind[j]:.2f} from everything already selected."))
        used += c
        sel.append(j)
        mind = np.minimum(mind, 1 - X @ X[j])
    return chosen, used


def _failure(P, budget, target_failure):
    bs = base_scores()
    if not bs:
        raise RuntimeError("Base-model pool scores not found. Run scripts/score_pool.py first.")
    scored = []
    for it in P:
        b = bs.get(it["id"])
        if not b:
            continue
        s = b["score"]
        loss = s["cross_entropy"] + 0.5 * s.get("action_regret", 0.0)
        hit = target_failure in s["failures"] if target_failure else bool(set(s["failures"]) - {"pass"})
        score = loss * (2.0 if hit else 1.0)
        reason = f"Base model loss {s['cross_entropy']:.2f} nats (regret {s.get('action_regret', 0):.2f}); failures: {', '.join(s['failures']) or 'none'}"
        if target_failure and hit:
            reason += f" — matches targeted failure '{target_failure}'."
        scored.append((score, it, reason))
    scored.sort(key=lambda x: -x[0])
    per_cluster = Counter()
    ordered = []
    for score, it, reason in scored:
        if per_cluster[it.get("cluster")] >= 2 and it.get("cluster"):
            continue
        per_cluster[it.get("cluster")] += 1
        ordered.append((it, reason))
    return _take_until(ordered, budget)


def _tvd(a, b):
    return 0.5 * sum(abs(a[k] - b.get(k, 0)) for k in a)


def _contrast(P, budget):
    clusters = defaultdict(list)
    for it in P:
        if it.get("cluster"):
            clusters[it["cluster"]].append(it)
    ranked = []
    for cid, members in clusters.items():
        anchor = next((m for m in members if m.get("contrast_factor") is None), None)
        if not anchor or len(members) < 2:
            continue
        deltas = [(m["contrast_factor"], _tvd(anchor["reference"], m["reference"])) for m in members if m is not anchor]
        score = max(d for _, d in deltas)
        ranked.append((score, cid, anchor, members, deltas))
    ranked.sort(key=lambda x: -x[0])
    out, used = [], 0
    for score, cid, anchor, members, deltas in ranked:
        cost = sum(token_cost(m) for m in members)
        if used + cost > budget:
            continue
        desc = "; ".join(f"{f} → ΔTVD {d:.2f}" for f, d in deltas)
        for m in members:
            role = "anchor" if m is anchor else f"one-factor counterfactual ({m['contrast_factor']})"
            out.append((m, f"Contrast cluster {cid} [{role}]. Changing a single factor moves the reference: {desc}."))
        used += cost
    return out, used


# --------------------------------------------------------------- diagnostics
LATENT_PATTERNS = ["true_intent", "private_state", "lssm", "patience", "hidden", "force_intent", "reference_distribution"]


def _shingles(text: str, n: int = 5) -> set:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def diagnostics(selection: dict, eval_sets: list[str] | None = None) -> dict:
    rows = selection["rows"]
    items = [r["item"] for r in rows]
    inc = selection.get("include_context", True)
    texts = [_text(it, inc) for it in items]
    # leakage: machine identifiers from the latent payload must never appear in
    # model-visible text. Natural-language words that legitimately occur in the
    # public option labels (e.g. "undecided", "hidden message") are not leaks.
    from pgym.social.factors import FACTORS
    level_ids = {lv.id for f in FACTORS for lv in f.levels if "_" in lv.id} | {f.id for f in FACTORS if "_" in f.id}
    leaks = []
    for it, t in zip(items, texts):
        public = " ".join(o["label"] + " " + o.get("description", "") for o in it["observable"]["interpretations"]).lower()
        intent_ids = {o["id"] for o in it["observable"]["interpretations"] if "_" in o["id"]}
        patterns = [p for p in LATENT_PATTERNS if p not in public] + sorted(intent_ids | level_ids)
        for pat in patterns:
            if re.search(rf"\b{re.escape(pat)}\b", t, flags=re.I):
                leaks.append({"id": it["id"], "pattern": pat})
    # duplicates
    exact = Counter(texts)
    exact_dups = sum(c - 1 for c in exact.values() if c > 1)
    sh = [_shingles(t) for t in texts]
    near = 0
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            if sh[i] and sh[j] and len(sh[i] & sh[j]) / len(sh[i] | sh[j]) >= 0.9:
                near += 1
    # imbalance
    tops = Counter(max(it["reference"], key=it["reference"].get) for it in items)
    acts = Counter(it["target_action"] for it in items)
    doms = Counter(it["domain"] for it in items)
    fams = Counter(it["family_id"] for it in items)
    caps = Counter(it["capability"] for it in items)
    srcs = Counter(it["source"] for it in items)
    n = max(1, len(items))
    warnings = []
    for name, c in (("target interpretation", tops), ("target action", acts), ("domain", doms)):
        k, v = c.most_common(1)[0] if c else ("", 0)
        if v / n > 0.5:
            warnings.append(f"{name} imbalance: '{k}' is {v / n:.0%} of examples")
    # train/test overlap (families, cells, near-duplicates)
    overlap = {}
    for es in eval_sets or ["eval_id", "eval_ood_family", "eval_ood_domain", "eval_handwritten"]:
        ev = read_jsonl(POOL_DIR / f"{es}.jsonl")
        fam_overlap = sorted({it["family_id"] for it in items} & {e["family_id"] for e in ev})
        cell_overlap = len({(it["family_id"], it.get("cell")) for it in items if it.get("cell")} & {(e["family_id"], e.get("cell")) for e in ev if e.get("cell")})
        ev_sh = [_shingles(_text(e)) for e in ev[:400]]
        near_x = 0
        for a in sh:
            for b in ev_sh:
                if a and b and len(a & b) / len(a | b) >= 0.9:
                    near_x += 1
                    break
        overlap[es] = {"family_overlap": fam_overlap, "cell_overlap": cell_overlap, "near_duplicate_items": near_x,
                       "expected_family_overlap": es == "eval_id"}
        if fam_overlap and es != "eval_id":
            warnings.append(f"family overlap with {es}: {fam_overlap}")
        if cell_overlap and es == "eval_id":
            warnings.append(f"{cell_overlap} factor cells shared with eval_id")
    return {
        "n": len(items), "tokens": selection["tokens_used"],
        "leakage": {"n_leaks": len(leaks), "examples": leaks[:10], "checked_patterns": LATENT_PATTERNS + ["<snake_case intent, factor and level ids>"]},
        "duplicates": {"exact": exact_dups, "near_jaccard_0.9": near},
        "distribution": {"target_interpretation": dict(tops), "target_action": dict(acts), "domain": dict(doms),
                         "family": dict(fams), "capability": dict(caps), "source": dict(srcs)},
        "overlap": overlap,
        "factor_coverage": _factor_coverage(items),
        "warnings": warnings,
    }


def _factor_coverage(items: list[dict]) -> dict:
    cov = defaultdict(Counter)
    for it in items:
        for f, lv in (it.get("factors") or {}).items():
            cov[f][lv] += 1
    return {f: dict(c) for f, c in cov.items()}


# --------------------------------------------------------------- export
def export(selection: dict, out_dir, n_interp_samples: int = 2, seed: int = 0) -> dict:
    """Write SFT (prompt/completion) and preference (chosen/rejected) JSONL."""
    from pathlib import Path

    from pgym.training.formats import LETTERS, chat_messages, permutation
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    inc = selection.get("include_context", True)
    sft, pref = [], []
    for r in selection["rows"]:
        it = r["item"]
        obs = Observable(**it["observable"])
        ids = [o.id for o in obs.interpretations]
        aids = [a.id for a in obs.actions]
        meta = {
            "scenario_id": it["scenario_id"], "item_id": it["id"], "family_id": it["family_id"], "domain": it["domain"],
            "capability": it["capability"], "split": "train", "selection_strategy": selection["strategy"],
            "selection_reason": r["reason"], "provenance": it["provenance"], "factors": it.get("factors"),
            "relationship": obs.relationship_label, "channel": obs.channel,
            "reference_distribution": it["reference"], "target_action": it["target_action"],
            "model_action": (r.get("base") or {}).get("action"), "source": it["source"],
            "outcome": it.get("behaviour_response"), "reward_components": it.get("behaviour_reward_components"),
            "include_context": inc,
        }
        for k in range(n_interp_samples):
            order = permutation(len(ids), f"{it['id']}|interp|{k}|{seed}")
            target = rng.choices(ids, weights=[it["reference"][i] for i in ids])[0]  # sample from the posterior: soft labels
            letter = LETTERS[order.index(ids.index(target))]
            msgs = chat_messages(obs, "interpret", order, include_context=inc)
            sft.append({"prompt": msgs, "completion": [{"role": "assistant", "content": letter}], "meta": {**meta, "kind": "interpret", "target": target}})
        order = permutation(len(aids), f"{it['id']}|act|{seed}")
        tgt = it["target_action"]
        letter = LETTERS[order.index(aids.index(tgt))]
        msgs = chat_messages(obs, "act", order, include_context=inc)
        sft.append({"prompt": msgs, "completion": [{"role": "assistant", "content": letter}], "meta": {**meta, "kind": "act", "target": tgt}})
        base_act = (r.get("base") or {}).get("action")
        rej = base_act if base_act and base_act != tgt else None
        if rej is None:  # most harmful alternative as a contrastive negative
            from pgym.social.domains import get_domain
            eu = get_domain(it["domain"]).expected_utility(it["reference"])
            rej = min(eu, key=eu.get)
        pref.append({"prompt": msgs, "chosen": [{"role": "assistant", "content": letter}],
                     "rejected": [{"role": "assistant", "content": LETTERS[order.index(aids.index(rej))]}],
                     "meta": {**meta, "kind": "act_preference", "chosen_action": tgt, "rejected_action": rej}})
    with (out_dir / "sft.jsonl").open("w") as f:
        for x in sft:
            f.write(json.dumps(x) + "\n")
    with (out_dir / "preference.jsonl").open("w") as f:
        for x in pref:
            f.write(json.dumps(x) + "\n")
    manifest = {"selection_id": selection["selection_id"], "strategy": selection["strategy"], "budget": selection["budget"],
                "tokens_used": selection["tokens_used"], "n_items": selection["n"], "n_sft": len(sft), "n_pref": len(pref),
                "seed": seed, "include_context": inc,
                "sha1": hashlib.sha1((out_dir / "sft.jsonl").read_bytes()).hexdigest()[:12]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
