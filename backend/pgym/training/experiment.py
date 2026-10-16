"""The post-training experiment: does *which* social experience you train on matter?

Pre-registered comparison (see RESEARCH.md §4):

  H1  At a matched token budget, counterfactual-contrast training reduces
      cross-entropy to the reference interpretation distribution on the
      *generalisation* sets (unseen families, unseen domain, hand-written)
      more than random selection.            H0: no difference.
  H2  Context-rich examples beat the same experiences with context stripped
      ("isolated"), at the same token budget. H0: no difference.
  H3  Failure-targeted selection beats random. H0: no difference.

Test: paired bootstrap over evaluation items (predictions averaged over
seeds), Holm–Bonferroni across the hypotheses. Secondary outcomes: context
sensitivity, paraphrase robustness, calibration, action regret, and the
regression tripwire.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pgym.config import EXPERIMENTS_DIR
from pgym.evaluation.metrics import bootstrap_ci, paired_bootstrap_diff
from pgym.selection.pool import POOL_DIR, read_jsonl
from pgym.social.domains import get_domain

EVAL_SPLITS = ["eval_id", "eval_ood_family", "eval_ood_domain", "eval_handwritten"]
GENERALISATION = ["eval_ood_family", "eval_ood_domain", "eval_handwritten"]
RUNS = EXPERIMENTS_DIR / "runs"
RESULTS = EXPERIMENTS_DIR / "results"


def eval_items() -> dict[str, list[dict]]:
    return {s: read_jsonl(POOL_DIR / f"{s}.jsonl") for s in EVAL_SPLITS}


def evaluate_checkpoint(model_id: str, adapter: str | None, out_path: Path, cfg: dict) -> dict:
    from pgym.models.local_hf import load
    from pgym.training.local_eval import score_items, score_regression
    load.cache_clear()
    tok, model = load(model_id, adapter, cfg["eval"]["dtype"])
    t0 = time.time()
    preds = {}
    for split, items in eval_items().items():
        preds[split] = score_items(tok, model, items, action_perms=cfg["eval"]["action_perms"])
    reg = score_regression(tok, model)
    out = {"model": model_id, "adapter": adapter, "preds": preds, "regression": reg, "eval_runtime_s": round(time.time() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    del model
    load.cache_clear()
    return out


# ------------------------------------------------------------------ metrics
def _key_vec(probs: dict, ids: list[str]) -> np.ndarray:
    return np.array([probs[i] for i in ids])


def split_metrics(items: list[dict], preds: list[dict]) -> dict:
    sc = [p["score"] for p in preds]
    m = {
        "cross_entropy": bootstrap_ci([s["cross_entropy"] for s in sc]),
        "kl": bootstrap_ci([s["kl"] for s in sc]),
        "tvd": bootstrap_ci([s["tvd"] for s in sc]),
        "top1_decisive": bootstrap_ci([float(s["top1_match"]) for s in sc if s["top1_match"] is not None]),
        "confidence_gap": bootstrap_ci([s["confidence_gap"] for s in sc]),
        "overconfident_on_ambiguous": bootstrap_ci([float("overconfident" in s["failures"]) for s in sc if s["ambiguous"]]),
        "action_regret": bootstrap_ci([s.get("action_regret", 0.0) for s in sc]),
        "harmful_action_rate": bootstrap_ci([float(s.get("action_harmful", False)) for s in sc]),
        "literalism_rate": bootstrap_ci([float("literalism" in s["failures"]) for s in sc]),
    }
    # deference bait: picking a 'deferential'-styled move that is not acceptable
    bait = []
    for it, p in zip(items, preds):
        styles = {a["id"]: a.get("style", []) for a in it["observable"]["actions"]}
        if any("deferential" in v for v in styles.values()):
            bait.append(float("deferential" in styles.get(p["action"], []) and not p["score"].get("action_acceptable", True)))
    m["unwarranted_deference_rate"] = bootstrap_ci(bait)
    # context sensitivity & paraphrase robustness from clusters
    by_id = {it["id"]: (it, p) for it, p in zip(items, preds)}
    clusters: dict[str, list] = {}
    for it in items:
        if it.get("cluster"):
            clusters.setdefault(it["cluster"], []).append(it["id"])
    dm, dr, para = [], [], []
    for cid, ids in clusters.items():
        anchor = next((by_id[i] for i in ids if by_id[i][0].get("contrast_factor") is None), None)
        if not anchor:
            continue
        a_it, a_p = anchor
        intents = [o["id"] for o in a_it["observable"]["interpretations"]]
        for i in ids:
            it, p = by_id[i]
            if it is a_it:
                continue
            if it.get("contrast_factor") == "__paraphrase__":
                para.append(0.5 * float(np.abs(_key_vec(p["probs"], intents) - _key_vec(a_p["probs"], intents)).sum()))
            else:
                dm.extend((_key_vec(p["probs"], intents) - _key_vec(a_p["probs"], intents)).tolist())
                dr.extend((_key_vec(it["reference"], intents) - _key_vec(a_it["reference"], intents)).tolist())
    if len(dm) > 3 and np.std(dm) > 1e-9:
        m["context_sensitivity_r"] = float(np.corrcoef(dm, dr)[0, 1])
        m["sensitivity_ratio"] = float(np.mean(np.abs(dm)) / max(1e-9, np.mean(np.abs(dr))))
    else:
        m["context_sensitivity_r"], m["sensitivity_ratio"] = None, None
    m["paraphrase_tvd"] = bootstrap_ci(para)
    by_cap: dict[str, list] = {}
    for it, s in zip(items, sc):
        by_cap.setdefault(it["capability"], []).append(s["cross_entropy"])
    m["cross_entropy_by_capability"] = {k: float(np.mean(v)) for k, v in by_cap.items()}
    m["n"] = len(items)
    return m


def reference_baselines() -> dict:
    """Non-learned predictors that bound what is achievable *without reading context*.

    uniform          equal mass on every reading
    utterance_prior  the hypothesis' distribution for the bare utterance (family base, all
                     factors at their 'no information' default) — knows the words, ignores the
                     relationship and history. Hand-written items have no family prior, so
                     uniform is used there.
    """
    from pgym.evaluation.metrics import score_item
    from pgym.social.generator import FAMILY_BY_ID, full_levels
    from pgym.social.lssm import posterior
    out = {}
    for split, items in eval_items().items():
        rows = {"uniform": [], "utterance_prior": []}
        for it in items:
            ids = [o["id"] for o in it["observable"]["interpretations"]]
            uni = {i: 1 / len(ids) for i in ids}
            fam = FAMILY_BY_ID.get(it["family_id"])
            prior = posterior(fam.domain, fam.base, full_levels(fam.domain, {})) if fam else uni
            for name, q in (("uniform", uni), ("utterance_prior", prior)):
                rows[name].append(score_item(it["domain"], it["reference"], q, None))
        out[split] = {name: {"cross_entropy": float(np.mean([r["cross_entropy"] for r in rs])),
                             "kl": float(np.mean([r["kl"] for r in rs])),
                             "top1_decisive": float(np.mean([float(r["top1_match"]) for r in rs if r["top1_match"] is not None] or [np.nan]))}
                      for name, rs in rows.items()}
    return out


def arm_report(run: dict) -> dict:
    ev = eval_items()
    return {"splits": {s: split_metrics(ev[s], run["preds"][s]) for s in EVAL_SPLITS},
            "regression": {k: run["regression"][k] for k in ("accuracy", "p_correct", "n")}}


def _avg_preds(runs: list[dict]) -> dict[str, list[float]]:
    """Seed-averaged per-item cross-entropy, aligned across arms by item order."""
    out = {}
    for s in EVAL_SPLITS:
        arr = np.array([[p["score"]["cross_entropy"] for p in r["preds"][s]] for r in runs])
        out[s] = arr.mean(axis=0).tolist()
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        adj[name] = running
    return adj


def aggregate(exp_id: str, arms: dict[str, list[dict]], meta: dict) -> dict:
    """arms: arm -> list of run dicts (one per seed; base has one)."""
    reports = {a: [arm_report(r) for r in runs] for a, runs in arms.items()}
    summary = {}
    for a, reps in reports.items():
        summary[a] = {"n_seeds": len(reps), "splits": {}, "regression_accuracy": float(np.mean([r["regression"]["accuracy"] for r in reps]))}
        for s in EVAL_SPLITS:
            keys = ["cross_entropy", "kl", "tvd", "top1_decisive", "confidence_gap", "overconfident_on_ambiguous",
                    "action_regret", "harmful_action_rate", "literalism_rate", "unwarranted_deference_rate", "paraphrase_tvd"]
            d = {}
            for k in keys:
                vals = [r["splits"][s][k]["mean"] for r in reps if r["splits"][s][k]["mean"] is not None]
                seed_mean = float(np.mean(vals)) if vals else None
                item_ci = reps[0]["splits"][s][k] if len(reps) == 1 else None
                d[k] = {"mean": seed_mean, "seed_values": vals,
                        "seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                        "item_ci": [item_ci["lo"], item_ci["hi"]] if item_ci else None}
            for k in ("context_sensitivity_r", "sensitivity_ratio"):
                vals = [r["splits"][s][k] for r in reps if r["splits"][s][k] is not None]
                d[k] = {"mean": float(np.mean(vals)) if vals else None, "seed_values": vals}
            caps = {}
            for r in reps:
                for c, v in r["splits"][s]["cross_entropy_by_capability"].items():
                    caps.setdefault(c, []).append(v)
            d["cross_entropy_by_capability"] = {c: float(np.mean(v)) for c, v in caps.items()}
            summary[a]["splits"][s] = d
    # hypothesis tests on the pooled generalisation sets
    ce = {a: _avg_preds(runs) for a, runs in arms.items()}

    def pooled(a):
        return sum((ce[a][s] for s in GENERALISATION), [])

    tests, pvals = {}, {}
    for name, (treat, ctrl) in {"H1_contrast_vs_random": ("counterfactual_contrast", "random"),
                                "H2_context_vs_isolated": ("random", "isolated"),
                                "H3_failure_vs_random": ("failure_targeted", "random")}.items():
        if treat in ce and ctrl in ce:
            # diff = treat - ctrl in cross-entropy (negative = treatment better)
            r = paired_bootstrap_diff(pooled(ctrl), pooled(treat))
            tests[name] = {"treatment": treat, "control": ctrl, "metric": "cross-entropy to reference (nats), pooled generalisation sets",
                           "diff_treat_minus_control": r["mean"], "ci": [r["lo"], r["hi"]], "p_value": r["p_value"], "n_items": r["n"]}
            pvals[name] = r["p_value"]
    for name, p in holm(pvals).items():
        tests[name]["p_holm"] = p
    for a in ce:
        if a != "base" and "base" in ce:
            r = paired_bootstrap_diff(pooled("base"), pooled(a))
            tests[f"vs_base:{a}"] = {"treatment": a, "control": "base", "diff_treat_minus_control": r["mean"], "ci": [r["lo"], r["hi"]], "p_value": r["p_value"]}
    result = {"experiment_id": exp_id, "meta": meta, "summary": summary, "tests": tests,
              "reference_baselines": reference_baselines(),
              "train_meta": {a: [r.get("train_meta") for r in runs] for a, runs in arms.items()},
              "label": "REAL MEASUREMENT — local LoRA fine-tuning and evaluation of " + meta.get("base_model", ""),
              "reference_note": "Targets and evaluation references are authored hypotheses (LSSM / hand-written), not human data."}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{exp_id}.json").write_text(json.dumps(result, indent=1))
    return result


def run(exp_id: str, cfg: dict, arms: list[str] | None = None, seeds: list[int] | None = None, budget: int | None = None,
        target_failure: str | None = None, log=print) -> dict:
    from pgym.selection.strategies import diagnostics, export, select
    from pgym.training.sft import train
    arms = arms or cfg["arms"]
    seeds = seeds if seeds is not None else cfg["seeds"]
    budget = budget or cfg["selection"]["token_budget"]
    root = RUNS / exp_id
    collected: dict[str, list[dict]] = {}
    if "base" in arms:
        p = root / "base" / "eval.json"
        if not p.exists():
            log("[base] evaluating base checkpoint")
            evaluate_checkpoint(cfg["base_model"], None, p, cfg)
        collected["base"] = [json.loads(p.read_text())]
    for seed in seeds:
        for arm in arms:
            if arm == "base":
                continue
            d = root / f"{arm}_s{seed}"
            ev = d / "eval.json"
            if not ev.exists():
                log(f"[{arm} s{seed}] selecting (budget {budget})")
                sel = select(arm, budget=budget, seed=seed, target_failure=target_failure)
                diag = diagnostics(sel)
                (d / "data").mkdir(parents=True, exist_ok=True)
                (d / "data" / "diagnostics.json").write_text(json.dumps(diag, indent=1))
                (d / "data" / "selection.json").write_text(json.dumps({k: v for k, v in sel.items() if k != "rows"} | {"items": [{"id": r["item"]["id"], "reason": r["reason"]} for r in sel["rows"]]}, indent=1))
                export(sel, d / "data", n_interp_samples=cfg["selection"]["interp_samples_per_item"], seed=seed)
                log(f"[{arm} s{seed}] training on {sel['n']} items / {sel['tokens_used']} est. tokens")
                tm = train(str(d / "data"), str(d / "adapter"), seed=seed, config=cfg)
                log(f"[{arm} s{seed}] trained in {tm['runtime_s']}s (loss {tm['train_loss']:.3f}); evaluating")
                evaluate_checkpoint(cfg["base_model"], str(d / "adapter"), ev, cfg)
            r = json.loads(ev.read_text())
            tmp = d / "adapter" / "train_meta.json"
            r["train_meta"] = json.loads(tmp.read_text()) if tmp.exists() else None
            collected.setdefault(arm, []).append(r)
        meta = {"base_model": cfg["base_model"], "budget": budget, "seeds_completed": [s for s in seeds if s <= seed],
                "arms": arms, "lora": cfg["lora"], "train": cfg["train"], "target_failure": target_failure,
                "eval_sizes": {s: len(v) for s, v in eval_items().items()}}
        res = aggregate(exp_id, collected, meta)
        log(f"[aggregate] seeds {meta['seeds_completed']} -> {RESULTS / (exp_id + '.json')}")
    return res
