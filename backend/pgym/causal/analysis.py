"""Estimators for factorial prompt-intervention experiments.

Outcome Y = the model's probability on a chosen interpretation (e.g. P(polite
no)). For every estimand we report the model's estimate with a cluster
bootstrap CI (clusters = surface variants) *and* the hypothesised effect
implied by the authored LSSM, so one can ask both "does context move the
model?" and "does it move the model the way the hypothesis says it should?".
"""
from __future__ import annotations

import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd

from pgym.config import DATA_DIR
from pgym.social.lssm import normalized_entropy

N_BOOT = 1000
MIN_CLUSTERS = 3


def frame(design: dict, results: list[dict]) -> pd.DataFrame:
    """results: one dict per unit with `probs` (model distribution) and `action`."""
    k = design["outcome_intent"]
    rows = []
    for u, res in zip(design["units"], results):
        if res is None:
            continue
        probs = res["probs"]
        rows.append({**u["assignment"], "cell": u["cell"], "variant": u["variant"], "rep": u["rep"],
                     "y": float(probs.get(k, 0.0)), "y_ref": float(u["reference"].get(k, 0.0)),
                     "h": normalized_entropy(probs), "h_ref": normalized_entropy(u["reference"]),
                     "top": max(probs, key=probs.get), "action": res.get("action")})
    return pd.DataFrame(rows)


def _masked_means(df: pd.DataFrame, masks: list, col: str) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array([df.loc[m, col].sum() / m.sum() if m.sum() else np.nan for m in masks])


def _cluster_boot(df: pd.DataFrame, masks: list, combine, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Cluster bootstrap (clusters = surface variants) of a statistic built from means of masked subsets.

    Resampling clusters only reweights per-cluster sums, so each replicate is a matrix product
    instead of a rebuilt DataFrame: identical statistic, ~1000x faster.
    """
    clusters = sorted(df["variant"].unique())
    est = combine(_masked_means(df, masks, "y"))
    if est is None or math.isnan(est):
        return {"estimate": None, "lo": None, "hi": None, "n_clusters": len(clusters)}
    if len(clusters) < MIN_CLUSTERS:
        return {"estimate": float(est), "lo": None, "hi": None, "n_clusters": len(clusters),
                "note": f"CI requires ≥{MIN_CLUSTERS} variants"}
    y = df["y"].to_numpy()
    in_c = [(df["variant"] == c).to_numpy() for c in clusters]
    sums = np.array([[y[c & m].sum() for m in masks] for c in in_c])      # clusters × masks
    counts = np.array([[(c & m).sum() for m in masks] for c in in_c], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    w = np.stack([np.bincount(row, minlength=len(clusters)) for row in picks])  # times each cluster is drawn
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (w @ sums) / (w @ counts)                                      # n_boot × masks
    boots = combine(means.T)
    boots = boots[~np.isnan(boots)]
    lo, hi = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if len(boots) else (None, None)
    return {"estimate": float(est), "lo": lo, "hi": hi, "n_clusters": len(clusters),
            "significant": bool(lo is not None and (lo > 0 or hi < 0))}


def main_effects(df: pd.DataFrame, design: dict) -> list[dict]:
    out = []
    diff = lambda m: m[0] - m[1]  # noqa: E731
    for f in design["factors"]:
        levels = design["spec"]["vary"][f]
        ref = levels[0]
        for lv in levels[1:]:
            masks = [(df[f] == lv).to_numpy(), (df[f] == ref).to_numpy()]
            m = _cluster_boot(df, masks, diff)
            hyp = float(diff(_masked_means(df, masks, "y_ref")))
            out.append({"factor": f, "level": lv, "reference_level": ref, "model": m, "hypothesis": hyp,
                        "direction_agrees": None if m["estimate"] is None else bool(np.sign(m["estimate"]) == np.sign(hyp) or abs(hyp) < 0.02)})
    return out


def interactions(df: pd.DataFrame, design: dict) -> list[dict]:
    out = []
    facs = design["factors"]
    did = lambda m: (m[0] - m[1]) - (m[2] - m[3])  # noqa: E731  difference in differences
    for f1, f2 in itertools.combinations(facs, 2):
        l1, l2 = design["spec"]["vary"][f1], design["spec"]["vary"][f2]
        for a, b in itertools.product(l1[1:], l2[1:]):
            a0, b0 = l1[0], l2[0]
            masks = [((df[f1] == x) & (df[f2] == y)).to_numpy() for x, y in ((a, b), (a0, b), (a, b0), (a0, b0))]
            out.append({"factors": [f1, f2], "levels": [a, b], "baseline": [a0, b0],
                        "model": _cluster_boot(df, masks, did), "hypothesis": float(did(_masked_means(df, masks, "y_ref")))})
    return out


def ols(df: pd.DataFrame, design: dict) -> dict:
    import statsmodels.formula.api as smf
    facs = design["factors"]
    if df["y"].nunique() < 2:
        return {"error": "Outcome has no variation."}
    terms = [f"C({f}, Treatment(reference='{design['spec']['vary'][f][0]}'))" for f in facs]
    rhs = " + ".join(terms)
    if len(facs) >= 2:
        rhs = " + ".join(terms) + " + " + " + ".join(f"{a}:{b}" for a, b in itertools.combinations(terms, 2))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.ols(f"y ~ {rhs}", data=df)
            if df["variant"].nunique() >= MIN_CLUSTERS:
                fit = model.fit(cov_type="cluster", cov_kwds={"groups": df["variant"]})
                se_type = "cluster-robust (variants)"
            else:
                fit = model.fit(cov_type="HC1")
                se_type = "HC1"
        coefs = []
        ci = fit.conf_int()
        for name in fit.params.index:
            coefs.append({"term": name.replace("C(", "").replace(", Treatment(reference=", " vs ").replace("))", ")").replace(")[T.", " = ").replace("]", ""),
                          "coef": float(fit.params[name]), "se": float(fit.bse[name]),
                          "lo": float(ci.loc[name, 0]), "hi": float(ci.loc[name, 1]), "p": float(fit.pvalues[name])})
        return {"r2": float(fit.rsquared), "n": int(fit.nobs), "se_type": se_type, "coefficients": coefs}
    except Exception as e:  # singular designs etc.
        return {"error": str(e)}


def variance_decomposition(df: pd.DataFrame) -> dict:
    """Share of outcome variance explained by context (cell) vs wording/names (variant)."""
    y = df["y"]
    tot = float(((y - y.mean()) ** 2).sum())
    if tot <= 1e-12:
        return {"context": 0.0, "wording": 0.0, "residual": 0.0, "total_ss": 0.0}
    cell_m = df.groupby("cell")["y"].transform("mean")
    ss_cell = float(((cell_m - y.mean()) ** 2).sum())
    resid1 = y - cell_m
    var_m = resid1.groupby(df["variant"]).transform("mean")
    ss_var = float((var_m ** 2).sum())
    ss_res = float(((resid1 - var_m) ** 2).sum())
    return {"context": ss_cell / tot, "wording": ss_var / tot, "residual": ss_res / tot, "total_ss": tot,
            "note": "Two-way decomposition (cell, then variant). Variants change only paraphrases, names, sentence order and the utterance's surface form, so the hypothesis attributes 0% to wording by construction."}


def cell_table(df: pd.DataFrame, design: dict) -> list[dict]:
    out = []
    for c in design["cells"]:
        d = df[df["cell"] == c["cell"]]
        if not len(d):
            continue
        vals = d["y"].to_numpy()
        out.append({"cell": c["cell"], "assignment": c["assignment"], "n": int(len(d)),
                    "y_mean": float(vals.mean()), "y_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "y_ref": float(d["y_ref"].mean()), "entropy": float(d["h"].mean()), "entropy_ref": float(d["h_ref"].mean()),
                    "top_counts": d["top"].value_counts().to_dict(), "action_counts": d["action"].value_counts().to_dict()})
    return out


def sensitivity(cells: list[dict]) -> dict:
    """Across cells: slope of model outcome on hypothesised outcome (1 = matches the hypothesis' sensitivity)."""
    if len(cells) < 3:
        return {"slope": None, "corr": None}
    x = np.array([c["y_ref"] for c in cells])
    y = np.array([c["y_mean"] for c in cells])
    if x.std() < 1e-9:
        return {"slope": None, "corr": None}
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1]) if y.std() > 1e-9 else 0.0
    return {"slope": float(slope), "intercept": float(intercept), "corr": corr,
            "mean_abs_entropy_gap": float(np.mean([abs(c["entropy"] - c["entropy_ref"]) for c in cells])),
            "calibration_note": "Entropy gap compares the model's spread of belief with the hypothesis' in each cell (calibration across contexts)."}


def paired(df: pd.DataFrame, cell_a: str, cell_b: str) -> dict:
    a = df[df["cell"] == cell_a].groupby("variant")["y"].mean()
    b = df[df["cell"] == cell_b].groupby("variant")["y"].mean()
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return {"error": "no shared variants"}
    d = (b.loc[common] - a.loc[common]).to_numpy()
    res = {"n_pairs": len(common), "mean_diff": float(d.mean()), "diffs": d.tolist()}
    if len(common) >= MIN_CLUSTERS:
        rng = np.random.default_rng(0)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
        res.update({"lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))})
        from scipy.stats import wilcoxon
        try:
            res["wilcoxon_p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            res["wilcoxon_p"] = None
    return res


def human_divergence(design: dict, cells: list[dict]) -> dict:
    path = DATA_DIR / "eval" / "human_annotations.jsonl"
    if not path.exists():
        return {"available": False, "note": "No human annotations collected for these cells. Model–human divergence is not reported; LSSM values are authored hypotheses, not human data."}
    ann = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ann[(r["family_id"], r["cell"])] = r
    k = design["outcome_intent"]
    rows = []
    for c in cells:
        a = ann.get((design["spec"]["family_id"], c["cell"]))
        if a:
            rows.append({"cell": c["cell"], "human": a["distribution"].get(k), "n": a["n"], "model": c["y_mean"]})
    if not rows:
        return {"available": False, "note": "Annotation file present but no matching cells."}
    return {"available": True, "rows": rows, "mean_abs_diff": float(np.mean([abs(r["human"] - r["model"]) for r in rows]))}


def analyze(design: dict, results: list[dict]) -> dict:
    df = frame(design, results)
    if df.empty:
        return {"error": "no results"}
    cells = cell_table(df, design)
    return {
        "outcome_intent": design["outcome_intent"],
        "n_units": int(len(df)),
        "cells": cells,
        "main_effects": main_effects(df, design),
        "interactions": interactions(df, design),
        "ols": ols(df, design),
        "variance": variance_decomposition(df),
        "sensitivity": sensitivity(cells),
        "human": human_divergence(design, cells),
        "caveat": "Estimates are effects of randomized prompt interventions on model behaviour, not causal effects on human interpretation.",
    }
