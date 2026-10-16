"""Render the results section of RESEARCH.md from experiments/results/<exp>.json.

Numbers in the write-up are generated, never typed by hand.

    backend/.venv/bin/python scripts/write_results_md.py --exp social-sft-v1
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = {"base": "Base", "random": "Random", "diversity": "Diversity", "failure_targeted": "Failure-targeted",
       "counterfactual_contrast": "Contrast", "isolated": "Isolated"}
SPLIT = {"eval_id": "ID (held-out cells)", "eval_ood_family": "Unseen families", "eval_ood_domain": "Unseen domain", "eval_handwritten": "Hand-written"}


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="social-sft-v1")
    a = ap.parse_args()
    r = json.loads((ROOT / "experiments" / "results" / f"{a.exp}.json").read_text())
    S, arms, splits = r["summary"], list(r["summary"]), list(next(iter(r["summary"].values()))["splits"])
    out = [f"**Experiment `{r['experiment_id']}`** — {r['label']}. Seeds completed: {', '.join(map(str, r['meta']['seeds_completed']))}. "
           f"Budget {r['meta']['budget'] // 1000}k tokens per arm. Eval items: " + ", ".join(f"{SPLIT[k]} {v}" for k, v in r["meta"]["eval_sizes"].items()) + ".", ""]
    out += ["**Pre-registered tests** (Δ cross-entropy, treatment − control, pooled generalisation sets; negative = treatment better):", "",
            "| Hypothesis | Δ (nats) | 95% CI | p | p (Holm) |", "|---|---|---|---|---|"]
    for k in ("H1_contrast_vs_random", "H2_context_vs_isolated", "H3_failure_vs_random"):
        t = r["tests"].get(k)
        if t:
            out.append(f"| {k.split('_', 1)[0]} {ARM[t['treatment']]} vs {ARM[t['control']]} | {t['diff_treat_minus_control']:+.3f} | [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | {t['p_value']:.3f} | {t.get('p_holm', float('nan')):.3f} |")
    vb = [(k, t) for k, t in r["tests"].items() if k.startswith("vs_base")]
    out += ["", "vs base: " + "; ".join(f"{ARM[t['treatment']]} {t['diff_treat_minus_control']:+.3f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]" for _, t in vb), ""]
    for metric, title, d in (("kl", "Excess loss KL(reference‖model), nats — lower is better", 3),
                             ("top1_decisive", "Top-1 agreement on decisive items — higher is better", 2),
                             ("context_sensitivity_r", "Context sensitivity r (Δmodel vs Δreference under one-factor changes) — higher is better", 2),
                             ("sensitivity_ratio", "Sensitivity ratio |Δmodel| / |Δreference| — 1 = moves as much as the reference", 2),
                             ("action_regret", "Action regret (expected utility) — lower is better", 3),
                             ("overconfident_on_ambiguous", "Overconfident on ambiguous items — lower is better", 2)):
        out += [f"**{title}** (mean over seeds; seed values in the app)", "", "| Arm | " + " | ".join(SPLIT[s] for s in splits) + " |", "|---" * (len(splits) + 1) + "|"]
        for arm in arms:
            out.append(f"| {ARM.get(arm, arm)} | " + " | ".join(f(S[arm]["splits"][s][metric]["mean"], d) for s in splits) + " |")
        if metric in ("kl", "top1_decisive") and r.get("reference_baselines"):
            for b, name in (("uniform", "*Uniform (no model)*"), ("utterance_prior", "*Utterance-only prior*")):
                out.append(f"| {name} | " + " | ".join(f(r["reference_baselines"][s][b][metric], d) for s in splits) + " |")
        out.append("")
    out += ["**Regression tripwire** (40 general MCQ): " + ", ".join(f"{ARM.get(a_, a_)} {S[a_]['regression_accuracy']:.3f}" for a_ in arms), ""]
    md = (ROOT / "RESEARCH.md").read_text()
    block = "<!-- RESULTS -->\n" + "\n".join(out) + "\n<!-- /RESULTS -->"
    if "<!-- /RESULTS -->" in md:
        md = re.sub(r"<!-- RESULTS -->.*?<!-- /RESULTS -->", lambda _: block, md, flags=re.S)
    else:
        md = md.replace("<!-- RESULTS -->", block)
    (ROOT / "RESEARCH.md").write_text(md)
    print("\n".join(out))


if __name__ == "__main__":
    main()
