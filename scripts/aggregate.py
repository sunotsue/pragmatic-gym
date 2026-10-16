"""Re-aggregate an experiment from cached per-run predictions (no retraining).

    backend/.venv/bin/python scripts/aggregate.py --exp social-sft-v1
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pgym.training.experiment import RUNS, aggregate, eval_items  # noqa: E402
from pgym.training.sft import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="social-sft-v1")
a = ap.parse_args()
cfg = load_config()
root = RUNS / a.exp
arms: dict[str, list] = {}
seeds = set()
for d in sorted(root.iterdir()):
    ev = d / "eval.json"
    if not ev.exists():
        continue
    r = json.loads(ev.read_text())
    tm = d / "adapter" / "train_meta.json"
    r["train_meta"] = json.loads(tm.read_text()) if tm.exists() else None
    m = re.match(r"(.+)_s(\d+)$", d.name)
    arm = m.group(1) if m else d.name
    if m:
        seeds.add(int(m.group(2)))
    arms.setdefault(arm, []).append(r)
# only keep seeds for which every trained arm finished (balanced comparison)
trained = [k for k in arms if k != "base"]
complete = sorted(s for s in seeds if all((root / f"{k}_s{s}" / "eval.json").exists() for k in trained))
for k in trained:
    arms[k] = [json.loads((root / f"{k}_s{s}" / "eval.json").read_text()) | {"train_meta": json.loads((root / f"{k}_s{s}" / "adapter" / "train_meta.json").read_text())} for s in complete]
meta = {"base_model": cfg["base_model"], "budget": cfg["selection"]["token_budget"], "seeds_completed": complete, "arms": list(arms),
        "lora": cfg["lora"], "train": cfg["train"], "target_failure": None, "eval_sizes": {s: len(v) for s, v in eval_items().items()}}
res = aggregate(a.exp, arms, meta)
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in t.items() if kk in ("diff_treat_minus_control", "ci", "p_value", "p_holm")} for k, t in res["tests"].items()}, indent=1))
