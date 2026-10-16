"""Score every candidate in the training pool with the *base* model.

These per-item diagnostics (cross-entropy vs reference, failure flags, chosen
action) are what failure-targeted selection uses. Only the base model's
observable-input behaviour is used — no hidden state.

    backend/.venv/bin/python scripts/score_pool.py [--model Qwen/Qwen2.5-0.5B-Instruct]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pgym.models.local_hf import load  # noqa: E402
from pgym.selection.pool import POOL_DIR, read_jsonl  # noqa: E402
from pgym.training.local_eval import score_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--out", default=str(POOL_DIR / "base_scores.jsonl"))
args = ap.parse_args()

items = read_jsonl(POOL_DIR / "train_pool.jsonl")
tok, model = load(args.model, None, "float16")
t0 = time.time()
res = score_items(tok, model, items, batch_size=16)
with open(args.out, "w") as f:
    for r in res:
        f.write(json.dumps({"id": r["id"], "model": args.model, "probs": r["probs"], "action": r["action"],
                            "score": r["score"]}, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
print(f"scored {len(res)} pool items in {time.time() - t0:.0f}s -> {args.out}")
