"""Run (or resume) the post-training experiment.

    backend/.venv/bin/python scripts/run_experiment.py --exp soft-social-v1
    backend/.venv/bin/python scripts/run_experiment.py --exp quick --arms base random counterfactual_contrast --seeds 0

Completed runs are cached under experiments/runs/<exp>/, so re-running resumes.
Results are aggregated after each seed into experiments/results/<exp>.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pgym.training.experiment import run  # noqa: E402
from pgym.training.sft import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="soft-social-v1")
ap.add_argument("--arms", nargs="*")
ap.add_argument("--seeds", nargs="*", type=int)
ap.add_argument("--budget", type=int)
ap.add_argument("--target-failure", default=None)
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
run(a.exp, cfg, arms=a.arms, seeds=a.seeds, budget=a.budget, target_failure=a.target_failure,
    log=lambda m: print(m, flush=True))
