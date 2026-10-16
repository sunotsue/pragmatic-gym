"""Record every local-model output the app can ask for in a demo, so it never runs a model live.

    backend/.venv/bin/python scripts/record_demo.py [--only explorer,diagnosis,cf,playground]

Runs Qwen2.5-0.5B-Instruct once and saves each result to data/recorded/local_qwen.json,
keyed by a hash of the exact observable input (see pgym.models.recorded). Resumable:
inputs already recorded are skipped.

Coverage
  explorer    every corpus scenario (the same scoring path as live Explorer inference)
  diagnosis   the corpus-wide diagnosis on the Explorer tab
  cf          Counterfactual Lab: every preset, and every single variable compared on its own
              for each preset's utterance family, with 6 surface variants (covers 2/3/4/6)
  playground  model-played episodes: seeds 0-20 at default settings for every environment,
              plus seeds 0-5 for each level of the opening selectors shown in the UI
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PGYM_LIVE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pgym import config  # noqa: E402
from pgym.models import recorded  # noqa: E402

# Mirrors PRESETS in apps/web/src/app/counterfactual/page.tsx
PRESETS = [
    ("inv_let_you_know", {"shared_history": ["unspecified", "repeated_disappointment", "established_trust"], "audience": ["private", "group_chat"]}),
    ("inv_let_you_know", {"prior_pattern": ["none", "repeated"], "placebo": ["none", "neutral"]}),
    ("inv_let_you_know", {"relationship": ["close_friend", "romantic_interest", "acquaintance"], "prior_pattern": ["none", "repeated"]}),
    ("inv_think_about_it", {"norms": ["unspecified", "indirect_norm", "literal_norm"], "comm_style": ["unspecified", "direct", "avoidant"]}),
    ("rel_like_what_we_have", {"commitment": ["undefined", "explicit"], "prior_pattern": ["none", "repeated"]}),
    ("inv_well_see", {"directness": ["indirect", "explicit"], "shared_history": ["unspecified", "established_trust"]}),
]
N_VARIANTS = 6
# Mirrors the opening selectors in apps/web/src/app/playground/page.tsx
PLAYGROUND_FACTOR_IDS = ["prior_pattern", "shared_history", "relationship", "audience", "comm_style", "stakes", "norms", "commitment"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def explorer():
    from pgym.models.local_hf import LocalHFAdapter
    from pgym.scenarios.corpus import load_corpus
    a = LocalHFAdapter()
    todo = [s for s in load_corpus() if not recorded.has("interpret", s.observable())]
    for i, s in enumerate(todo):
        out = a.interpret(s.observable())
        recorded.put("interpret", s.observable(), {"output": out.to_dict()})
        print(f"explorer {i + 1}/{len(todo)} {s.id}", flush=True)
        if i % 10 == 9:
            recorded.save()
    recorded.save()


def diagnosis():
    from pgym.api import main
    out = main.diagnosis(backend="local_hf", model=None, x_researcher_token=config.RESEARCHER_TOKEN)
    out["provenance"] = recorded.provenance(now())
    recorded.put("diagnosis", "corpus", out)
    recorded.save()
    print("diagnosis recorded", flush=True)


def cf():
    from pgym.causal.design import DesignSpec, build
    from pgym.models.local_hf import DEFAULT_LOCAL_MODEL, load
    from pgym.selection.pool import item_from_scenario
    from pgym.social.factors import FACTORS
    from pgym.social.generator import FAMILY_BY_ID, render
    from pgym.training.local_eval import score_items
    specs = [DesignSpec(family_id=f, vary=v, n_variants=N_VARIANTS) for f, v in PRESETS]
    for fam_id in sorted({f for f, _ in PRESETS}):
        dom = FAMILY_BY_ID[fam_id].domain
        specs += [DesignSpec(family_id=fam_id, vary={f.id: f.level_ids}, n_variants=N_VARIANTS) for f in FACTORS if dom in f.domains]
    scens = {}
    for spec in specs:
        d = build(spec)
        levels = {c["cell"]: c["levels"] for c in d["cells"]}
        for u in d["units"]:
            s = render(spec.family_id, levels[u["cell"]], u["variant"])
            if not recorded.has("cf", s.observable()):
                scens[recorded.key("cf", s.observable())] = s
    todo = list(scens.values())
    print(f"cf: {len(todo)} prompts to score", flush=True)
    if not todo:
        return
    tok, m = load(DEFAULT_LOCAL_MODEL, None, "float16")
    for i in range(0, len(todo), 64):
        chunk = todo[i:i + 64]
        res = score_items(tok, m, [item_from_scenario(s, split="cf") for s in chunk])
        t = now()
        for s, r in zip(chunk, res):
            recorded.put("cf", s.observable(), {"probs": r["probs"], "action": r["action"], "recorded_at": t})
        recorded.save()
        print(f"cf {min(i + 64, len(todo))}/{len(todo)}", flush=True)


def playground():
    """Model-played episodes, stepped in lockstep so each turn's observations are scored in one batch."""
    from pgym.environments.specs import SPECS, make
    from pgym.models.local_hf import DEFAULT_LOCAL_MODEL, load
    from pgym.models.prompts import PROMPT_VERSION
    from pgym.social.factors import FACTORS
    from pgym.training.formats import FORMAT_VERSION
    from pgym.training.local_eval import score_items
    runs = []
    for env_id, spec in SPECS.items():
        runs += [(env_id, seed, {}) for seed in range(21)]
        if spec.opening[0] == "family":
            dom = spec.domain
            shown = [f for f in FACTORS if dom in f.domains and f.group != "utterance" and f.id in PLAYGROUND_FACTOR_IDS][:4]
            for f in shown:
                for lv in f.level_ids:
                    runs += [(env_id, seed, {f.id: lv}) for seed in range(6)]
    envs = []
    for env_id, seed, factors in runs:
        env = make(env_id)
        try:
            env.reset(seed=seed, options={"factors": factors} if factors else {})
        except ValueError:  # incoherent opening (the UI rejects it too)
            continue
        envs.append(env)
    tok, m = load(DEFAULT_LOCAL_MODEL, None, "float16")
    t0, turn = time.time(), 0
    while any(not e.done for e in envs):
        active = [e for e in envs if not e.done]
        todo = {}
        for e in active:
            obs = e.observable()
            if not recorded.has("interpret", obs):
                todo[recorded.key("interpret", obs)] = (obs, e.scenario.domain)
        items = [{"id": k, "observable": obs.model_dump(), "domain": dom, "reference": {o.id: 1 / len(obs.interpretations) for o in obs.interpretations}}
                 for k, (obs, dom) in todo.items()]
        for i in range(0, len(items), 64):
            res = score_items(tok, m, items[i:i + 64], action_perms=7)
            t = now()
            for it, r in zip(items[i:i + 64], res):
                obs = todo[it["id"]][0]
                action = next(a for a in obs.actions if a.id == r["action"])
                prov = {"backend": "local_hf", "model": DEFAULT_LOCAL_MODEL.split("/")[-1], "is_real_inference": True,
                        "label": f"Local inference · {DEFAULT_LOCAL_MODEL.split('/')[-1]}", "prompt_version": f"{PROMPT_VERSION}/{FORMAT_VERSION}",
                        "prompt_hash": "", "timestamp": t, "latency_ms": None, "recorded_at": None,
                        "settings": {"model_id": DEFAULT_LOCAL_MODEL, "scoring": "next-token letter distribution, averaged over cyclic option orders",
                                     "action_probs": r["action_probs"], "reply_source": "action template (the model chose the move; wording is the move's fixed template)"},
                        "usage": {}}
                recorded.put("interpret", obs, {"output": {
                    "interpretation_probs": r["probs"], "most_likely": max(r["probs"], key=r["probs"].get),
                    "rationale": "Distribution read from the model's next-token probabilities over option letters (no verbalized reasoning).",
                    "action": r["action"], "reply": action.example, "provenance": prov, "raw": ""}})
        recorded.save()
        for e in active:
            out = recorded.get("interpret", e.observable())["output"]
            e.step({"action": out["action"], "belief": out["interpretation_probs"], "message": out["reply"]})
        turn += 1
        print(f"playground turn {turn}: scored {len(items)} states, {sum(not e.done for e in envs)}/{len(envs)} episodes still running · {time.time() - t0:.0f}s", flush=True)


def audit():
    """Reward audits are pure simulation (no model), but slow (~1 min each), so they are saved too."""
    import json
    from pgym.environments.policies import audit as run_audit
    from pgym.environments.specs import SPECS
    path = config.DATA_DIR / "recorded" / "audits.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for env_id in SPECS:
        key = f"{env_id}:300:True"  # what the Playground requests
        if key not in out:
            out[key] = json.loads(json.dumps(run_audit(env_id, n=300, planner=True), default=float))
            path.write_text(json.dumps(out, separators=(",", ":")))
            print(f"audit {env_id} saved", flush=True)


def datalab():
    """Every Data Lab selection, comparison and export the UI can request, saved (deterministic, no model)."""
    import gzip
    import shutil
    from pgym.api import main
    budgets = [60000, 120000, 240000]  # Data Lab budget options
    targets = [None] + [k for k in main.data_pool_diagnosis()["any"] if k != "pass"]
    store = recorded.data_lab()
    reqs = [main.SelectRequest(strategy=s, budget=b, seed=0) for s in ("random", "diversity", "counterfactual_contrast", "isolated") for b in budgets]
    reqs += [main.SelectRequest(strategy="failure_targeted", budget=b, seed=0, target_failure=t) for b in budgets for t in targets]
    for i, req in enumerate(reqs):
        args = (req.strategy, req.budget, req.seed, req.target_failure)
        for kind, fn in (("select", main.data_select), ("compare", main.data_compare), ("export", main.data_export)):
            key = recorded.data_key(kind, *args)
            if key in store:
                continue
            try:
                out = fn(req)
            except Exception as e:  # e.g. too few pool items with this failure; the UI shows "not recorded"
                print(f"datalab skip {key}: {e}", flush=True)
                continue
            store[key] = out
            if kind == "export":
                recorded.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
                for f in ("sft.jsonl", "preference.jsonl"):
                    (recorded.EXPORTS_DIR / f"{key.replace(':', '_')}.{f}.gz").write_bytes(gzip.compress((Path(out["path"]) / f).read_bytes()))
                shutil.rmtree(out["path"], ignore_errors=True)
        recorded.save_data_lab()
        print(f"datalab {i + 1}/{len(reqs)} {req.strategy} {req.budget} {req.target_failure or 'any'}", flush=True)


STEPS = {"explorer": explorer, "diagnosis": diagnosis, "cf": cf, "playground": playground, "audit": audit, "datalab": datalab}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STEPS))
    for name in ap.parse_args().only.split(","):
        STEPS[name]()
    print(f"done · {len(recorded.store())} recorded outputs in {recorded.STORE_PATH}")
