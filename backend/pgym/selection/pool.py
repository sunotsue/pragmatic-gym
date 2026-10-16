"""Candidate pool and evaluation sets (split at the *family* level).

Splits
------
train pool          template items from TRAIN families only, organised as
                    contrast clusters (anchor + one-factor neighbours), plus
                    multi-turn states harvested from Soft No rollouts
eval_id             TRAIN families, but held-out *cells* (no factor assignment
                    that appears in the pool) and held-out variants
eval_ood_family     unseen utterance families in seen domains ("We'll see!", ...)
eval_ood_domain     an entire unseen domain (family subtext)
eval_handwritten    the hand-authored corpus (different author process, never trained on)
regression          small general-knowledge MCQ set (catastrophic-forgetting smoke test)

Every item stores the observable view used for training and the reference
used for targets; latent fields are kept in separate keys and the exporter
(`selection/export.py`) never emits them into model-visible text.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from pgym.config import DATA_DIR
from pgym.environments.policies import policy_bayes
from pgym.environments.specs import make
from pgym.social.factors import factors_for, violations
from pgym.social.generator import FAMILIES, cell_key, full_levels, render

POOL_DIR = DATA_DIR / "pools"
POOL_VERSION = "pool-v1"


def _random_levels(domain: str, rng: random.Random) -> dict[str, str]:
    for _ in range(200):
        lv = {}
        for f in factors_for(domain):
            if f.group == "placebo":
                continue  # design-time control only; never part of the training pool
            if f.group == "utterance":
                lv[f.id] = "explicit" if rng.random() < 0.08 else "indirect"  # rare manipulation-check items
            else:
                # bias towards default to keep stories readable: ~45% of factors stay at default
                lv[f.id] = f.default if rng.random() < 0.45 else rng.choice(f.level_ids)
        lv = full_levels(domain, lv)
        if not violations(lv, domain):
            return lv
    raise RuntimeError("could not sample a valid assignment")


def _neighbours(domain: str, lv: dict[str, str], rng: random.Random, k: int = 2) -> list[tuple[str, dict]]:
    facs = [f for f in factors_for(domain) if f.group not in ("utterance", "placebo")]
    rng.shuffle(facs)
    out = []
    for f in facs:
        alts = [x for x in f.level_ids if x != lv[f.id]]
        rng.shuffle(alts)
        for a in alts:
            nb = {**lv, f.id: a}
            if not violations(nb, domain):
                out.append((f.id, nb))
                break
        if len(out) >= k:
            break
    return out


def item_from_scenario(scen, *, split: str, cluster: str | None = None, contrast_factor: str | None = None,
                       source: str = "template", extra: dict | None = None) -> dict:
    from pgym.social.domains import get_domain
    dom = get_domain(scen.domain)
    ref_act = dom.reference_actions(scen.reference.distribution)
    return {
        "id": scen.id + (f"::{extra['turn_tag']}" if extra and extra.get("turn_tag") else ""),
        "scenario_id": scen.id,
        "family_id": scen.family_id,
        "domain": scen.domain,
        "capability": scen.capability,
        "split": split,
        "source": source,
        "cluster": cluster,
        "contrast_factor": contrast_factor,
        "factors": scen.factors,
        "cell": cell_key(scen.factors) if scen.factors else None,
        "variant": scen.variant,
        "observable": scen.observable().model_dump(),
        "reference": scen.reference.distribution,
        "reference_source": scen.reference.source,
        "target_action": ref_act["best"],
        "acceptable_actions": ref_act["acceptable"],
        "provenance": {"generator": POOL_VERSION, "family": scen.family_id, "reference_version": scen.reference.version},
        **(extra or {}),
    }


def build_train_pool(clusters_per_family: int = 70, seed: int = 0, rollout_states: int = 240) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for fam in FAMILIES:
        if fam.split != "train":
            continue
        for c in range(clusters_per_family):
            lv = _random_levels(fam.domain, rng)
            variant = rng.randrange(0, 6)
            cid = f"{fam.id}:c{c}"
            anchor = render(fam.id, lv, variant=variant, split="train")
            items.append(item_from_scenario(anchor, split="train", cluster=cid))
            for fid, nb in _neighbours(fam.domain, lv, rng):
                s = render(fam.id, nb, variant=variant, split="train")
                items.append(item_from_scenario(s, split="train", cluster=cid, contrast_factor=fid))
    items += harvest_rollout_states(rollout_states, seed=seed + 1)
    # de-duplicate by scenario id (identical renders)
    seen, out = set(), []
    for it in items:
        if it["id"] not in seen:
            seen.add(it["id"])
            out.append(it)
    return out


def harvest_rollout_states(n: int, seed: int = 1) -> list[dict]:
    """Multi-turn decision points from Soft No episodes (behaviour policy: random).

    The target at a mid-episode state is the *observable posterior* given the
    whole transcript (the env's exact Bayes filter) — never the sampled hidden
    intent — so the training signal is calibrated by construction. Reward
    components and outcome of the behaviour action are preserved for audit.
    """
    from pgym.scenarios.schema import Reference
    from pgym.social.domains import get_domain
    rng = random.Random(seed)
    dom = get_domain("invitation")
    out = []
    ep = 0
    while len(out) < n and ep < n * 5:
        env = make("soft_no")
        fam_lv = _random_levels("invitation", rng)
        fam_lv["directness"] = "indirect"
        obs, info = env.reset(seed=10_000 + ep, options={"factors": fam_lv})
        ep += 1
        steps = 0
        while not env.done and steps < 3:
            behaviour = rng.choice(dom.action_ids)
            obs, r, term, trunc, info = env.step({"action": behaviour})
            steps += 1
            if env.done:
                break
            # decision point after the interlocutor's response
            scen = env.scenario.model_copy(update={
                "id": f"{env.scenario.id}::ep{ep}t{steps}",
                "history": [m.model_copy() for m in env.transcript],
                "reference": Reference(distribution=dict(env.posterior), source="lssm",
                                       version=env.scenario.reference.version,
                                       rationale="Exact observable posterior from the environment's Bayes filter."),
                "split": "train",
            })
            bayes = policy_bayes(env, obs)
            out.append(item_from_scenario(scen, split="train", source="rollout", extra={
                "turn": steps, "turn_tag": f"t{steps}", "behaviour_action": behaviour,
                "behaviour_reward_components": info["last"]["components"], "behaviour_response": info["last"]["response"],
                "bayes_action": bayes["action"],
            }))
    return out[:n]


def build_eval_sets(seed: int = 100, n_per_family: int = 40, pool_cells: set[str] | None = None) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    pool_cells = pool_cells or set()
    sets: dict[str, list[dict]] = {"eval_id": [], "eval_ood_family": [], "eval_ood_domain": []}
    for fam in FAMILIES:
        split = "eval_id" if fam.split == "train" else fam.split
        n_clusters = 8 if split == "eval_id" else n_per_family // 3 + 1
        made = 0
        tries = 0
        while made < n_clusters and tries < 2000:
            tries += 1
            lv = _random_levels(fam.domain, rng)
            if split == "eval_id" and (fam.id + "|" + cell_key(lv)) in pool_cells:
                continue
            variant = rng.randrange(6, 9)  # variants never used in the pool
            cid = f"{fam.id}:e{made}"
            group = [(None, lv)] + _neighbours(fam.domain, lv, rng)
            if split == "eval_id" and any((fam.id + "|" + cell_key(g[1])) in pool_cells for g in group):
                continue
            for fid, g in group:
                s = render(fam.id, g, variant=variant, split=split)
                sets[split].append(item_from_scenario(s, split=split, cluster=cid, contrast_factor=fid))
            # paraphrase replicate of the anchor (robustness to wording)
            s2 = render(fam.id, lv, variant=(variant + 1 - 6) % 3 + 6, split=split)
            sets[split].append(item_from_scenario(s2, split=split, cluster=cid, contrast_factor="__paraphrase__"))
            made += 1
    from pgym.scenarios.corpus import load_corpus
    sets["eval_handwritten"] = [item_from_scenario(s, split="eval_handwritten", source="handwritten") for s in load_corpus()]
    return sets


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def build_all(seed: int = 0) -> dict:
    pool = build_train_pool(seed=seed)
    pool_cells = {it["family_id"] + "|" + it["cell"] for it in pool if it.get("cell")}
    evals = build_eval_sets(pool_cells=pool_cells)
    write_jsonl(POOL_DIR / "train_pool.jsonl", pool)
    for k, v in evals.items():
        write_jsonl(POOL_DIR / f"{k}.jsonl", v)
    return {"train_pool": len(pool), **{k: len(v) for k, v in evals.items()}}
