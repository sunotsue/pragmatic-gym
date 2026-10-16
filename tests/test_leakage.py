"""Hidden state must never reach an agent, a prompt or a training example."""
import json
import re

import pytest

from pgym.environments.specs import SPECS, make
from pgym.models.prompts import build_interpretation_prompt
from pgym.scenarios.corpus import load_corpus
from pgym.selection.pool import POOL_DIR
from pgym.social.factors import FACTORS
from pgym.social.generator import FAMILIES, render
from pgym.training.formats import chat_messages, cyclic

LATENT_KEYS = ["true_intent", "private_state", "lssm_terms", "patience", "force_intent"]
LEVEL_IDS = sorted({lv.id for f in FACTORS for lv in f.levels if "_" in lv.id} | {f.id for f in FACTORS if "_" in f.id})


def assert_clean(text: str, scen=None, ids_public: bool = False):
    for k in LATENT_KEYS + LEVEL_IDS:
        assert not re.search(rf"\b{re.escape(k)}\b", text), f"latent identifier '{k}' leaked"
    if scen is not None:
        for v in scen.hidden.values():
            if isinstance(v, str) and len(v) > 20 and v not in {i.id for i in scen.interpretations}:
                assert v not in text, "hidden private state text leaked into prompt"
        assert scen.reference.rationale not in text
        for i in ([] if ids_public else scen.interpretations):  # JSON prompts name option ids as schema keys (public)
            if "_" in i.id:
                assert not re.search(rf"\b{i.id}\b", text)


def test_corpus_prompts_exclude_latent_state():
    for s in load_corpus():
        assert_clean(build_interpretation_prompt(s.observable()), s, ids_public=True)
        for kind in ("interpret", "act"):
            n = len(s.interpretations if kind == "interpret" else s.actions)
            msgs = chat_messages(s.observable(), kind, cyclic(n, 1))
            assert_clean(json.dumps(msgs), s)


def test_observable_has_no_latent_fields():
    s = load_corpus()[0]
    d = s.observable().model_dump()
    for k in ("hidden", "reference", "factors", "why_hidden_matters"):
        assert k not in d


@pytest.mark.parametrize("fam", [f.id for f in FAMILIES])
def test_generated_prompts_exclude_factor_ids(fam):
    s = render(fam, {}, variant=1)
    assert_clean(build_interpretation_prompt(s.observable()), s, ids_public=True)
    assert_clean(json.dumps(chat_messages(s.observable(), "interpret", cyclic(4, 0))), s)


@pytest.mark.parametrize("env_id", list(SPECS))
def test_env_observation_never_contains_hidden_intent(env_id):
    env = make(env_id)
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        blob = json.dumps(obs) + json.dumps(info)
        assert "patience" not in blob and "intent" not in json.dumps({k: v for k, v in obs.items() if k != "interpretations"})
        while not env.done:
            obs, r, term, trunc, info = env.step({"action": env.domain.action_ids[seed % len(env.domain.action_ids)]})
            visible = json.dumps({k: v for k, v in obs.items() if k != "interpretations"})
            assert env._hidden["intent"] not in visible
        # the adapter-facing observable view also excludes it
        assert env._hidden["intent"] not in json.dumps(env.observable().model_dump(exclude={"interpretations"}))


def test_exported_training_data_has_no_hidden_state(tmp_path):
    from pgym.selection.strategies import diagnostics, export, select
    sel = select("random", budget=20000, seed=0)
    assert diagnostics(sel)["leakage"]["n_leaks"] == 0
    export(sel, tmp_path)
    for line in (tmp_path / "sft.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert_clean(json.dumps(row["prompt"]))


def test_family_level_split():
    from pgym.selection.pool import read_jsonl
    pool = read_jsonl(POOL_DIR / "train_pool.jsonl")
    train_fams = {r["family_id"] for r in pool}
    for split in ("eval_ood_family", "eval_ood_domain", "eval_handwritten"):
        ev = read_jsonl(POOL_DIR / f"{split}.jsonl")
        assert not (train_fams & {e["family_id"] for e in ev}), split
    ev_id = read_jsonl(POOL_DIR / "eval_id.jsonl")
    pool_cells = {(r["family_id"], r["cell"]) for r in pool if r.get("cell")}
    assert not pool_cells & {(e["family_id"], e["cell"]) for e in ev_id if e.get("cell")}
    assert all(r["split"] == "train" for r in pool)
