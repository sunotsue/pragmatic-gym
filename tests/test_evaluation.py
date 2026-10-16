"""Metrics, failure taxonomy, causal estimators and selection invariants."""
import numpy as np
import pytest

from pgym.causal import analysis
from pgym.causal.design import DesignSpec, build
from pgym.evaluation.metrics import paired_bootstrap_diff, score_item, style_features
from pgym.scenarios.corpus import load_corpus
from pgym.social import lssm
from pgym.social.domains import DOMAINS
from pgym.social.generator import FAMILIES, render

REF = {"follow_through": 0.12, "undecided": 0.28, "soft_decline": 0.45, "keeping_options": 0.15}


def test_perfect_model_has_zero_kl_and_passes():
    s = score_item("invitation", REF, REF, "graceful_exit")
    assert s["kl"] < 1e-6 and s["tvd"] < 1e-9
    assert s["primary_failure"] == "pass"


def test_literalism_detected():
    ref = {"follow_through": 0.05, "undecided": 0.10, "soft_decline": 0.80, "keeping_options": 0.05}
    model = {"follow_through": 0.1, "undecided": 0.7, "soft_decline": 0.1, "keeping_options": 0.1}
    s = score_item("invitation", ref, model, "acknowledge_wait")
    assert "literalism" in s["failures"] and "misread" in s["failures"]


def test_overconfidence_on_ambiguous_items():
    ref = {"follow_through": 0.25, "undecided": 0.25, "soft_decline": 0.3, "keeping_options": 0.2}
    model = {"follow_through": 0.02, "undecided": 0.02, "soft_decline": 0.94, "keeping_options": 0.02}
    s = score_item("invitation", ref, model, "clarify_low_pressure")
    assert s["ambiguous"] and "overconfident" in s["failures"]
    assert s["top1_match"] is None  # not decisive -> top-1 not scored


def test_harmful_action_flagged():
    s = score_item("invitation", REF, REF, "pressure")
    assert s["action_harmful"] and s["primary_failure"] == "harmful_action"


def test_style_features_are_separate_from_scores():
    f = style_features("Sorry!! No pressure at all, totally fine, sorry")
    assert f["apologies"] == 2 and f["hedges"] >= 2


def test_reference_distributions_valid():
    for s in load_corpus():
        d = s.reference.distribution
        assert abs(sum(d.values()) - 1) < 1e-6 and set(d) == set(DOMAINS[s.domain].intent_ids)
    assert len([s for s in load_corpus() if not s.counterfactual_of]) >= 20


def test_corpus_counterfactuals_change_reference():
    corpus = {s.id: s for s in load_corpus()}
    for s in corpus.values():
        if s.counterfactual_of:
            base = corpus[s.counterfactual_of]
            tvd = 0.5 * sum(abs(s.reference.distribution[k] - base.reference.distribution[k]) for k in base.reference.distribution)
            assert tvd >= 0.08, s.id  # every authored counterfactual must move the reference (some deliberately subtly)


def test_variant_is_pure_surface():
    for fam in FAMILIES:
        refs = [render(fam.id, {}, v).reference.distribution for v in range(4)]
        assert all(r == refs[0] for r in refs)


def test_lssm_effect_directions():
    fam = next(f for f in FAMILIES if f.id == "inv_let_you_know")
    base = lssm.posterior("invitation", fam.base, {})
    rep = lssm.posterior("invitation", fam.base, {"prior_pattern": "repeated"})
    lit = lssm.posterior("invitation", fam.base, {"norms": "literal_norm"})
    assert rep["soft_decline"] > base["soft_decline"] and lit["undecided"] > base["undecided"]


def test_design_excludes_incoherent_cells():
    d = build(DesignSpec(family_id="inv_let_you_know", vary={"relationship": ["close_friend", "acquaintance"], "familiarity": ["stranger", "established"]}, n_variants=2))
    assignments = [c["assignment"] for c in d["cells"]]
    assert {"relationship": "close_friend", "familiarity": "stranger"} not in assignments
    assert d["excluded"]


def test_analysis_recovers_planted_effect():
    """Feed the estimator synthetic 'model' outputs with a known effect and check it is recovered."""
    spec = DesignSpec(family_id="inv_let_you_know", vary={"prior_pattern": ["none", "repeated"], "audience": ["private", "group_chat"]}, n_variants=6)
    d = build(spec)
    rng = np.random.default_rng(0)
    results = []
    for u in d["units"]:
        y = 0.2 + (0.3 if u["assignment"]["prior_pattern"] == "repeated" else 0) + rng.normal(0, 0.02)
        y = float(np.clip(y, 0.01, 0.97))
        rest = (1 - y) / 3
        results.append({"probs": {"soft_decline": y, "follow_through": rest, "undecided": rest, "keeping_options": rest}, "action": "clarify_low_pressure"})
    an = analysis.analyze(d, results)
    me = {(e["factor"], e["level"]): e for e in an["main_effects"]}
    pp = me[("prior_pattern", "repeated")]["model"]
    assert abs(pp["estimate"] - 0.3) < 0.03 and pp["lo"] < 0.3 < pp["hi"] and pp["significant"]
    aud = me[("audience", "group_chat")]["model"]
    assert abs(aud["estimate"]) < 0.03
    assert an["variance"]["context"] > 0.9
    assert abs(an["interactions"][0]["model"]["estimate"]) < 0.04


def test_paired_bootstrap_detects_shift():
    a = list(np.linspace(1, 2, 60))
    b = [x - 0.2 for x in a]
    r = paired_bootstrap_diff(a, b)
    assert r["mean"] == pytest.approx(-0.2) and r["p_value"] < 0.01


def test_selection_respects_budget_and_strategies_differ():
    from pgym.selection.strategies import select
    sels = {s: select(s, budget=60000, seed=0) for s in ("random", "diversity", "counterfactual_contrast", "isolated")}
    for s in sels.values():
        assert s["tokens_used"] <= 60000
    ids = {k: {r["item"]["id"] for r in v["rows"]} for k, v in sels.items()}
    assert ids["random"] != ids["diversity"] != ids["counterfactual_contrast"]
    assert ids["isolated"] and not sels["isolated"]["include_context"]
    # contrast arm selects whole clusters
    clusters = {}
    for r in sels["counterfactual_contrast"]["rows"]:
        clusters.setdefault(r["item"]["cluster"], []).append(r)
    assert all(len(v) >= 2 for v in clusters.values())


def test_rollout_targets_are_observable_posteriors():
    from pgym.selection.pool import POOL_DIR, read_jsonl
    rows = [r for r in read_jsonl(POOL_DIR / "train_pool.jsonl") if r["source"] == "rollout"][:30]
    assert rows
    for r in rows:
        # not one-hot: never trained on the realised hidden intent
        assert max(r["reference"].values()) < 0.999 or r["turn"] >= 1
        assert abs(sum(r["reference"].values()) - 1) < 1e-6
