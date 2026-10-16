"""Reward components are auditable and resistant to the obvious hacks."""
import math

import pytest

from pgym.environments.policies import audit, policy_bayes, rollout
from pgym.environments.specs import SPECS, make


def test_message_text_does_not_affect_reward():
    """Length, apologies and deference in the free-text message are never rewarded."""
    def ret(msg):
        env = make("soft_no")
        env.reset(seed=4)
        total = 0.0
        while not env.done:
            _, r, *_ = env.step({"action": "clarify_low_pressure", "message": msg})
            total += r
        return total
    short = ret("?")
    long = ret("I am so, so sorry to bother you, " * 30 + "no pressure at all, totally fine either way, sorry!!")
    assert math.isclose(short, long)


def test_belief_score_is_proper():
    """Expected log score is maximised by reporting the true distribution (not by over/under-confidence)."""
    env = make("soft_no")
    env.reset(seed=0)
    prior = dict(env.prior)
    K = len(prior)

    def expected(belief):
        return sum(prior[i] * (math.log(max(belief[i], 1e-3)) - math.log(1 / K)) for i in prior)

    truth = expected(prior)
    top = max(prior, key=prior.get)
    overconf = {i: (0.97 if i == top else 0.01) for i in prior}
    uniform = {i: 1 / K for i in prior}
    assert truth > expected(overconf)
    assert truth > expected(uniform)


@pytest.mark.parametrize("env_id", list(SPECS))
def test_no_constant_policy_beats_bayes(env_id):
    res = audit(env_id, n=200)
    assert res["checks"]["bayes_beats_best_constant"], res["checks"]
    assert res["checks"]["oracle_beats_bayes"]
    assert res["checks"]["bayes_beats_random"]


@pytest.mark.parametrize("env_id", list(SPECS))
def test_inference_pays_even_without_belief_reward(env_id):
    """With the belief-scoring term switched off, a planner that uses only the observable
    posterior must still beat every constant move — i.e. the *action* reward rewards social
    inference. (A greedy one-step policy does not always manage this; see RESEARCH.md §6.)"""
    res = audit(env_id, n=60, weights={"pragmatic_understanding": 0.0}, planner=True)
    assert res["checks"]["planner_beats_best_constant"], (env_id, res["checks"])


def test_deferential_policies_are_not_rewarded():
    res = audit("soft_no", n=200)
    pol = res["policies"]
    assert pol["always:over_apologize"]["return"]["mean"] < pol["bayes"]["return"]["mean"] - 0.5


def test_always_clarify_is_penalised_for_repetition():
    env = make("soft_no")
    env.reset(seed=1, options={"force_intent": "undecided"})
    comps = []
    while not env.done:
        env.step({"action": "clarify_low_pressure"})
        comps.append(env.log[-1]["components"]["clarification"])
    if len(comps) > 1:
        assert comps[0] >= 0 and all(c < 0 for c in comps[1:])


def test_pressure_penalised_more_when_intent_is_refusal():
    def boundary(intent):
        env = make("soft_no")
        env.reset(seed=0, options={"force_intent": intent})
        env.step({"action": "pressure"})
        return env.log[-1]["components"]["boundary_respect"]
    assert boundary("soft_decline") < boundary("follow_through")


def test_no_single_action_optimal_for_all_intents():
    """Social ambiguity is real in the spec: the best constant action depends on the hidden intent."""
    from pgym.social.domains import DOMAINS
    for d in ("invitation", "relationship", "emotional", "family", "group"):
        dom = DOMAINS[d]
        best = {i: max(dom.action_ids, key=lambda a: dom.utility[a][i]) for i in dom.intent_ids}
        assert len(set(best.values())) > 1, d


def test_weights_are_configurable_and_decomposed():
    r0 = rollout("soft_no", policy_bayes, seed=3)
    r1 = rollout("soft_no", policy_bayes, seed=3, weights={"task_completion": 0.0})
    assert math.isclose(r0["return"] - r1["return"], r0["components"]["task_completion"], abs_tol=1e-9)
