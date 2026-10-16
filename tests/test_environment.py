"""Environment transitions, Gymnasium API contract, determinism, Bayes filter."""
import math

import numpy as np
import pytest

from pgym.environments.specs import SPECS, make

ENV_IDS = list(SPECS)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_spec_tables_are_valid_distributions(env_id):
    SPECS[env_id].validate()  # raises on any row not summing to 1 or unknown response


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_step_api_shape(env_id):
    env = make(env_id)
    obs, info = env.reset(seed=3)
    assert isinstance(obs, dict) and isinstance(info, dict)
    assert obs["turn"] == 0 and not obs["done"]
    a = env.action_space.sample()
    out = env.step(int(a))
    assert len(out) == 5
    obs2, r, term, trunc, info2 = out
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    assert obs2["turn"] == 1


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_horizon_truncates(env_id):
    env = make(env_id)
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step({"action": env.domain.action_ids[1]})
        steps += 1
        done = term or trunc
    assert steps <= env.spec_.horizon
    with pytest.raises(RuntimeError):
        env.step({"action": env.domain.action_ids[0]})


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_seeded_determinism(env_id):
    def run(seed):
        env = make(env_id)
        env.reset(seed=seed)
        trace = [env._hidden["intent"]]
        for a in env.domain.action_ids[:3]:
            if env.done:
                break
            _, r, *_ = env.step({"action": a})
            trace += [env.log[-1]["response"], round(r, 6)]
        return trace
    assert run(11) == run(11)


def test_hidden_intent_follows_prior():
    env = make("soft_no")
    counts = {}
    for s in range(1500):
        env.reset(seed=s)
        counts[env._hidden["intent"]] = counts.get(env._hidden["intent"], 0) + 1
    prior = env.prior  # same factors every episode (default_factors), variant varies but prior does not
    for k, p in prior.items():
        assert abs(counts.get(k, 0) / 1500 - p) < 0.05


def test_bayes_filter_matches_manual_update():
    env = make("soft_no")
    env.reset(seed=5)
    prior = dict(env.posterior)
    env.step({"action": "clarify_low_pressure"})
    rid = env.log[-1]["response"]
    like = {i: env.spec_.table[i]["clarify_low_pressure"].get(rid, 0.0) for i in prior}
    z = sum(prior[i] * like[i] for i in prior)
    for i in prior:
        assert math.isclose(env.posterior[i], prior[i] * like[i] / z, rel_tol=1e-9)


def test_reveal_makes_posterior_certain_or_consistent():
    env = make("soft_no")
    for s in range(200):
        env.reset(seed=s, options={"force_intent": "soft_decline"})
        env.step({"action": "graceful_exit"})
        if env.log[-1]["response"] == "relieved_no":
            # relieved_no is only produced by declining/undecided/keeping intents -> follow_through ruled out
            assert env.posterior["follow_through"] == 0.0
            return
    pytest.fail("expected a relieved_no within 200 seeds")


def test_withdrawal_when_friction_exceeds_patience():
    env = make("soft_no")
    for s in range(100):
        env.reset(seed=s, options={"force_intent": "soft_decline"})
        while not env.done:
            env.step({"action": "pressure"})
        if env.outcome == "withdrew":
            assert env.log[-1]["friction_after"] >= env._hidden["patience"] or env.log[-1]["response"] == "withdraw"
            return
    pytest.fail("pressure on a declining friend should sometimes end in withdrawal")


def test_custom_factors_change_prior():
    a, b = make("soft_no"), make("soft_no")
    a.reset(seed=0, options={"factors": {"prior_pattern": "none", "shared_history": "established_trust"}})
    b.reset(seed=0, options={"factors": {"prior_pattern": "repeated", "shared_history": "repeated_disappointment"}})
    assert b.prior["soft_decline"] > a.prior["soft_decline"] + 0.3


def test_observation_space_numeric_part():
    env = make("soft_no")
    obs, _ = env.reset(seed=0)
    assert env.observation_space["turn"].contains(np.int64(obs["turn"]))
