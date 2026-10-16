"""Reference policies and the reward audit.

The audit answers the question a skeptical reviewer should ask of any social
reward: *what does it actually incentivize?* We roll out a fixed battery of
policies on matched seeds:

* constant policies (always pressure, always clarify, always release, ...)
* random
* bayes        — acts on the observable posterior (no hidden information)
* oracle       — knows the hidden intent (an upper bound, not a fair agent)
* model        — any inference adapter acting through the observable view

If a constant or stylistic policy matches `bayes`, the reward is hackable.
"""
from __future__ import annotations

import numpy as np

from pgym.environments.specs import SPECS, make
from pgym.evaluation.metrics import bootstrap_ci


def policy_constant(aid: str):
    def f(env, obs):
        return {"action": aid}
    f.__name__ = f"always_{aid}"
    return f


def policy_random(env, obs):
    return {"action": env.rng.choice(env.domain.action_ids)}


def policy_bayes(env, obs):
    """Greedy expected utility under the observable posterior, with a belief report."""
    post = env.posterior
    eu = env.domain.expected_utility(post)
    roles = env.spec_.roles
    # Don't re-clarify once uncertainty has been reduced or after one attempt.
    for a in list(eu):
        if a in roles.get("clarify", set()) and env.clarify_count >= 1:
            eu[a] -= 0.3
    return {"action": max(eu, key=eu.get), "belief": dict(post)}


def make_planner(n_sims: int = 12, seed: int = 0):
    """POMDP planner: Monte-Carlo lookahead on the known interlocutor model.

    For each action, sample hidden intents from the *observable posterior*
    (never the true hidden state), simulate the rest of the episode with the
    greedy Bayes policy and pick the action with the highest mean return
    (belief-scoring term excluded, so the choice reflects action quality only).
    """
    import copy
    import random as _r

    def f(env, obs):
        rng = _r.Random(seed * 7919 + env.turn * 31 + len(env.log))
        best, best_v = None, -1e9
        ids = env.domain.intent_ids
        for a in env.domain.action_ids:
            tot = 0.0
            for k in range(n_sims):
                sim = copy.deepcopy(env)
                sim.rng = _r.Random(rng.random())
                sim._hidden["intent"] = rng.choices(ids, weights=[env.posterior[i] for i in ids])[0]
                sim.weights = {**sim.weights, "pragmatic_understanding": 0.0}
                _, r, term, trunc, _ = sim.step({"action": a})
                ret = r
                while not sim.done:
                    _, r, *_ = sim.step({"action": policy_bayes(sim, None)["action"]})
                    ret += r
                tot += ret
            v = tot / n_sims
            if v > best_v:
                best, best_v = a, v
        return {"action": best, "belief": dict(env.posterior)}
    f.__name__ = "bayes_planner"
    return f


def policy_oracle(env, obs):
    intent = env._hidden["intent"]  # privileged — upper bound only
    u = {a: env.domain.utility[a][intent] for a in env.domain.action_ids}
    roles = env.spec_.roles
    for a in list(u):
        if a in roles.get("clarify", set()) and env.clarify_count >= 1:
            u[a] -= 0.3
    belief = {i: (0.97 if i == intent else 0.03 / (len(u) and len(env.domain.intent_ids) - 1)) for i in env.domain.intent_ids}
    return {"action": max(u, key=u.get), "belief": belief}


def adapter_policy(adapter):
    """Wrap any InterpretationOutput-producing adapter as an agent policy."""
    def f(env, obs):
        out = adapter.interpret(env.observable())
        return {"action": out.action, "belief": out.interpretation_probs, "message": out.reply,
                "_provenance": out.provenance.__dict__}
    f.__name__ = f"model:{getattr(adapter, 'tag', adapter.name)}"
    return f


def rollout(env_id: str, policy, seed: int, options: dict | None = None, weights: dict | None = None) -> dict:
    env = make(env_id, weights)
    obs, info = env.reset(seed=seed, options=options)
    total, steps = 0.0, 0
    comps: dict[str, float] = {}
    while True:
        act = policy(env, obs)
        act = {k: v for k, v in act.items() if not k.startswith("_")}
        obs, r, term, trunc, info = env.step(act)
        total += r
        steps += 1
        for k, v in info["last"]["weighted"].items():
            comps[k] = comps.get(k, 0.0) + v
        if term or trunc:
            break
    spec = SPECS[env_id]
    return {"return": total, "steps": steps, "outcome": env.outcome,
            "success": spec.outcomes[env.outcome].success, "intent": env._hidden["intent"], "components": comps}


BATTERY = ["oracle", "bayes", "random"]


def audit(env_id: str, n: int = 300, seed0: int = 0, options: dict | None = None, weights: dict | None = None,
          planner: bool = False) -> dict:
    spec = SPECS[env_id]
    from pgym.social.domains import get_domain
    dom = get_domain(spec.domain)
    policies = {"oracle": policy_oracle, "bayes": policy_bayes, "random": policy_random}
    if planner:
        policies["planner"] = make_planner()
    for a in dom.action_ids:
        policies[f"always:{a}"] = policy_constant(a)
    rows = {}
    for name, pol in policies.items():
        res = [rollout(env_id, pol, seed0 + s, options, weights) for s in range(n)]
        comps = {}
        for r in res:
            for k, v in r["components"].items():
                comps.setdefault(k, []).append(v)
        outcomes = {}
        for r in res:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        by_intent = {}
        for r in res:
            by_intent.setdefault(r["intent"], []).append(r["return"])
        rows[name] = {
            "return": bootstrap_ci([r["return"] for r in res], n_boot=1000),
            "success_rate": float(np.mean([r["success"] for r in res])),
            "mean_steps": float(np.mean([r["steps"] for r in res])),
            "components": {k: float(np.mean(v)) for k, v in comps.items()},
            "outcomes": outcomes,
            "return_by_intent": {k: float(np.mean(v)) for k, v in by_intent.items()},
        }
    ranking = sorted(rows, key=lambda k: -rows[k]["return"]["mean"])
    best_constant = max((k for k in rows if k.startswith("always:")), key=lambda k: rows[k]["return"]["mean"])
    best_constant_return = rows[best_constant]["return"]["mean"]
    return {"env_id": env_id, "n_episodes": n, "seeds": [seed0, seed0 + n - 1], "policies": rows, "ranking": ranking,
            "checks": {
                "oracle_beats_bayes": rows["oracle"]["return"]["mean"] > rows["bayes"]["return"]["mean"],
                "bayes_beats_best_constant": rows["bayes"]["return"]["mean"] > rows[best_constant]["return"]["mean"],
                "bayes_beats_random": rows["bayes"]["return"]["mean"] > rows["random"]["return"]["mean"],
                **({"planner_beats_best_constant": rows["planner"]["return"]["mean"] > rows[best_constant]["return"]["mean"],
                    "planner_margin": rows["planner"]["return"]["mean"] - rows[best_constant]["return"]["mean"]} if planner else {}),
                "best_constant": best_constant,
            }}
