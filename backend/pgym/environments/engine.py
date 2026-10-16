"""A table-driven, Gymnasium-style engine for multi-turn social interaction.

    env = make("soft_no")
    obs, info = env.reset(seed=0, options={"factors": {"prior_pattern": "repeated"}})
    obs, reward, terminated, truncated, info = env.step({"action": "clarify_low_pressure",
                                                          "belief": {...}, "message": "..."})

Five layers are kept strictly apart:

1. Observable dialogue and context  -> `obs` (what the agent sees)
2. Hidden social state              -> `self._hidden` (never in `obs`)
3. The agent's inferred beliefs     -> the optional `belief` the agent submits
4. The agent's selected action      -> `action` (discrete) + free-text `message`
5. Simulated interaction outcomes   -> interlocutor responses, friction, outcome

The interlocutor is a stochastic policy P(response | hidden intent, action).
Because that policy is known, the environment can also compute the exact
Bayesian posterior over intents given only observable events. That
*observable posterior* is what clarification/assumption rewards are
conditioned on — the agent is never penalised for failing to know something
it could not have known.

The free-text `message` is stored for display only. No reward component reads
it, so verbosity and deference cannot be rewarded (see tests/test_rewards.py).
"""
from __future__ import annotations

import copy
import math
import random
import uuid
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pgym.scenarios.corpus import get_scenario
from pgym.scenarios.schema import Message, Observable
from pgym.social.domains import get_domain
from pgym.social.generator import render
from pgym.social.lssm import normalized_entropy


@dataclass
class ResponseType:
    id: str
    texts: list  # list[str] (speaker says) or list[list[tuple[member_slot, str]]] for groups
    friction: float = 0.0
    reveals: str | None = None   # intent that becomes common ground (observable)
    outcome: str | None = None   # terminal outcome id
    note: str = ""


@dataclass
class Outcome:
    id: str
    label: str
    value: dict[str, float]  # task-completion value by true intent
    success: bool = False


@dataclass
class EnvSpec:
    id: str
    title: str
    subtitle: str
    domain: str
    opening: tuple[str, str]           # ("family", family_id) | ("corpus", scenario_id)
    default_factors: dict[str, str]
    capabilities: list[str]
    responses: dict[str, ResponseType]
    table: dict[str, dict[str, dict[str, float]]]  # intent -> action -> response -> p
    outcomes: dict[str, Outcome]
    roles: dict[str, set[str]]         # clarify / assume / boundary / repair / exit
    withdraw_text: list[str]
    horizon: int = 4
    flagship: bool = False
    status: str = "full"

    def validate(self) -> None:
        dom = get_domain(self.domain)
        for i in dom.intent_ids:
            for a in dom.action_ids:
                row = self.table[i][a]
                assert abs(sum(row.values()) - 1) < 1e-6, (self.id, i, a, sum(row.values()))
                for r in row:
                    assert r in self.responses, (self.id, r)
        for r in self.responses.values():
            if r.outcome:
                assert r.outcome in self.outcomes, (self.id, r.outcome)


DEFAULT_WEIGHTS = {
    "pragmatic_understanding": 0.25,
    "boundary_respect": 1.0,
    "clarification": 1.0,
    "excessive_assumption": 1.0,
    "social_friction": 1.0,
    "repair": 1.0,
    "task_completion": 1.0,
}

COMPONENT_DOC = {
    "pragmatic_understanding": "Log score of the agent's submitted belief on the true hidden intent, relative to a uniform belief (a proper scoring rule: rewards calibration, punishes over- and under-confidence). Zero if no belief is submitted.",
    "boundary_respect": "Penalty for boundary-crossing moves (pressure, presumption); larger when the hidden intent is a refusal or avoidance.",
    "clarification": "+ for the first clarifying move while the observable posterior is genuinely uncertain; − for repeated or redundant clarification.",
    "excessive_assumption": "Penalty for acting as if certain while the observable posterior is still uncertain.",
    "social_friction": "Negative of the friction the interlocutor accumulates in response to the move.",
    "repair": "+ when a repair move reduces friction that had built up.",
    "task_completion": "Terminal value of the interaction outcome given the true intent (e.g. a confirmed plan with an interested friend, or a graceful close with a friend who wanted to decline).",
}


class PragmaticEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, spec: EnvSpec, weights: dict[str, float] | None = None):
        super().__init__()
        spec.validate()
        self.spec_ = spec
        self.domain = get_domain(spec.domain)
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.action_space = spaces.Discrete(len(self.domain.actions))
        # Observations are rich dicts (text + structured fields); the declared
        # space covers the numeric part.
        self.observation_space = spaces.Dict({"turn": spaces.Discrete(spec.horizon + 1)})
        self.session_id = None

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self._seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.rng = random.Random(self._seed)
        kind, ref = self.spec_.opening
        if kind == "family":
            factors = {**self.spec_.default_factors, **options.get("factors", {})}
            scen = render(ref, factors, variant=options.get("variant", self._seed % 3))
            prior = scen.reference.distribution
        else:
            scen = get_scenario(ref)
            factors = None
            prior = scen.reference.distribution
        self.scenario = scen
        self.prior = dict(prior)
        ids = self.domain.intent_ids
        if "force_intent" in options:  # used by tests and policy audits only
            intent = options["force_intent"]
        else:
            intent = self.rng.choices(ids, weights=[prior[i] for i in ids])[0]
        members = [p for p in scen.participants if p != "You"] or [scen.speaker]
        self._hidden = {
            "intent": intent,
            "patience": round(self.rng.uniform(0.55, 1.0), 2),
            "friction": 0.0,
            "members": members,
        }
        self.transcript: list[Message] = [m.model_copy() for m in scen.history]
        self.posterior = dict(prior)  # observable Bayesian posterior
        self.revealed: str | None = None
        self.turn = 0
        self.clarify_count = 0
        self.done = False
        self.outcome: str | None = None
        self.log: list[dict] = []
        self.factors = factors
        self.session_id = uuid.uuid4().hex[:12]
        return self._obs(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is over; call reset().")
        if isinstance(action, (int, np.integer)):
            action = {"action": self.domain.action_ids[int(action)]}
        aid = action["action"]
        if aid not in self.domain.action_ids:
            raise ValueError(f"Unknown action {aid}")
        belief = action.get("belief")
        message = action.get("message") or self.domain.action(aid).example
        hid = self._hidden
        intent = hid["intent"]
        roles = self.spec_.roles

        post_before = dict(self.posterior)
        h_before = normalized_entropy(post_before)
        friction_before = hid["friction"]

        self.transcript.append(Message(sender="You", text=message, time=f"Turn {self.turn + 1}", is_you=True))

        # --- interlocutor response -------------------------------------
        row = self.spec_.table[intent][aid]
        rid = self.rng.choices(list(row), weights=list(row.values()))[0]
        resp = self.spec_.responses[rid]
        friction_delta = resp.friction
        if aid in roles.get("clarify", set()) and self.clarify_count >= 1:
            friction_delta += 0.15 * self.clarify_count  # asking again and again is itself costly
        hid["friction"] = max(0.0, hid["friction"] + friction_delta)
        texts = self._render_response(resp)
        withdrew = hid["friction"] >= hid["patience"] and resp.outcome is None
        if withdrew:
            texts.append((self.scenario.speaker if len(hid["members"]) == 1 else hid["members"][0], self.rng.choice(self.spec_.withdraw_text)))
        for sender, txt in texts:
            self.transcript.append(Message(sender=sender, text=txt, time=f"Turn {self.turn + 1}"))

        # --- observable Bayesian update (what any agent *could* know) ----
        like = {i: self.spec_.table[i][aid].get(rid, 0.0) for i in self.domain.intent_ids}
        z = sum(self.posterior[i] * like[i] for i in like)
        if z > 0:
            self.posterior = {i: self.posterior[i] * like[i] / z for i in like}
        if resp.reveals:
            self.revealed = resp.reveals

        # --- reward components -----------------------------------------
        comp = {k: 0.0 for k in DEFAULT_WEIGHTS}
        K = len(self.domain.intent_ids)
        if belief:
            b = {i: max(float(belief.get(i, 0.0)), 0.0) for i in self.domain.intent_ids}
            s = sum(b.values()) or 1.0
            comp["pragmatic_understanding"] = math.log(max(b[intent] / s, 1e-3)) - math.log(1 / K)
        refusal_like = self.domain.utility.get(aid, {}).get(intent, 0) < -0.5
        if aid in roles.get("boundary", set()):
            comp["boundary_respect"] = -1.0 if refusal_like else -0.4
        if aid in roles.get("clarify", set()):
            if self.revealed and self.revealed != resp.reveals:
                comp["clarification"] = -0.2  # already known
            elif self.clarify_count == 0 and h_before >= 0.70:
                comp["clarification"] = 0.3
            elif self.clarify_count >= 1:
                comp["clarification"] = -0.25
            self.clarify_count += 1
        if aid in roles.get("assume", set()) and max(post_before.values()) < 0.6 and not self.revealed:
            comp["excessive_assumption"] = -0.5
        comp["social_friction"] = -max(0.0, friction_delta)
        if aid in roles.get("repair", set()) and friction_before >= 0.25 and friction_delta <= 0:
            comp["repair"] = 0.3

        self.turn += 1
        terminated = False
        outcome = resp.outcome
        if withdrew:
            outcome = "withdrew"
        if outcome:
            terminated = True
            self.outcome = outcome
            comp["task_completion"] = self.spec_.outcomes[outcome].value[intent]
        truncated = (not terminated) and self.turn >= self.spec_.horizon
        if truncated:
            self.outcome = "timeout"
            comp["task_completion"] = self.spec_.outcomes["timeout"].value[intent]
        self.done = terminated or truncated

        reward = float(sum(self.weights[k] * v for k, v in comp.items()))
        entry = {
            "turn": self.turn, "action": aid, "message": message, "belief": belief,
            "response": rid, "response_note": resp.note, "response_texts": texts,
            "friction_before": round(friction_before, 3), "friction_after": round(hid["friction"], 3),
            "posterior_before": post_before, "posterior_after": dict(self.posterior),
            "observable_entropy_before": round(h_before, 3),
            "components": comp, "weighted": {k: self.weights[k] * v for k, v in comp.items()},
            "reward": reward, "outcome": self.outcome if self.done else None,
        }
        self.log.append(entry)
        return self._obs(), reward, terminated, truncated, self._info(entry)

    # ------------------------------------------------------------ helpers
    def _render_response(self, resp: ResponseType) -> list[tuple[str, str]]:
        choice = self.rng.choice(resp.texts)
        members = self._hidden["members"]
        if isinstance(choice, str):
            return [(self.scenario.speaker if len(members) == 1 else members[0], choice)]
        out = []
        slots = {f"{{{c}}}": m for c, m in zip("ABCDE", members)}
        for slot, txt in choice:
            sender = slots.get(slot, slot)
            for k, v in slots.items():
                txt = txt.replace(k, v)
            out.append((sender, txt))
        return out

    def observable(self) -> Observable:
        """The agent's view, in the same contract used by every model adapter."""
        obs = self.scenario.observable()
        return obs.model_copy(update={"history": [m.model_copy() for m in self.transcript]})

    def _obs(self) -> dict:
        o = self.observable()
        return {
            "turn": self.turn,
            "horizon": self.spec_.horizon,
            "context": o.context,
            "history": [m.model_dump() for m in o.history],
            "speaker": o.speaker,
            "relationship": o.relationship_label,
            "channel": o.channel,
            "participants": o.participants,
            "question": o.question,
            "interpretations": [i.model_dump() for i in o.interpretations],
            "actions": [a.model_dump() for a in o.actions],
            "done": self.done,
        }

    def _info(self, entry: dict | None = None) -> dict:
        return {"session_id": self.session_id, "seed": self._seed, "env_id": self.spec_.id,
                "last": entry, "outcome": self.outcome if self.done else None}

    def debug_state(self) -> dict:
        """Researcher-only view. Never sent to agents."""
        return {
            "hidden": copy.deepcopy(self._hidden),
            "prior": self.prior,
            "observable_posterior": self.posterior,
            "revealed_intent": self.revealed,
            "factors": self.factors,
            "outcome": self.outcome,
            "outcome_label": self.spec_.outcomes[self.outcome].label if self.outcome else None,
            "log": self.log,
            "weights": self.weights,
            "return": float(sum(e["reward"] for e in self.log)),
        }

    def render(self):
        return "\n".join(f"{m.sender}: {m.text}" for m in self.transcript)
