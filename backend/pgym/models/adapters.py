"""Provider-agnostic inference adapters with explicit provenance.

Backends
--------
* ``anthropic``      Claude via the official Anthropic SDK (structured JSON output).
* ``openai_compat``  Any OpenAI-compatible /v1/chat/completions endpoint (vLLM,
                     llama.cpp server, LM Studio, ...). Configure with
                     PGYM_OPENAI_BASE_URL / PGYM_OPENAI_MODEL.
* ``replay``         Re-serves a *recorded real inference* from
                     data/fixtures/recorded_inference.jsonl, labeled as a replay
                     with the original model, timestamp and prompt hash.
* ``fixture``        A deterministic heuristic stand-in. NOT a model. Labeled
                     `is_real_inference = False` everywhere it is shown.

Every result carries a `Provenance` record; the UI renders it next to any
model output so a fixture can never be mistaken for a measurement.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pgym import config
from pgym.models.prompts import (PROMPT_VERSION, SYSTEM, build_interpretation_prompt, prompt_hash,
                                 response_schema)
from pgym.scenarios.schema import Observable

RECORD_PATH = config.DATA_DIR / "fixtures" / "recorded_inference.jsonl"


@dataclass
class Provenance:
    backend: str
    model: str
    is_real_inference: bool
    label: str
    prompt_version: str = PROMPT_VERSION
    prompt_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    latency_ms: int | None = None
    recorded_at: str | None = None
    settings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


@dataclass
class InterpretationOutput:
    interpretation_probs: dict[str, float]
    most_likely: str
    rationale: str
    action: str
    reply: str
    provenance: Provenance
    raw: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _normalize(p: dict[str, float], ids: list[str]) -> dict[str, float]:
    vals = {i: max(0.0, float(p.get(i, 0.0))) for i in ids}
    z = sum(vals.values())
    if z <= 0:
        return {i: 1 / len(ids) for i in ids}
    return {i: v / z for i, v in vals.items()}


def _parse(text: str, obs: Observable) -> dict:
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    ids = [o.id for o in obs.interpretations]
    data["interpretation_probs"] = _normalize(data.get("interpretation_probs", {}), ids)
    if data.get("most_likely") not in ids:
        data["most_likely"] = max(data["interpretation_probs"], key=data["interpretation_probs"].get)
    if data.get("action") not in [a.id for a in obs.actions]:
        data["action"] = obs.actions[0].id
    return data


class Adapter:
    name = "base"

    def available(self) -> bool:
        return True

    def interpret(self, obs: Observable) -> InterpretationOutput:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name, "available": self.available()}


class AnthropicAdapter(Adapter):
    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = None):
        self.model = model or config.ANTHROPIC_MODEL
        self.effort = effort or config.ANTHROPIC_EFFORT
        self._client = None

    def available(self) -> bool:
        import os
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def client(self):
        if self._client is None:
            import os

            import anthropic
            headers = {}
            if os.environ.get("ANTHROPIC_WORKSPACE_ID"):  # needed for keys not scoped to a workspace
                headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
            self._client = anthropic.Anthropic(max_retries=3, default_headers=headers or None)
        return self._client

    def complete_json(self, system: str, prompt: str, schema: dict, max_tokens: int = 16000) -> tuple[str, dict, int]:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        latency = int((time.time() - t0) * 1000)
        if resp.stop_reason == "refusal":
            raise RuntimeError("Model declined the request (stop_reason=refusal).")
        text = next(b.text for b in resp.content if b.type == "text")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return text, usage, latency

    def interpret(self, obs: Observable) -> InterpretationOutput:
        prompt = build_interpretation_prompt(obs)
        text, usage, latency = self.complete_json(SYSTEM, prompt, response_schema(obs))
        data = _parse(text, obs)
        prov = Provenance(backend=self.name, model=self.model, is_real_inference=True,
                          label=f"Live inference · {self.model}", prompt_hash=prompt_hash(SYSTEM, prompt),
                          latency_ms=latency, settings={"effort": self.effort}, usage=usage)
        return InterpretationOutput(**{k: data[k] for k in ("interpretation_probs", "most_likely", "rationale", "action", "reply")},
                                    provenance=prov, raw=text)

    def describe(self) -> dict:
        return {**super().describe(), "model": self.model, "effort": self.effort}


class OpenAICompatAdapter(Adapter):
    """Generic OpenAI-compatible HTTP endpoint (for local open-weight servers)."""
    name = "openai_compat"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.model = model or config.OPENAI_MODEL

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def complete_json(self, system: str, prompt: str, schema: dict, max_tokens: int = 2000) -> tuple[str, dict, int]:
        import httpx
        t0 = time.time()
        r = httpx.post(f"{self.base_url}/chat/completions", timeout=120,
                       headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                       json={"model": self.model, "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt + "\nRespond with JSON only."}],
                             "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}})
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"], j.get("usage", {}), int((time.time() - t0) * 1000)

    def interpret(self, obs: Observable) -> InterpretationOutput:
        prompt = build_interpretation_prompt(obs)
        text, usage, latency = self.complete_json(SYSTEM, prompt, response_schema(obs))
        data = _parse(text, obs)
        prov = Provenance(backend=self.name, model=self.model, is_real_inference=True,
                          label=f"Live inference · {self.model} (OpenAI-compatible)", prompt_hash=prompt_hash(SYSTEM, prompt),
                          latency_ms=latency, usage=usage)
        return InterpretationOutput(**{k: data[k] for k in ("interpretation_probs", "most_likely", "rationale", "action", "reply")},
                                    provenance=prov, raw=text)

    def describe(self) -> dict:
        return {**super().describe(), "model": self.model, "base_url": self.base_url}


class ReplayAdapter(Adapter):
    """Serves recorded real model outputs, keyed by prompt hash."""
    name = "replay"

    def __init__(self, path: Path = RECORD_PATH):
        self.path = path
        self._index: dict[str, list[dict]] | None = None
        self._cursor: dict[str, int] = {}

    def _load(self) -> dict[str, list[dict]]:
        if self._index is None:
            self._index = {}
            if self.path.exists():
                for line in self.path.read_text().splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        self._index.setdefault(rec["prompt_hash"], []).append(rec)
        return self._index

    def available(self) -> bool:
        return bool(self._load())

    def has(self, obs: Observable) -> bool:
        return prompt_hash(SYSTEM, build_interpretation_prompt(obs)) in self._load()

    def interpret(self, obs: Observable) -> InterpretationOutput:
        h = prompt_hash(SYSTEM, build_interpretation_prompt(obs))
        recs = self._load().get(h)
        if not recs:
            raise KeyError("No recorded inference for this exact prompt.")
        i = self._cursor.get(h, 0)
        self._cursor[h] = i + 1
        rec = recs[i % len(recs)]
        out = rec["output"]
        prov = Provenance(backend="replay", model=rec["model"], is_real_inference=True,
                          label=f"Replay of recorded inference · {rec['model']} · recorded {rec['recorded_at'][:10]}",
                          prompt_hash=h, recorded_at=rec["recorded_at"], settings=rec.get("settings", {}))
        return InterpretationOutput(**{k: out[k] for k in ("interpretation_probs", "most_likely", "rationale", "action", "reply")},
                                    provenance=prov, raw=json.dumps(out))


class FixtureAdapter(Adapter):
    """Deterministic, keyword-based heuristic. Deliberately literal-minded.

    Exists so every screen works offline. It is NOT a language model and its
    outputs are never presented as inference results.
    """
    name = "fixture"

    def interpret(self, obs: Observable) -> InterpretationOutput:
        ids = [o.id for o in obs.interpretations]
        literal = next((o.id for o in obs.interpretations if o.literal), ids[0])
        logits = {i: 0.0 for i in ids}
        logits[literal] += 1.6  # literal-minded prior
        text = " ".join(obs.context + [m.text for m in obs.history]).lower()
        cues = {"declined": 0.5, "didn't come": 0.5, "bailed": 0.5, "third time": 0.6, "again": 0.2}
        bump = sum(w for k, w in cues.items() if k in text)
        for i in ids:
            if i not in (literal,):
                logits[i] += bump / max(1, len(ids) - 1)
        z = sum(math.exp(v) for v in logits.values())
        probs = {i: math.exp(v) / z for i, v in logits.items()}
        top = max(probs, key=probs.get)
        action = obs.actions[0].id
        prov = Provenance(backend="fixture", model="heuristic-fixture-v1", is_real_inference=False,
                          label="FIXTURE · deterministic heuristic, not a model",
                          prompt_hash=prompt_hash(SYSTEM, build_interpretation_prompt(obs)))
        return InterpretationOutput(interpretation_probs=probs, most_likely=top,
                                    rationale="Fixture output: a literal-minded keyword heuristic used for offline demos. Not model inference.",
                                    action=action, reply=next(a.example for a in obs.actions if a.id == action), provenance=prov)


def record(obs: Observable, out: InterpretationOutput, path: Path = RECORD_PATH) -> None:
    """Append a real inference result to the replay store."""
    if not out.provenance.is_real_inference or out.provenance.backend == "replay":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"prompt_hash": out.provenance.prompt_hash, "scenario_id": obs.scenario_id, "model": out.provenance.model,
           "recorded_at": out.provenance.timestamp, "settings": out.provenance.settings,
           "output": {k: getattr(out, k) for k in ("interpretation_probs", "most_likely", "rationale", "action", "reply")}}
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


_ANTHROPIC = AnthropicAdapter()
_OPENAI = OpenAICompatAdapter()
_REPLAY = ReplayAdapter()
_FIXTURE = FixtureAdapter()


def get_adapter(name: str | None = None, model: str | None = None) -> Adapter:
    name = name or config.BACKEND
    if name == "anthropic":
        return AnthropicAdapter(model=model) if model else _ANTHROPIC
    if name == "openai_compat":
        return _OPENAI
    if name == "replay":
        return _REPLAY
    if name == "fixture":
        return _FIXTURE
    # auto: live if possible, else replay, else fixture
    if _ANTHROPIC.available():
        return _ANTHROPIC
    if _OPENAI.available():
        return _OPENAI
    if _REPLAY.available():
        return _REPLAY
    return _FIXTURE


def backends_status() -> list[dict]:
    return [a.describe() for a in (_ANTHROPIC, _OPENAI, _REPLAY, _FIXTURE)]
