"""Recorded mode (the default): serve saved outputs of the local model instead of running it.

`scripts/record_demo.py` runs Qwen2.5-0.5B-Instruct once over everything the app
can ask for in a demo (Explorer scenarios, the corpus diagnosis, Counterfactual
Lab presets and single-factor sweeps, and model-played Playground episodes) and
saves each result keyed by a hash of the exact observable input. The API then
serves those results, so no model runs while the app is in use. The numbers are
real model outputs, labeled as recorded with their recording date.

Set PGYM_LIVE=1 to run models live instead.
"""
from __future__ import annotations

import gzip
import hashlib
import json

from pgym import config
from pgym.models.adapters import Adapter, InterpretationOutput, Provenance
from pgym.scenarios.schema import Observable

STORE_PATH = config.DATA_DIR / "recorded" / "local_qwen.json"
MODEL = "Qwen2.5-0.5B-Instruct"


class NotRecorded(KeyError):
    pass


def key(kind: str, payload) -> str:
    if isinstance(payload, Observable):
        payload = payload.model_dump()
    return f"{kind}:" + hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


_store: dict | None = None


def store() -> dict:
    global _store
    if _store is None:
        _store = json.loads(STORE_PATH.read_text()) if STORE_PATH.exists() else {}
    return _store


def has(kind: str, payload) -> bool:
    return key(kind, payload) in store()


def get(kind: str, payload):
    v = store().get(key(kind, payload))
    if v is None:
        raise NotRecorded(kind)
    return v


def put(kind: str, payload, value) -> None:
    store()[key(kind, payload)] = value


def save() -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(store(), separators=(",", ":")))
    tmp.replace(STORE_PATH)


# Data Lab responses (selection, comparison, export) are deterministic computations, saved the same way.
DATA_LAB_PATH = config.DATA_DIR / "recorded" / "data_lab.json.gz"
EXPORTS_DIR = config.DATA_DIR / "recorded" / "exports"
_data_lab: dict | None = None


def data_key(kind: str, strategy: str, budget: int, seed: int, target: str | None) -> str:
    if kind == "compare":
        strategy = "*"  # compare covers every strategy
    elif strategy != "failure_targeted":
        target = None  # only failure-targeted selection reads the target
    return f"{kind}:{strategy}:{budget}:{seed}:{target or 'any'}"


def data_lab() -> dict:
    global _data_lab
    if _data_lab is None:
        _data_lab = json.loads(gzip.decompress(DATA_LAB_PATH.read_bytes())) if DATA_LAB_PATH.exists() else {}
    return _data_lab


def save_data_lab() -> None:
    DATA_LAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_LAB_PATH.write_bytes(gzip.compress(json.dumps(data_lab(), separators=(",", ":")).encode()))


def provenance(recorded_at: str) -> dict:
    return {"backend": "recorded", "model": MODEL, "is_real_inference": True, "recorded_at": recorded_at,
            "label": f"Recorded local inference · {MODEL} · recorded {recorded_at[:10]}"}


class RecordedAdapter(Adapter):
    name = "recorded"
    model = MODEL

    def available(self) -> bool:
        return bool(store())

    def describe(self) -> dict:
        return {"backend": "recorded", "available": self.available(), "model": MODEL,
                "note": "Saved outputs of the local model, recorded once by scripts/record_demo.py. Nothing runs live."}

    def interpret(self, obs: Observable) -> InterpretationOutput:
        rec = get("interpret", obs)
        out = rec["output"]
        p = out["provenance"]
        prov = Provenance(**{**p, **provenance(p["timestamp"]), "latency_ms": None})
        return InterpretationOutput(**{k: out[k] for k in ("interpretation_probs", "most_likely", "rationale", "action", "reply")},
                                    provenance=prov)
