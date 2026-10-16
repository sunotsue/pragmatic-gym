"""Pragmatic Gym API.

    cd backend && .venv/bin/uvicorn pgym.api.main:app --port 8000

Latent fields (hidden state, reference labels, LSSM terms, environment
internals) are only returned when the request carries the researcher token
(header `x-researcher-token`), mirroring the rule that hidden state never
reaches an agent.
"""
from __future__ import annotations

import base64
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from urllib.parse import urlencode

import yaml
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pgym import config
from pgym.causal import analysis as cf_analysis
from pgym.causal.design import DesignSpec, build
from pgym.environments.engine import COMPONENT_DOC, DEFAULT_WEIGHTS
from pgym.environments.policies import audit, policy_bayes
from pgym.environments.specs import SPECS, describe, make
from pgym.evaluation.metrics import FAILURE_INFO, score_item, summarize
from pgym.models.adapters import backends_status, get_adapter, record
from pgym.models.recorded import NotRecorded
from pgym.scenarios.corpus import get_scenario, load_corpus
from pgym.scenarios.schema import CAPABILITIES, Observable, Scenario
from pgym.social.domains import DOMAINS
from pgym.social.factors import FACTORS, violations
from pgym.social.generator import FAMILY_BY_ID, describe_families, full_levels, render
from pgym.storage import db

app = FastAPI(title="Pragmatic Gym", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()  # serializes local-model inference (one MPS model at a time)


def is_researcher(token: str | None) -> bool:
    return token == config.RESEARCHER_TOKEN


# ------------------------------------------------------------------ adapters
def adapter_for(backend: str | None, model: str | None = None):
    if backend == "recorded":
        from pgym.models.recorded import RecordedAdapter
        return RecordedAdapter()
    if backend in ("local_hf", "local"):
        from pgym.models.local_hf import LocalHFAdapter
        return LocalHFAdapter(model or None)
    if backend == "local_lora":
        from pgym.models.local_hf import LocalHFAdapter
        if not model:
            raise HTTPException(400, "local_lora requires `model` = adapter path")
        return LocalHFAdapter(adapter_path=model, label="Qwen2.5-0.5B + LoRA")
    return get_adapter(backend, model)


@lru_cache(maxsize=1)
def _anthropic_probe() -> dict:
    """One tiny request to learn whether the configured key actually works."""
    from pgym.models.adapters import AnthropicAdapter
    a = AnthropicAdapter()
    if not a.available():
        return {"ok": False, "reason": "ANTHROPIC_API_KEY not set"}
    try:
        a.client.messages.create(model=a.model, max_tokens=16, messages=[{"role": "user", "content": "ping"}])
        return {"ok": True}
    except Exception as e:  # report the provider's reason verbatim
        return {"ok": False, "reason": str(e)[:300]}


def default_backend() -> str:
    if not config.LIVE:
        return "recorded"
    if config.BACKEND not in ("auto", ""):
        return config.BACKEND
    if _anthropic_probe()["ok"]:
        return "anthropic"
    from pgym.models.local_hf import LocalHFAdapter
    if LocalHFAdapter().available():
        return "local_hf"
    from pgym.models.adapters import ReplayAdapter
    return "replay" if ReplayAdapter().available() else "fixture"


def backend_or_recorded(backend: str | None) -> str:
    """In recorded mode every request is served from saved outputs, whatever backend it names."""
    return (backend or default_backend()) if config.LIVE else "recorded"


NOT_RECORDED = "Not recorded. The app is in recorded mode and only serves saved model outputs; this input wasn't recorded."


def _json_safe(o):
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


@app.get("/api/status")
def status():
    if not config.LIVE:
        from pgym.models.recorded import RecordedAdapter
        return {"default_backend": "recorded", "backends": [RecordedAdapter().describe()], "versions": versions(), "live": False}
    from pgym.models.local_hf import LocalHFAdapter
    probe = _anthropic_probe()
    backs = backends_status()
    for b in backs:
        if b["backend"] == "anthropic":
            b["available"] = probe["ok"]
            b["reason"] = probe.get("reason")
    loc = LocalHFAdapter()
    backs.insert(0, {**loc.describe(), "note": "Real inference with a local open-weight model (next-token letter distribution)."})
    return {"default_backend": default_backend(), "backends": backs, "versions": versions()}


def versions() -> dict:
    from pgym.models.prompts import PROMPT_VERSION
    from pgym.scenarios.corpus import CORPUS_VERSION
    from pgym.social.domains import DOMAIN_VERSION
    from pgym.social.lssm import LSSM_VERSION
    from pgym.training.formats import FORMAT_VERSION
    return {"corpus": CORPUS_VERSION, "lssm": LSSM_VERSION, "domains": DOMAIN_VERSION, "prompt": PROMPT_VERSION, "mc_format": FORMAT_VERSION}


@app.get("/api/meta")
def meta():
    return {
        "capabilities": CAPABILITIES,
        "domains": {k: d.to_dict() for k, d in DOMAINS.items()},
        "factors": [f.to_dict() for f in FACTORS],
        "families": describe_families(),
        "envs": [{"id": s.id, "title": s.title, "subtitle": s.subtitle, "flagship": s.flagship, "domain": s.domain,
                  "capabilities": s.capabilities, "status": s.status} for s in SPECS.values()],
        "failures": FAILURE_INFO,
        "reward_components": COMPONENT_DOC,
        "versions": versions(),
    }


# ------------------------------------------------------------------ scenarios
def scenario_payload(s: Scenario, researcher: bool) -> dict:
    d = {"id": s.id, "family_id": s.family_id, "title": s.title, "capability": s.capability, "capabilities": s.capabilities,
         "domain": s.domain, "source": s.source, "split": s.split, "synthetic": s.synthetic,
         "counterfactual_of": s.counterfactual_of, "counterfactual_change": s.counterfactual_change,
         "observable": s.observable().model_dump()}
    if researcher:
        dom = DOMAINS[s.domain]
        d["latent"] = {"reference": s.reference.model_dump(), "hidden": s.hidden, "why_hidden_matters": s.why_hidden_matters,
                       "factors": s.factors, "reference_actions": dom.reference_actions(s.reference.distribution)}
    return d


@app.get("/api/scenarios")
def scenarios(x_researcher_token: str | None = Header(None)):
    r = is_researcher(x_researcher_token)
    out = []
    for s in load_corpus():
        last = s.history[-1]
        out.append({"id": s.id, "title": s.title, "capability": s.capability, "domain": s.domain,
                    "relationship": s.relationship_label, "speaker": s.speaker, "channel": s.channel,
                    "counterfactual_of": s.counterfactual_of, "counterfactual_change": s.counterfactual_change,
                    "preview": last.text, "n_messages": len(s.history),
                    **({"ambiguity": __import__("pgym.social.lssm", fromlist=["x"]).normalized_entropy(s.reference.distribution)} if r else {})})
    return out


@app.get("/api/scenarios/{sid}")
def scenario(sid: str, x_researcher_token: str | None = Header(None)):
    try:
        s = get_scenario(sid)
    except NotRecorded:
        raise HTTPException(404, sid)
    return scenario_payload(s, is_researcher(x_researcher_token))


class InferRequest(BaseModel):
    scenario_id: str | None = None
    family_id: str | None = None
    levels: dict[str, str] = Field(default_factory=dict)
    variant: int = 0
    backend: str | None = None
    model: str | None = None


def _resolve(req: InferRequest) -> Scenario:
    if req.scenario_id:
        return get_scenario(req.scenario_id)
    if req.family_id:
        return render(req.family_id, req.levels, req.variant)
    raise HTTPException(400, "scenario_id or family_id required")


@app.post("/api/infer")
def infer(req: InferRequest, x_researcher_token: str | None = Header(None)):
    s = _resolve(req)
    backend = backend_or_recorded(req.backend)
    a = adapter_for(backend, req.model)
    obs = s.observable()
    t0 = time.time()
    try:
        with _lock:
            out = a.interpret(obs)
    except NotRecorded:
        raise HTTPException(409, NOT_RECORDED)
    except Exception as e:
        raise HTTPException(502, f"{backend} inference failed: {e}")
    if out.provenance.backend == "anthropic":
        record(obs, out)
    res = {"scenario_id": s.id, "output": out.to_dict(), "wall_ms": int((time.time() - t0) * 1000)}
    if is_researcher(x_researcher_token):
        lit = next((o.id for o in s.interpretations if o.literal), None)
        res["score"] = score_item(s.domain, s.reference.distribution, out.interpretation_probs, out.action, lit)
    db.save(f"infer-{int(time.time() * 1000)}", "inference", s.title, {"scenario_id": s.id, "backend": backend},
            {"probs": out.interpretation_probs, "action": out.action}, backend, out.provenance.model, out.provenance.is_real_inference)
    return res


@app.get("/api/diagnosis")
def diagnosis(backend: str = "local_hf", model: str | None = None, x_researcher_token: str | None = Header(None)):
    """Model-wide diagnosis on the hand-written corpus (cached on disk per backend/model)."""
    if not is_researcher(x_researcher_token):
        raise HTTPException(403, "researcher token required")
    if not config.LIVE:
        from pgym.models import recorded
        try:
            return recorded.get("diagnosis", "corpus")
        except recorded.NotRecorded:
            raise HTTPException(409, NOT_RECORDED)
    key = f"{backend}:{model or 'default'}".replace("/", "_")
    path = config.RUNTIME_DIR / "diagnosis" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    items = [s for s in load_corpus()]
    rows = []
    if backend == "local_hf":
        from pgym.models.local_hf import DEFAULT_LOCAL_MODEL, load
        from pgym.selection.pool import item_from_scenario
        from pgym.training.local_eval import score_items
        with _lock:
            tok, m = load(model or DEFAULT_LOCAL_MODEL, None, "float16")
            res = score_items(tok, m, [item_from_scenario(s, split="eval_handwritten", source="handwritten") for s in items])
        for s, r in zip(items, res):
            rows.append({"scenario_id": s.id, "title": s.title, "capability": s.capability, "probs": r["probs"], "action": r["action"], "score": r["score"]})
        prov = {"backend": "local_hf", "model": model or DEFAULT_LOCAL_MODEL, "is_real_inference": True}
    else:
        a = adapter_for(backend, model)
        def one(s):
            out = a.interpret(s.observable())
            lit = next((o.id for o in s.interpretations if o.literal), None)
            return {"scenario_id": s.id, "title": s.title, "capability": s.capability, "probs": out.interpretation_probs,
                    "action": out.action, "score": score_item(s.domain, s.reference.distribution, out.interpretation_probs, out.action, lit)}
        with ThreadPoolExecutor(6) as ex:
            rows = list(ex.map(one, items))
        prov = {"backend": a.name, "model": getattr(a, "model", a.name), "is_real_inference": a.name not in ("fixture",)}
    by_cap = {}
    for r in rows:
        by_cap.setdefault(r["capability"], []).append(r["score"])
    out = {"provenance": prov, "rows": rows, "overall": summarize([r["score"] for r in rows]),
           "by_capability": {k: summarize(v) for k, v in by_cap.items()}, "created": time.strftime("%Y-%m-%d %H:%M")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------ counterfactual lab
class PreviewRequest(BaseModel):
    family_id: str = "inv_let_you_know"
    levels: dict[str, str] = Field(default_factory=dict)
    variant: int = 0


@app.post("/api/counterfactual/preview")
def cf_preview(req: PreviewRequest, x_researcher_token: str | None = Header(None)):
    fam = FAMILY_BY_ID[req.family_id]
    lv = full_levels(fam.domain, req.levels)
    bad = violations(lv, fam.domain)
    if bad:
        return {"valid": False, "violations": bad, "levels": lv}
    s = render(req.family_id, lv, req.variant)
    return {"valid": True, "levels": lv, "scenario": scenario_payload(s, is_researcher(x_researcher_token))}


@app.post("/api/counterfactual/design")
def cf_design(spec: DesignSpec):
    try:
        d = build(spec)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {k: v for k, v in d.items() if k != "units"} | {"n_units": d["n_units"]}


class RunRequest(BaseModel):
    spec: DesignSpec
    backend: str | None = None
    model: str | None = None


def _run_units(design: dict, backend: str, model: str | None) -> tuple[list[dict], dict]:
    fam_id = design["spec"]["family_id"]
    scens = [render(fam_id, next(c["levels"] for c in design["cells"] if c["cell"] == u["cell"]), u["variant"]) for u in design["units"]]
    if backend == "recorded":
        from pgym.models import recorded
        missing = sum(not recorded.has("cf", s.observable()) for s in scens)
        if missing:
            raise HTTPException(409, f"This design wasn't recorded ({missing} of {len(scens)} prompts missing). "
                                     "Recorded designs: every preset, and comparing any one variable on its own, with up to 6 surface variants.")
        recs = [recorded.get("cf", s.observable()) for s in scens]
        return [{"probs": r["probs"], "action": r["action"]} for r in recs], recorded.provenance(recs[0]["recorded_at"])
    if backend == "local_hf":
        from pgym.models.local_hf import DEFAULT_LOCAL_MODEL, load
        from pgym.selection.pool import item_from_scenario
        from pgym.training.local_eval import score_items
        with _lock:
            tok, m = load(model or DEFAULT_LOCAL_MODEL, None, "float16")
            res = score_items(tok, m, [item_from_scenario(s, split="cf") for s in scens])
        return [{"probs": r["probs"], "action": r["action"]} for r in res], {"backend": "local_hf", "model": model or DEFAULT_LOCAL_MODEL, "is_real_inference": True, "label": f"Live local inference · {(model or DEFAULT_LOCAL_MODEL).split('/')[-1]}"}
    a = adapter_for(backend, model)

    def one(s):
        try:
            o = a.interpret(s.observable())
            if o.provenance.backend == "anthropic":
                record(s.observable(), o)
            return {"probs": o.interpretation_probs, "action": o.action, "prov": o.provenance.__dict__}
        except Exception as e:
            return {"error": str(e)}
    with ThreadPoolExecutor(8 if backend == "anthropic" else 1) as ex:
        res = list(ex.map(one, scens))
    errs = [r for r in res if "error" in r]
    if len(errs) == len(res):
        raise HTTPException(502, errs[0]["error"])
    prov = next((r["prov"] for r in res if "prov" in r), {})
    return [None if "error" in r else r for r in res], {k: prov.get(k) for k in ("backend", "model", "is_real_inference", "label")}


@app.post("/api/counterfactual/run")
def cf_run(req: RunRequest):
    try:
        design = build(req.spec)
    except ValueError as e:
        raise HTTPException(400, str(e))
    backend = backend_or_recorded(req.backend)
    t0 = time.time()
    results, prov = _run_units(design, backend, req.model)
    an = _json_safe(cf_analysis.analyze(design, results))
    eid = f"cf-{design['design_id']}-{backend}-{int(time.time())}"
    payload = {"experiment_id": eid, "design": {k: v for k, v in design.items() if k != "units"}, "provenance": prov,
               "analysis": an, "runtime_s": round(time.time() - t0, 1),
               "units": [{**u, "result": r} for u, r in zip(design["units"], results)]}
    path = config.RUNTIME_DIR / "counterfactual" / f"{eid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    db.save(eid, "counterfactual", f"{design['spec']['family_id']} × {', '.join(design['factors'])}", design["spec"],
            {"n_units": design["n_units"], "variance": an.get("variance")}, prov.get("backend") or backend, prov.get("model") or "",
            prov.get("is_real_inference"), str(path))
    return payload


@app.get("/api/experiments")
def experiments(kind: str | None = None):
    return db.list_experiments(kind)


@app.get("/api/counterfactual/{eid}")
def cf_get(eid: str):
    p = config.RUNTIME_DIR / "counterfactual" / f"{eid}.json"
    if not p.exists():
        raise HTTPException(404, eid)
    return json.loads(p.read_text())


# ------------------------------------------------------------------ environments
# Sessions are stateless: the session id is a token holding the environment, seed, opening
# factors, reward weights and every step taken so far. Environments are deterministic given
# the seed, so any server instance rebuilds the exact state by replaying the steps.
def _token(state: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(state, separators=(",", ":")).encode()).decode().rstrip("=")


def _session(sid: str):
    try:
        state = json.loads(base64.urlsafe_b64decode(sid + "=" * (-len(sid) % 4)))
        env = make(state["env"], state.get("weights"))
        env.reset(seed=state["seed"], options={"factors": state["factors"]} if state["factors"] else {})
        for act in state["steps"]:
            env.step(act)
    except Exception:
        raise HTTPException(404, "session not found")
    return env, state


class ResetRequest(BaseModel):
    seed: int | None = None
    factors: dict[str, str] = Field(default_factory=dict)
    weights: dict[str, float] | None = None


class StepRequest(BaseModel):
    action: str
    belief: dict[str, float] | None = None
    message: str | None = None


class AgentRequest(BaseModel):
    policy: str = "model"  # model | bayes
    backend: str | None = None
    model: str | None = None


@app.get("/api/envs")
def envs():
    return [describe(s) for s in SPECS.values()]


@app.post("/api/envs/{env_id}/reset")
def env_reset(env_id: str, req: ResetRequest, x_researcher_token: str | None = Header(None)):
    if env_id not in SPECS:
        raise HTTPException(404, env_id)
    env = make(env_id, req.weights)
    opts = {"factors": req.factors} if req.factors else {}
    try:
        obs, info = env.reset(seed=req.seed, options=opts)
    except ValueError as e:
        raise HTTPException(400, str(e))
    sid = _token({"env": env_id, "seed": env._seed, "factors": req.factors, "weights": req.weights, "steps": []})
    return {"session_id": sid, "observation": obs, "info": info,
            **({"debug": env.debug_state()} if is_researcher(x_researcher_token) else {})}


def _step(env, state: dict, act: dict, researcher: bool, extra: dict | None = None):
    obs, r, term, trunc, info = env.step(act)
    state["steps"].append(act)
    out = {"session_id": _token(state), "observation": obs, "reward": r, "terminated": term, "truncated": trunc, "info": info, **(extra or {})}
    if not researcher:  # agents and non-researcher clients never see reward internals that depend on hidden state
        out["info"] = {k: v for k, v in info.items() if k != "last"} | {"last": {"turn": info["last"]["turn"], "action": info["last"]["action"]}}
        out.pop("reward")
    else:
        out["debug"] = env.debug_state()
    return out


@app.post("/api/envs/sessions/{sid}/step")
def env_step(sid: str, req: StepRequest, x_researcher_token: str | None = Header(None)):
    env, state = _session(sid)
    if env.done:
        raise HTTPException(400, "episode finished")
    return _step(env, state, req.model_dump(exclude_none=True), is_researcher(x_researcher_token))


@app.post("/api/envs/sessions/{sid}/agent_step")
def env_agent_step(sid: str, req: AgentRequest, x_researcher_token: str | None = Header(None)):
    env, state = _session(sid)
    if env.done:
        raise HTTPException(400, "episode finished")
    if req.policy == "bayes":
        act = policy_bayes(env, None)
        act["belief"] = {k: float(v) for k, v in act["belief"].items()}
        act["message"] = env.domain.action(act["action"]).example
        prov = {"backend": "policy", "model": "bayes-observable-posterior", "is_real_inference": False,
                "label": "Reference policy · Bayes-optimal on the observable posterior (not a model)"}
        agent = {"belief": act["belief"], "action": act["action"], "rationale": "Greedy expected utility under the exact observable posterior."}
    else:
        backend = backend_or_recorded(req.backend)
        a = adapter_for(backend, req.model)
        try:
            with _lock:
                out = a.interpret(env.observable())  # observable view only
        except NotRecorded:
            raise HTTPException(409, "Not recorded. Recorded model moves cover episodes the model plays from the start: "
                                     "press Reset episode, then use Let the model act for every turn.")
        except Exception as e:
            raise HTTPException(502, f"{backend} inference failed: {e}")
        act = {"action": out.action, "belief": out.interpretation_probs, "message": out.reply}
        prov = out.provenance.__dict__
        agent = {"belief": out.interpretation_probs, "action": out.action, "rationale": out.rationale}
    return _step(env, state, act, is_researcher(x_researcher_token), {"agent": agent, "provenance": prov})


@app.get("/api/envs/sessions/{sid}/debug")
def env_debug(sid: str, x_researcher_token: str | None = Header(None)):
    if not is_researcher(x_researcher_token):
        raise HTTPException(403, "researcher token required")
    env, _ = _session(sid)
    return env.debug_state()


_AUDITS: dict = {}


@app.get("/api/envs/{env_id}/audit")
def env_audit(env_id: str, n: int = 200, planner: bool = True):
    key = (env_id, n, planner)
    saved = config.DATA_DIR / "recorded" / "audits.json"
    if key not in _AUDITS and saved.exists():  # precomputed by scripts/record_demo.py --only audit (no model involved)
        hit = json.loads(saved.read_text()).get(f"{env_id}:{n}:{planner}")
        if hit:
            _AUDITS[key] = hit
    if key not in _AUDITS and not config.LIVE:
        raise HTTPException(409, NOT_RECORDED)
    if key not in _AUDITS:
        _AUDITS[key] = audit(env_id, n=n, planner=planner)
    return _AUDITS[key]


# ------------------------------------------------------------------ data lab
def _saved_data_lab(kind: str, req) -> dict:
    from pgym.models import recorded
    hit = recorded.data_lab().get(recorded.data_key(kind, req.strategy, req.budget, req.seed, req.target_failure))
    if hit is None:
        raise HTTPException(409, NOT_RECORDED)
    return hit


class SelectRequest(BaseModel):
    strategy: str = "failure_targeted"
    budget: int = 120000
    seed: int = 0
    target_failure: str | None = None


def _slim(row: dict) -> dict:
    it = row["item"]
    o = it["observable"]
    return {"id": it["id"], "family_id": it["family_id"], "domain": it["domain"], "capability": it["capability"],
            "source": it["source"], "cluster": it.get("cluster"), "contrast_factor": it.get("contrast_factor"),
            "factors": it.get("factors"), "speaker": o["speaker"], "relationship": o["relationship_label"],
            "context": o["context"], "history": o["history"][-4:], "reference": it["reference"],
            "target_action": it["target_action"], "reason": row["reason"], "tokens": row["tokens"],
            "base": {"probs": row["base"]["probs"], "action": row["base"]["action"], "failures": row["base"]["score"]["failures"],
                     "cross_entropy": row["base"]["score"]["cross_entropy"]} if row.get("base") else None,
            "turn": it.get("turn"), "behaviour_action": it.get("behaviour_action"), "behaviour_response": it.get("behaviour_response"),
            "behaviour_reward_components": it.get("behaviour_reward_components")}


@app.get("/api/data/strategies")
def data_strategies():
    from pgym.selection.strategies import STRATEGIES, STRATEGY_INFO, base_scores, pool
    return {"strategies": STRATEGIES, "info": STRATEGY_INFO, "pool_size": len(pool()), "base_scored": len(base_scores())}


@app.get("/api/data/pool_diagnosis")
def data_pool_diagnosis():
    """Base-model failure profile on the candidate pool (what failure-targeting acts on)."""
    from collections import Counter

    from pgym.selection.strategies import base_scores, pool
    bs = base_scores()
    if not bs:
        return {"available": False}
    P = {it["id"]: it for it in pool()}
    prim = Counter(r["score"]["primary_failure"] for r in bs.values())
    anyf = Counter(f for r in bs.values() for f in r["score"]["failures"])
    by_dom = {}
    for rid, r in bs.items():
        d = P[rid]["domain"] if rid in P else "?"
        by_dom.setdefault(d, []).append(r["score"]["cross_entropy"])
    return {"available": True, "n": len(bs), "model": next(iter(bs.values()))["model"], "primary": dict(prim), "any": dict(anyf),
            "cross_entropy_by_domain": {k: sum(v) / len(v) for k, v in by_dom.items()}}


@app.post("/api/data/select")
def data_select(req: SelectRequest):
    if not config.LIVE:
        return _saved_data_lab("select", req)
    from pgym.selection.strategies import diagnostics, select
    try:
        sel = select(req.strategy, req.budget, req.seed, req.target_failure)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    diag = diagnostics(sel)
    return {k: v for k, v in sel.items() if k != "rows"} | {"rows": [_slim(r) for r in sel["rows"]], "diagnostics": diag}


@app.post("/api/data/compare")
def data_compare(req: SelectRequest):
    if not config.LIVE:
        return _saved_data_lab("compare", req)
    from pgym.selection.strategies import STRATEGIES, diagnostics, select
    out = {}
    for s in STRATEGIES:
        try:
            sel = select(s, req.budget, req.seed, req.target_failure)
        except RuntimeError as e:
            out[s] = {"error": str(e)}
            continue
        d = diagnostics(sel)
        bases = [r["base"] for r in sel["rows"] if r.get("base")]
        out[s] = {"n": sel["n"], "tokens": sel["tokens_used"], "diagnostics": {k: d[k] for k in ("leakage", "duplicates", "distribution", "warnings", "factor_coverage")},
                  "mean_base_loss": (sum(b["score"]["cross_entropy"] for b in bases) / len(bases)) if bases else None,
                  "base_failure_rate": (sum(bool(set(b["score"]["failures"]) - {"pass"}) for b in bases) / len(bases)) if bases else None,
                  "n_clusters": len({r["item"].get("cluster") for r in sel["rows"] if r["item"].get("cluster")}),
                  "n_contrast_pairs": sum(1 for r in sel["rows"] if r["item"].get("contrast_factor"))}
    return out


@app.post("/api/data/export")
def data_export(req: SelectRequest):
    if not config.LIVE:
        return _saved_data_lab("export", req)
    from pgym.selection.strategies import export, select
    sel = select(req.strategy, req.budget, req.seed, req.target_failure)
    out_dir = config.RUNTIME_DIR / "datasets" / f"{req.strategy}-{sel['selection_id']}"
    man = export(sel, out_dir, seed=req.seed)
    db.save(f"ds-{sel['selection_id']}", "dataset", f"{req.strategy} @ {req.budget} tokens", req.model_dump(), man, artifact_path=str(out_dir))
    sample = [json.loads(x) for x in (out_dir / "sft.jsonl").read_text().splitlines()[:2]]
    return {"manifest": man, "path": str(out_dir), "sample_sft": sample,
            "sample_preference": json.loads((out_dir / "preference.jsonl").read_text().splitlines()[0]),
            "downloads": {f: "/api/data/download?" + urlencode({**req.model_dump(exclude_none=True), "file": f}) for f in ("sft.jsonl", "preference.jsonl")}}


@app.get("/api/data/download")
def data_download(file: str, strategy: str = "failure_targeted", budget: int = 120000, seed: int = 0, target_failure: str | None = None):
    """Rebuilds the export from its settings (exports are deterministic), so it works on any server instance."""
    from fastapi.responses import FileResponse, Response
    from pgym.selection.strategies import export, select
    if file not in ("sft.jsonl", "preference.jsonl"):
        raise HTTPException(404, "unknown export file")
    if not config.LIVE:
        from pgym.models import recorded
        key = recorded.data_key("export", strategy, budget, seed, target_failure)
        path = recorded.EXPORTS_DIR / f"{key.replace(':', '_')}.{file}.gz"
        if not path.exists():
            raise HTTPException(409, NOT_RECORDED)
        name = recorded.data_lab()[key]["path"].rsplit("/", 1)[-1]
        return Response(gzip.decompress(path.read_bytes()), media_type="application/jsonl",
                        headers={"Content-Disposition": f'attachment; filename="{name}-{file}"'})
    sel = select(strategy, budget, seed, target_failure)
    out_dir = config.RUNTIME_DIR / "datasets" / f"{strategy}-{sel['selection_id']}"
    if not (out_dir / file).exists():
        export(sel, out_dir, seed=seed)
    return FileResponse(out_dir / file, media_type="application/jsonl", filename=f"{out_dir.name}-{file}")


# ------------------------------------------------------------------ results
@app.get("/api/results")
def results_list():
    d = config.EXPERIMENTS_DIR / "results"
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        r = json.loads(p.read_text())
        out.append({"experiment_id": r["experiment_id"], "label": r.get("label"), "meta": r.get("meta"), "arms": list(r["summary"])})
    return out


@app.get("/api/results/{eid}")
def results_get(eid: str):
    p = config.EXPERIMENTS_DIR / "results" / f"{eid}.json"
    if not p.exists():
        raise HTTPException(404, eid)
    return json.loads(p.read_text())


@app.get("/api/training/config")
def training_config():
    cfg = yaml.safe_load((config.CONFIG_DIR / "sft_lora.yaml").read_text())
    return {"config": cfg, "commands": {
        "score_pool": "backend/.venv/bin/python scripts/score_pool.py",
        "run_all": "backend/.venv/bin/python scripts/run_experiment.py --exp social-sft-v1",
        "single_arm": "backend/.venv/bin/python -m pgym.training.sft --data <dataset_dir> --out <adapter_dir> --seed 0",
        "dpo_extension": "backend/.venv/bin/python -m pgym.training.dpo --data <dataset_dir> --out <adapter_dir>",
    }}
