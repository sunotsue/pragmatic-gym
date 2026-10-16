"""API contract smoke tests (no model inference: fixture backend only)."""
from fastapi.testclient import TestClient

from pgym.api.main import app

c = TestClient(app)
R = {"x-researcher-token": "local-researcher"}


def test_meta_and_scenarios():
    m = c.get("/api/meta").json()
    assert len(m["factors"]) == 13 and len(m["capabilities"]) == 8 and len(m["envs"]) == 5  # 12 social factors + placebo
    lst = c.get("/api/scenarios").json()
    assert len(lst) >= 50


def test_latent_requires_researcher_token():
    assert "latent" not in c.get("/api/scenarios/soft_no_birthday").json()
    assert "latent" in c.get("/api/scenarios/soft_no_birthday", headers=R).json()
    assert c.get("/api/diagnosis").status_code == 403


def test_fixture_inference_is_labeled():
    r = c.post("/api/infer", json={"scenario_id": "soft_no_birthday", "backend": "fixture"}, headers=R).json()
    p = r["output"]["provenance"]
    assert p["is_real_inference"] is False and "FIXTURE" in p["label"]
    assert "score" in r


def test_counterfactual_design_and_fixture_run():
    spec = {"family_id": "inv_let_you_know", "vary": {"prior_pattern": ["none", "repeated"]}, "n_variants": 3}
    d = c.post("/api/counterfactual/design", json=spec).json()
    assert len(d["cells"]) == 2 and d["n_units"] == 6
    r = c.post("/api/counterfactual/run", json={"spec": spec, "backend": "fixture"}).json()
    assert r["provenance"]["is_real_inference"] is False
    assert r["analysis"]["main_effects"]


def test_env_session_flow():
    s = c.post("/api/envs/soft_no/reset", json={"seed": 1}, headers=R).json()
    sid = s["session_id"]
    assert "debug" in s
    out = c.post(f"/api/envs/sessions/{sid}/step", json={"action": "clarify_low_pressure"}, headers=R).json()
    assert "reward" in out and out["debug"]["log"]
    # without the token, reward internals and hidden state are not returned
    s2 = c.post("/api/envs/soft_no/reset", json={"seed": 1}).json()
    o2 = c.post(f"/api/envs/sessions/{s2['session_id']}/step", json={"action": "pressure"}).json()
    assert "reward" not in o2 and "debug" not in o2
    b = c.post(f"/api/envs/sessions/{sid}/agent_step", json={"policy": "bayes"}, headers=R)
    assert b.status_code in (200, 400)


def test_recorded_mode_never_runs_a_model(monkeypatch):
    from pgym import config
    from pgym.models import recorded
    monkeypatch.setattr(config, "LIVE", False)
    monkeypatch.setattr(recorded, "_store", {})  # nothing recorded
    s = c.get("/api/status").json()
    assert s["default_backend"] == "recorded" and [b["backend"] for b in s["backends"]] == ["recorded"]
    # even when a request names a live backend, recorded mode answers from the store (here: a clear 409)
    r = c.post("/api/infer", json={"scenario_id": "soft_no_birthday", "backend": "local_hf"}, headers=R)
    assert r.status_code == 409 and "Not recorded" in r.json()["detail"]
    spec = {"family_id": "inv_let_you_know", "vary": {"audience": ["private", "group_chat"]}, "n_variants": 2}
    r = c.post("/api/counterfactual/run", json={"spec": spec})
    assert r.status_code == 409 and "wasn't recorded" in r.json()["detail"]


def test_json_safe_replaces_nan():
    from pgym.api.main import _json_safe
    assert _json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": "x"}) == {"a": None, "b": [1.0, None], "c": "x"}


def test_sessions_are_stateless_tokens():
    for seed in range(50):  # find an episode that lasts at least two turns
        s = c.post("/api/envs/soft_no/reset", json={"seed": seed}, headers=R).json()
        a = c.post(f"/api/envs/sessions/{s['session_id']}/step", json={"action": "acknowledge_wait"}, headers=R).json()
        if not a["terminated"]:
            break
    b = c.post(f"/api/envs/sessions/{a['session_id']}/agent_step", json={"policy": "bayes"}, headers=R).json()
    # replaying the token on a fresh lookup reproduces the same state (works across server instances)
    assert c.get(f"/api/envs/sessions/{b['session_id']}/debug", headers=R).json()["log"] == b["debug"]["log"]
    assert len(b["debug"]["log"]) == 2
    assert c.post("/api/envs/sessions/not-a-token/step", json={"action": "pressure"}).status_code == 404
