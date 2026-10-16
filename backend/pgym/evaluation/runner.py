"""Run a model over scenarios and score it."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pgym.evaluation.metrics import score_item, summarize
from pgym.models.adapters import Adapter
from pgym.scenarios.schema import Scenario


def run_scenario(adapter: Adapter, scen: Scenario) -> dict:
    out = adapter.interpret(scen.observable())
    lit = next((o.id for o in scen.interpretations if o.literal), None)
    sc = score_item(scen.domain, scen.reference.distribution, out.interpretation_probs, out.action, lit)
    return {"scenario_id": scen.id, "family_id": scen.family_id, "capability": scen.capability, "domain": scen.domain,
            "split": scen.split, "counterfactual_of": scen.counterfactual_of,
            "output": out.to_dict(), "score": sc}


def run_many(adapter: Adapter, scenarios: list[Scenario], workers: int = 1) -> list[dict]:
    if workers <= 1 or adapter.name in ("local_hf",):
        return [run_scenario(adapter, s) for s in scenarios]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda s: run_scenario(adapter, s), scenarios))


def report(rows: list[dict]) -> dict:
    scores = [r["score"] for r in rows]
    by_cap: dict[str, list] = {}
    for r in rows:
        by_cap.setdefault(r["capability"], []).append(r["score"])
    return {"overall": summarize(scores), "by_capability": {k: summarize(v) for k, v in by_cap.items()}}
