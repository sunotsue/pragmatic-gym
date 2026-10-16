"""Factorial designs over social-context factors.

A design holds the utterance (family) fixed, crosses the levels of 1–4 chosen
factors (all others pinned at their base level), and replicates every cell
over `n_variants` surface variants (paraphrase choice, sentence order, names,
utterance surface form). Variants are the unit of clustering for inference:
the same variant id renders the *same* names and wording choices across
cells, so cell contrasts within a variant are paired comparisons.

What these experiments estimate: the causal effect of a prompt-level
intervention on *model behavior*. They say nothing about how real people
would interpret the same messages — that requires human data, which the
analysis reports separately when available.
"""
from __future__ import annotations

import hashlib
import itertools
import json

from pydantic import BaseModel, Field

from pgym.social.factors import FACTOR_BY_ID, violations
from pgym.social.generator import FAMILY_BY_ID, cell_key, full_levels, render

MAX_CELLS = 64


class DesignSpec(BaseModel):
    family_id: str = "inv_let_you_know"
    base: dict[str, str] = Field(default_factory=dict)
    vary: dict[str, list[str]] = Field(default_factory=lambda: {"shared_history": ["unspecified", "repeated_disappointment", "established_trust"],
                                                              "audience": ["private", "group_chat"]})
    n_variants: int = 3
    reps: int = 1
    outcome_intent: str | None = None  # default: domain-specific key intent
    seed: int = 0

    def design_id(self) -> str:
        return hashlib.sha1(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:10]


KEY_INTENT = {"invitation": "soft_decline", "relationship": "avoiding_talk", "emotional": "hurt_face_saving", "family": "indirect_request"}


def build(spec: DesignSpec) -> dict:
    fam = FAMILY_BY_ID[spec.family_id]
    for f, levels in spec.vary.items():
        fac = FACTOR_BY_ID[f]
        if fam.domain not in fac.domains:
            raise ValueError(f"Factor {f} does not apply to the {fam.domain} domain")
        for lv in levels:
            if lv not in fac.level_ids:
                raise ValueError(f"Unknown level {lv} for {f}")
    names = list(spec.vary)
    cells, excluded = [], []
    for combo in itertools.product(*[spec.vary[n] for n in names]):
        lv = full_levels(fam.domain, {**spec.base, **dict(zip(names, combo))})
        bad = violations(lv, fam.domain)
        assign = dict(zip(names, combo))
        if bad:
            excluded.append({"assignment": assign, "reasons": bad})
        else:
            cells.append({"cell": cell_key(lv), "assignment": assign, "levels": lv})
    if len(cells) > MAX_CELLS:
        raise ValueError(f"Design has {len(cells)} cells; limit is {MAX_CELLS}.")
    units = []
    for c in cells:
        for v in range(spec.n_variants):
            scen = render(spec.family_id, c["levels"], variant=v)
            for r in range(spec.reps):
                units.append({"cell": c["cell"], "assignment": c["assignment"], "variant": v, "rep": r,
                              "scenario_id": scen.id, "reference": scen.reference.distribution})
    return {
        "design_id": spec.design_id(),
        "spec": spec.model_dump(),
        "domain": fam.domain,
        "outcome_intent": spec.outcome_intent or KEY_INTENT[fam.domain],
        "factors": names,
        "cells": cells,
        "excluded": excluded,
        "units": units,
        "n_units": len(units),
    }


def preview(spec: DesignSpec, cell_index: int = 0, variant: int = 0):
    d = build(spec)
    c = d["cells"][cell_index]
    return render(spec.family_id, c["levels"], variant=variant)
