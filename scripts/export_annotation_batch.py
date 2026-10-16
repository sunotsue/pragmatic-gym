"""Export a human-annotation batch for a counterfactual design (observable text only).

    backend/.venv/bin/python scripts/export_annotation_batch.py --family inv_let_you_know \
        --vary shared_history=unspecified,repeated_disappointment,established_trust --vary audience=private,group_chat \
        --variants 3 --out data/eval/batch_soft_no.jsonl

Each row contains only what a model would see, items are shuffled, and the
answer format is a 100-point allocation over interpretations (so annotators
can express genuine ambiguity). Aggregate into human_annotations.jsonl
following human_annotations.schema.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pgym.causal.design import DesignSpec, build  # noqa: E402
from pgym.models.prompts import render_conversation  # noqa: E402
from pgym.social.generator import render  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True)
ap.add_argument("--vary", action="append", default=[])
ap.add_argument("--variants", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
vary = {k: v.split(",") for k, v in (x.split("=") for x in a.vary)}
d = build(DesignSpec(family_id=a.family, vary=vary, n_variants=a.variants))
rows = []
for c in d["cells"]:
    for v in range(a.variants):
        s = render(a.family, c["levels"], v)
        o = s.observable()
        rows.append({"item_id": s.id, "family_id": a.family, "cell": c["cell"], "conversation": render_conversation(o),
                     "question": o.question, "options": [{"id": i.id, "label": i.label, "description": i.description} for i in o.interpretations],
                     "instructions": "Distribute 100 points across the readings according to how likely each is. Use several if the message is genuinely ambiguous."})
random.Random(0).shuffle(rows)
Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"{len(rows)} items -> {a.out}")
