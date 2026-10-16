"""Batched local-model evaluation over pool / eval items.

For each item: interpretation distribution (averaged over all cyclic option
orders) and action distribution (averaged over `action_perms` cyclic
orders), then item metrics from evaluation.metrics.
"""
from __future__ import annotations

import time

import numpy as np
import yaml

from pgym.config import DATA_DIR
from pgym.evaluation.metrics import score_item
from pgym.scenarios.schema import Observable
from pgym.training.formats import LETTERS, chat_messages, cyclic


def _batched_letter_probs(tok, model, texts: list[str], n_opts: list[int], batch_size: int = 16) -> list[np.ndarray]:
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for j, row in enumerate(logits):
            n = n_opts[i + j]
            out.append(torch.softmax(row[letter_ids[:n]], dim=-1).cpu().numpy())
    return out


def score_items(tok, model, items: list[dict], action_perms: int = 3, include_context: bool = True,
                batch_size: int = 16, progress: bool = False) -> list[dict]:
    jobs = []  # (item_idx, kind, order, text, n)
    for idx, it in enumerate(items):
        obs = Observable(**it["observable"])
        ni, na = len(obs.interpretations), len(obs.actions)
        for s in range(ni):
            order = cyclic(ni, s)
            jobs.append((idx, "interpret", order, tok.apply_chat_template(chat_messages(obs, "interpret", order, include_context=include_context), tokenize=False, add_generation_prompt=True), ni))
        for s in range(min(action_perms, na)):
            order = cyclic(na, s * max(1, na // action_perms))
            jobs.append((idx, "act", order, tok.apply_chat_template(chat_messages(obs, "act", order, include_context=include_context), tokenize=False, add_generation_prompt=True), na))
    # sort by length for efficient padding, then restore
    order_idx = sorted(range(len(jobs)), key=lambda k: len(jobs[k][3]))
    t0 = time.time()
    probs_sorted = _batched_letter_probs(tok, model, [jobs[k][3] for k in order_idx], [jobs[k][4] for k in order_idx], batch_size)
    probs = [None] * len(jobs)
    for k, p in zip(order_idx, probs_sorted):
        probs[k] = p
    acc = {}
    for (idx, kind, order, _, n), p in zip(jobs, probs):
        a = acc.setdefault((idx, kind), np.zeros(n))
        cnt = acc.setdefault((idx, kind, "n"), np.zeros(1))
        for k, opt_i in enumerate(order):
            a[opt_i] += p[k]
        cnt += 1
    results = []
    for idx, it in enumerate(items):
        obs = it["observable"]
        ip = acc[(idx, "interpret")] / acc[(idx, "interpret", "n")]
        ap = acc[(idx, "act")] / acc[(idx, "act", "n")]
        ip = ip / ip.sum()
        ap = ap / ap.sum()
        ids = [o["id"] for o in obs["interpretations"]]
        aids = [a["id"] for a in obs["actions"]]
        probs_d = dict(zip(ids, ip.tolist()))
        action = aids[int(ap.argmax())]
        lit = next((o["id"] for o in obs["interpretations"] if o.get("literal")), None)
        sc = score_item(it["domain"], it["reference"], probs_d, action, lit)
        results.append({"id": it["id"], "probs": probs_d, "action": action, "action_probs": dict(zip(aids, ap.tolist())), "score": sc})
    if progress:
        print(f"scored {len(items)} items ({len(jobs)} forward rows) in {time.time() - t0:.1f}s")
    return results


def load_regression() -> list[dict]:
    return yaml.safe_load((DATA_DIR / "eval" / "regression.yaml").read_text())


def score_regression(tok, model, batch_size: int = 16) -> dict:
    items = load_regression()
    texts, ns, meta = [], [], []
    for qi, it in enumerate(items):
        n = len(it["options"])
        for s in range(n):
            order = cyclic(n, s)
            lines = "\n".join(f"{LETTERS[k]}. {it['options'][i]}" for k, i in enumerate(order))
            msgs = [{"role": "user", "content": f"{it['q']}\n{lines}\nAnswer with a single letter."}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            ns.append(n)
            meta.append((qi, order))
    probs = _batched_letter_probs(tok, model, texts, ns, batch_size)
    agg = {}
    for (qi, order), p in zip(meta, probs):
        a = agg.setdefault(qi, np.zeros(len(order)))
        for k, i in enumerate(order):
            a[i] += p[k]
    correct = [float(int(agg[qi].argmax()) == items[qi]["answer"]) for qi in agg]
    p_correct = [float(agg[qi][items[qi]["answer"]] / agg[qi].sum()) for qi in agg]
    return {"accuracy": float(np.mean(correct)), "p_correct": float(np.mean(p_correct)), "n": len(items),
            "per_item_correct": correct}
