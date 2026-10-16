"""Local Hugging Face model backend (base checkpoint or base + LoRA adapter).

Reads the model's next-token distribution over option letters, averaged over
all cyclic option orders (removes position bias). This is real inference with
a real model; provenance records the checkpoint and adapter.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

from pgym.models.adapters import Adapter, InterpretationOutput, Provenance
from pgym.models.prompts import PROMPT_VERSION
from pgym.scenarios.schema import Observable
from pgym.training.formats import FORMAT_VERSION, LETTERS, chat_messages, cyclic

DEFAULT_LOCAL_MODEL = os.environ.get("PGYM_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=4)
def load(model_id: str, adapter_path: str | None = None, dtype: str = "float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device()).eval()
    return tok, model


def letter_ids(tok, n: int) -> list[int]:
    ids = []
    for L in LETTERS[:n]:
        t = tok.encode(L, add_special_tokens=False)
        ids.append(t[0])
    return ids


def option_distribution(tok, model, obs: Observable, kind: str, n_perms: int | None = None, include_context: bool = True) -> list[float]:
    """P(option) averaged over cyclic permutations. Returned in canonical option order."""
    import torch
    n = len(obs.interpretations if kind == "interpret" else obs.actions)
    perms = [cyclic(n, s) for s in range(n if n_perms is None else min(n, n_perms))]
    lids = letter_ids(tok, n)
    texts = [tok.apply_chat_template(chat_messages(obs, kind, order, include_context=include_context),
                                     tokenize=False, add_generation_prompt=True) for order in perms]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
    probs = torch.softmax(logits[:, lids].float(), dim=-1).cpu().numpy()
    out = [0.0] * n
    for row, order in zip(probs, perms):
        for k, idx in enumerate(order):
            out[idx] += float(row[k]) / len(perms)
    return out


def generate_reply(tok, model, obs: Observable, action_label: str, max_new_tokens: int = 48) -> str:
    import torch
    from pgym.models.prompts import SYSTEM, render_conversation
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{render_conversation(obs)}\n\nWrite the next message you would send. Your approach: {action_label}. Reply with the message only."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class LocalHFAdapter(Adapter):
    name = "local_hf"

    def __init__(self, model_id: str | None = None, adapter_path: str | None = None, label: str | None = None, with_reply: bool = False):
        self.model_id = model_id or DEFAULT_LOCAL_MODEL
        self.adapter_path = adapter_path
        self.tag = label or (self.model_id.split("/")[-1] + (" + LoRA" if adapter_path else ""))
        self.with_reply = with_reply

    def available(self) -> bool:
        # Cached → ready. Not cached but torch/transformers installed and online → the first call downloads it.
        from importlib.util import find_spec
        if find_spec("torch") is None or find_spec("transformers") is None:
            return False
        try:
            from huggingface_hub import try_to_load_from_cache
            if try_to_load_from_cache(self.model_id, "config.json") is not None:
                return True
        except Exception:
            return False
        return os.environ.get("HF_HUB_OFFLINE", "0") in ("", "0")

    def interpret(self, obs: Observable) -> InterpretationOutput:
        t0 = time.time()
        tok, model = load(self.model_id, self.adapter_path)
        ip = option_distribution(tok, model, obs, "interpret")
        ap = option_distribution(tok, model, obs, "act")
        probs = {o.id: p for o, p in zip(obs.interpretations, ip)}
        action = obs.actions[max(range(len(ap)), key=lambda i: ap[i])]
        reply = generate_reply(tok, model, obs, action.label) if self.with_reply else action.example
        reply_source = "generated" if self.with_reply else "action template (the model chose the move; wording is the move's fixed template)"
        top = max(probs, key=probs.get)
        prov = Provenance(backend=self.name, model=self.tag, is_real_inference=True,
                          label=f"Live local inference · {self.tag}", prompt_version=f"{PROMPT_VERSION}/{FORMAT_VERSION}",
                          latency_ms=int((time.time() - t0) * 1000),
                          settings={"model_id": self.model_id, "adapter": self.adapter_path, "scoring": "next-token letter distribution, averaged over cyclic option orders",
                                    "action_probs": {a.id: p for a, p in zip(obs.actions, ap)}, "reply_source": reply_source})
        return InterpretationOutput(interpretation_probs=probs, most_likely=top,
                                    rationale="Distribution read from the model's next-token probabilities over option letters (no verbalized reasoning).",
                                    action=action.id, reply=reply, provenance=prov)

    def describe(self) -> dict:
        return {**super().describe(), "model": self.tag, "model_id": self.model_id, "adapter": self.adapter_path}
