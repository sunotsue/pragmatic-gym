"""EXTENSION (not part of the headline experiment): DPO on move preferences.

Uses `preference.jsonl` exported by the Data Intervention Lab: chosen = the
best move under the reference uncertainty, rejected = the base model's own
(worse) move or, failing that, the most harmful move. Kept separate from the
SFT comparison so the main result is not confounded by a different objective.

    backend/.venv/bin/python -m pgym.training.dpo --data <dataset_dir> --out <adapter_dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def train(data_dir: str, out_dir: str, seed: int = 0, smoke: bool = False, beta: float = 0.1) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from pgym.training.sft import load_config
    from pgym.training.trl_compat import patch
    patch()
    cfg = load_config()
    t0 = time.time()
    ds = load_dataset("json", data_files=str(Path(data_dir) / "preference.jsonl"), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")])
    if smoke:
        ds = ds.select(range(min(8, len(ds))))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.float32)
    lc = cfg["lora"]
    args = DPOConfig(output_dir=str(Path(out_dir) / "trainer"), num_train_epochs=1, max_steps=2 if smoke else -1,
                     learning_rate=5e-6 if not smoke else 1e-5, per_device_train_batch_size=4, beta=beta,
                     logging_steps=5, save_strategy="no", report_to="none", seed=seed, dataloader_pin_memory=False)
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                                target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    res = trainer.train()
    trainer.model.save_pretrained(out_dir)
    meta = {"objective": "dpo", "beta": beta, "n": len(ds), "steps": res.global_step, "loss": res.training_loss,
            "runtime_s": round(time.time() - t0, 1), "smoke": smoke}
    (Path(out_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(json.dumps(train(a.data, a.out, a.seed, a.smoke), indent=2))
