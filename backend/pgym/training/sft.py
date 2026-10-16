"""LoRA SFT with TRL on an exported Pragmatic Gym dataset.

    backend/.venv/bin/python -m pgym.training.sft --data experiments/runs/<arm>_s0/data \
        --out experiments/runs/<arm>_s0/adapter --seed 0

Works on CPU, Apple MPS or CUDA (device is picked by transformers). The
`--smoke` flag trains on 16 examples for 2 steps (used by the test suite).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from pgym.config import CONFIG_DIR


def load_config(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "sft_lora.yaml").read_text())


def train(data_dir: str, out_dir: str, seed: int = 0, config: dict | None = None, smoke: bool = False) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from pgym.training.trl_compat import patch
    patched = patch()
    cfg = config or load_config()
    t0 = time.time()
    ds = load_dataset("json", data_files=str(Path(data_dir) / "sft.jsonl"), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "completion")])
    ds = ds.shuffle(seed=seed)
    if smoke:
        ds = ds.select(range(min(16, len(ds))))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.float32)
    lc = cfg["lora"]
    tc = cfg["train"]
    peft_config = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                             target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=str(Path(out_dir) / "trainer"),
        num_train_epochs=1 if smoke else tc["epochs"],
        max_steps=2 if smoke else -1,
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"],
        warmup_steps=tc["warmup_ratio"],  # transformers>=5: a float < 1 is a ratio of total steps
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["grad_accum"],
        weight_decay=tc["weight_decay"],
        max_length=tc["max_length"],
        completion_only_loss=tc["completion_only_loss"],
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    result = trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    log = [x for x in trainer.state.log_history if "loss" in x]
    n_tokens = sum(len(tok(tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False))["input_ids"]) for r in ds)
    meta = {"base_model": cfg["base_model"], "seed": seed, "n_examples": len(ds), "train_tokens": n_tokens,
            "epochs": args.num_train_epochs, "steps": result.global_step, "train_loss": result.training_loss,
            "loss_curve": [{"step": x["step"], "loss": x["loss"]} for x in log],
            "runtime_s": round(time.time() - t0, 1), "lora": lc, "trl_torch_logprob_shim": patched, "train_config": tc, "smoke": smoke,
            "dataset_manifest": json.loads((Path(data_dir) / "manifest.json").read_text()) if (Path(data_dir) / "manifest.json").exists() else None}
    (Path(out_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(json.dumps(train(a.data, a.out, a.seed, load_config(a.config), a.smoke), indent=2))
