"""Compatibility shim for TRL >= 1.x on macOS / CPU.

TRL's SFTTrainer computes the loss with a fused LM head whose log-prob kernel
is written in Triton (Linux + CUDA only). On other platforms the import fails,
the symbol is left as ``None`` and training crashes. This shim installs a
plain-PyTorch implementation with the same contract. It materialises logits
only for the supervised (completion) tokens — one answer letter per example
here — so the memory cost is negligible. On Linux/CUDA the original kernel is
used unchanged.
"""
from __future__ import annotations

import torch

OPTIONAL_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")


class TorchChunkedLogProb:
    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None,
              logit_scale=1.0, outputs=OPTIONAL_OUTPUTS):
        logits = hidden @ weight.t()
        if bias is not None:
            logits = logits + bias
        logits = logits.float() * logit_scale
        if final_logit_softcapping is not None:
            logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
        logits = logits / temperature
        logp = torch.log_softmax(logits, dim=-1)
        lp = logp.gather(-1, targets[:, None]).squeeze(-1)
        with torch.no_grad():
            p = logp.exp()
            entropy = -(p * logp).sum(-1) if "entropy" in outputs else None
            lss = torch.log((p * p).sum(-1)) if "log_sum_sq_probs" in outputs else None
            mean_logits = logits.mean(-1) if "mean_logits" in outputs else None
            is_top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return lp, entropy, lss, mean_logits, is_top1


def patch() -> bool:
    import trl.trainer.utils as u
    if getattr(u, "_ChunkedLogProbFunction", None) is None:
        u._ChunkedLogProbFunction = TorchChunkedLogProb
        return True
    return False
