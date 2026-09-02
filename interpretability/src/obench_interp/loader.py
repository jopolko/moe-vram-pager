"""Model + SAE loading. HF transformers -> TransformerLens, fp16, CPU fallback.

Kept deliberately small: one function to get a HookedTransformer, one to get a
Gemma Scope SAE for a layer. No shared runtime with the llama.cpp path.
"""
from __future__ import annotations

import gc
import os

from .config import DEVICE, DTYPE, ModelConfig
from .env import HF_CACHE

# Pin the HF cache inside this module (gitignored), not ~/.cache.
os.environ.setdefault("HF_HOME", str(HF_CACHE))


def load_model(cfg: ModelConfig | None = None, device: str | None = None):
    """Return a TransformerLens HookedTransformer for cfg.

    gemma-2 needs eager attention; TransformerLens applies the attn/final
    logit soft-capping itself.

    TransformerLens otherwise materializes the HF weights in fp32 on CPU before
    converting, a ~15 GB transient peak that OOMs a 16 GB WSL box. We preload the
    HF model ourselves in fp16 with low_cpu_mem_usage, hand it to TL, then free
    it. fp16 (not bf16): the Pascal target has no hardware bf16.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    cfg = cfg or ModelConfig()
    device = device or DEVICE
    tl_dtype = DTYPE if device == "cuda" else torch.float16

    def build(dev: str, dtype: torch.dtype):
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        try:
            m = HookedTransformer.from_pretrained(
                cfg.tl_name,
                hf_model=hf_model,
                dtype=dtype,
                device=dev,
                attn_implementation="eager",
                # Keep the residual stream faithful: do not fold in the norms.
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
        finally:
            del hf_model
            gc.collect()
        return m

    try:
        model = build(device, tl_dtype)
    except torch.cuda.OutOfMemoryError:
        print("[loader] CUDA OOM during load; falling back to CPU (slow).")
        torch.cuda.empty_cache()
        model = build("cpu", torch.float16)

    model.eval()
    return model


def load_sae(cfg: ModelConfig | None = None, device: str | None = None):
    """Return a Gemma Scope SAE for cfg.sae_release / cfg.sae_id.

    sae_lens changed its return signature across majors; handle tuple or object.
    """
    from sae_lens import SAE

    cfg = cfg or ModelConfig()
    device = device or DEVICE
    if not cfg.sae_release:
        raise ValueError(f"No public SAE configured for {cfg.hf_name}; train one with sae_lens.")

    out = SAE.from_pretrained(cfg.sae_release, cfg.sae_id, device=device)
    sae = out[0] if isinstance(out, tuple) else out
    sae.eval()
    return sae
